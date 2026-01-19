import gc
import os
import sys
from collections import deque
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
from tqdm.notebook import tqdm

from models.algorithms.utils import checker, plot_statistics, tensor
from models.neural_networks.lstm import LSTM
from settings import DEVICE, DIR


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent using LSTM-based actor-critic architecture
    for hybrid action spaces (discrete + continuous).

    Args:
        num_features (int): Number of features in each observation.
        num_actions (int): Number of discrete actions.
        cont_actions (int): Number of continuous action dimensions.
        memory_size (int): Number of past observations stored for LSTM input.
        replay_buffer_size (int): Number of transitions stored before training update.
        hidden_layers (int): Number of LSTM layers.
        hidden_units (int): Number of hidden units per LSTM layer.
        dropout (float): Dropout rate for neural networks.
        actor_lr (float | np.float32): Learning rate for the actor network.
        critic_lr (float | np.float32): Learning rate for the critic network.
        advantage_type (str): Advantage estimation method ('simple' or 'gae').
        gamma (float | np.float32): Discount factor.
        lam (float | np.float32): GAE lambda parameter.
        clip_ratio (float | np.float32): PPO clipping ratio.
        grad_norm (float | None): Maximum gradient norm for clipping.
        opt_epochs (int): Number of optimization epochs per update.
        c1 (float | np.float32): Value loss coefficient.
        c2 (float | np.float32): Entropy bonus coefficient.
        debug (bool): Enables debug output and tensor checks.
    """
    def __init__(
            self, num_features: int,
            num_actions: int,
            cont_actions: int,
            memory_size: int = 3,
            replay_buffer_size: int = 8096,
            hidden_layers: int = 16,
            hidden_units: int = 32,
            dropout: float = 0.2,
            actor_lr: float | np.float32 = 1e-4,
            critic_lr: float | np.float32 = 5e-4,
            advantage_type: str = 'gae',
            gamma: float | np.float32=0.99,
            lam: float | np.float32 = 0.95,
            clip_ratio: float | np.float32 = 0.2,
            grad_norm: float | None = 0.5,
            opt_epochs: int = 10,
            c1: float | np.float32 = 0.05,
            c2: float | np.float32 = 0.01,
            debug: bool = False,
        ):

        self.opt_epochs = opt_epochs
        self.clip_ratio = clip_ratio
        self.memory_size = memory_size
        self.num_actions = num_actions
        self.cont_actions = cont_actions
        self.num_features = num_features
        self.advantage_type = advantage_type
        self.replay_buffer_size = replay_buffer_size

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = LSTM(
            timestamps=memory_size,
            features=num_features,
            dropout=dropout,
            output_shape=4,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            lr=actor_lr,
            debug=debug
        ).to(DEVICE)
        self.critic = LSTM(
            timestamps=memory_size,
            features=num_features,
            dropout=dropout,
            output_shape=1,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            lr=critic_lr,
            debug=debug
        ).to(DEVICE)

        self.c1 = c1
        self.c2 = c2
        self.lam = lam
        self.gamma = gamma
        self.debug = debug
        self.grad_norm = grad_norm

        self.EPS = 1e-5
        self.LOG_STD_MIN = -6.0
        self.LOG_STD_MAX = 0.0


    def save(self, save_path: str = 'models_params') -> None:
        """
        Saves actor and critic model parameters to disk.

        Args:
            save_path (str): Directory name for saving model parameters.

        Returns:
            None
        """
        intermediate_path = os.path.join('models', 'algorithms')
        self.actor.save(path=os.path.join(DIR, intermediate_path, save_path, 'ppo', 'actor.pth'))
        self.critic.save(path=os.path.join(DIR, intermediate_path, save_path, 'ppo', 'critic.pth'))


    def act(self, observation: np.ndarray) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Computes an action and value estimate from the current observation.

        Args:
            observation (np.ndarray): Current environment observation window.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
                Continuous log probabilities,
                discrete action logits,
                continuous action,
                discrete action,
                value estimate.
        """
        with torch.no_grad():
            mixed = self.actor.forward(observation.reshape(1, *observation.shape))

            logits_disc_open, logits_disc_close, mean, log_std = torch.split(mixed, 1, dim=1)
            logits_disc = torch.concat([logits_disc_open, logits_disc_close], dim=1)

            action_distr = torch.distributions.Categorical(logits=logits_disc)
            action_disc = torch.squeeze(action_distr.sample()).cpu().numpy()

            action_cont, cont_log_probs, _, _ = self.compute_cont_action(mean=mean, log_std=log_std)
            action_cont = action_cont.cpu().numpy()

            value = torch.squeeze(self.critic.forward(observation.reshape(1, *observation.shape)))

            if self.debug:
                checker(mixed, 'mixed-act')
                checker(value, 'value-act')

        return cont_log_probs.detach(), logits_disc.detach(), action_cont, action_disc, value.detach()


    def compute_cont_log_probs(self, actions: np.ndarray | torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """
        Computes log probabilities of continuous actions under a squashed Gaussian policy.

        Args:
            actions (np.ndarray | torch.Tensor): Continuous actions taken.
            mean (torch.Tensor): Mean of the Gaussian distribution.
            log_std (torch.Tensor): Log standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: Log probabilities of the continuous actions.
        """
        actions = torch.reshape(actions, shape=(actions.shape[0], 1))

        log_std_clipped = torch.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std_clipped) + self.EPS

        actions_clipped = torch.clip(actions, -1.0 + self.EPS, 1.0 - self.EPS)
        u = torch.atanh(actions_clipped)

        quad = ((u - mean) ** 2) / (std ** 2) + self.EPS

        cont_log_probs = -0.5 * torch.sum(quad + 2.0 * log_std_clipped + torch.log(tensor(2 * np.pi)), dim=1, keepdims=True)
        jacobian = torch.log(tensor(1.0 - actions_clipped ** 2 + self.EPS))

        cont_log_probs = cont_log_probs - jacobian

        if self.debug:
            checker(cont_log_probs, 'cont_log_probs-compute_cont_log_probs')
            checker(jacobian, 'jacobian-compute_cont_log_probs')
            checker(actions, 'actions-compute_cont_log_probs')
            checker(quad, 'quad-compute_cont_log_probs')
            checker(std, 'std-compute_cont_log_probs')
            checker(u, 'u-compute_cont_log_probs')

        return cont_log_probs


    def compute_cont_action(self, mean, log_std) -> tuple[torch.Tensor]:
        """
        Samples continuous actions from a squashed Gaussian policy and computes log probabilities.

        Args:
            mean (torch.Tensor): Mean of the Gaussian distribution.
            log_std (torch.Tensor): Log standard deviation of the Gaussian distribution.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Sampled continuous actions,
                log probabilities,
                mean,
                standard deviation.
        """
        log_std_clipped = torch.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std_clipped) + self.EPS

        norm_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
        z = norm_dist.sample(sample_shape=(self.cont_actions,)).to(DEVICE)
        u = mean + std * z

        action_cont = torch.squeeze(torch.tanh(u))

        cont_log_probs = -0.5 * torch.sum((z ** 2) + 2.0 * log_std_clipped + torch.log(tensor(2 * np.pi)), dim=1, keepdims=True)
        jacobian = torch.log(tensor(1.0 - action_cont ** 2 + self.EPS))
        cont_log_probs = cont_log_probs - jacobian

        if self.debug:
            checker(cont_log_probs, 'cont_log_probs-compute_cont_action')
            checker(action_cont, 'action_cont-compute_cont_action')
            checker(jacobian, 'jacobian-compute_cont_action')
            checker(std, 'std-compute_cont_action')
            checker(z, 'z-compute_cont_action')

        return action_cont, cont_log_probs, mean, std


    def discount_reward(self, rewards: torch.tensor, dones: torch.tensor) -> torch.tensor:
        """
        Computes discounted returns for a batch of rewards.

        Args:
            rewards (torch.tensor): Reward sequence.
            dones (torch.tensor): Episode termination flags.

        Returns:
            torch.tensor: Discounted return values.
        """
        result = []
        discounted_sum = 0.0

        for reward, done in zip(torch.flip(rewards, dims=[0]), torch.flip(dones, dims=[0])):
            if done:
                discounted_sum = 0.0

            discounted_sum = reward + self.gamma * discounted_sum
            result.append(discounted_sum)

        result = tensor(result[::-1], dtype=torch.float32)

        return result


    def standardize(self, values):
        """
        Normalizes a tensor to zero mean and unit variance.

        Args:
            values (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Standardized tensor.
        """
        return (values - torch.mean(values)) / (torch.std(values) + 1e-8)


    def simple_advantages(self, returns, values) -> torch.Tensor:
        advantages = returns - values
        return self.standardize(advantages)


    def gae_advantages(self, rewards, values, dones, states) -> torch.tensor:
        """
        Computes Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor): Reward sequence.
            values (torch.Tensor): Value estimates.
            dones (torch.Tensor): Episode termination flags.
            states (torch.Tensor): State sequence (unused, kept for API consistency).

        Returns:
            torch.Tensor: Standardized GAE advantages.
        """
        T = rewards.shape[0]
        dones = dones.int()
        advantages = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        gae = .0
        next_value = .0
        if not dones[-1]:
            next_value = values[-1]

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            next_value = values[t]
            advantages[t] = gae

        advantages = self.standardize(advantages)

        return advantages


    def get_advantages(self, rewards, values, dones, states) -> torch.Tensor:
        """
        Selects and computes advantages using the configured method.

        Args:
            rewards (torch.Tensor): Reward sequence.
            values (torch.Tensor): Value estimates.
            dones (torch.Tensor): Episode termination flags.
            states (torch.Tensor): State sequence.

        Returns:
            torch.Tensor: Computed advantage values.
        """
        if self.advantage_type == 'simple':
            return self.simple_advantages(rewards, values)
        return self.gae_advantages(rewards, values, dones, states)


    def _compute_actor_loss(self, new_log_probs, old_log_probs, new_logits_disc, old_logits_disc, log_std_new, actions_disc, value_loss, advantages):
        """
        Computes PPO actor loss for hybrid (discrete + continuous) action spaces.

        Args:
            new_log_probs (torch.Tensor): Log probabilities from current policy.
            old_log_probs (torch.Tensor): Log probabilities from old policy.
            new_logits_disc (torch.Tensor): Current discrete action logits.
            old_logits_disc (torch.Tensor): Old discrete action logits.
            log_std_new (torch.Tensor): Log standard deviation for entropy bonus.
            actions_disc (torch.Tensor): Discrete actions taken.
            value_loss (torch.Tensor): Critic loss term.
            advantages (torch.Tensor): Advantage estimates.

        Returns:
            torch.Tensor: Actor loss value.
        """
        actions_one_hot = torch.nn.functional.one_hot(actions_disc, self.num_actions).to(torch.float32)

        new_policy_disc = torch.softmax(new_logits_disc, dim=-1)
        old_policy_disc = torch.softmax(old_logits_disc, dim=-1)

        new_action_porbs_disc = torch.sum(actions_one_hot * new_policy_disc, dim=1)
        old_action_porbs_disc = torch.sum(actions_one_hot * old_policy_disc, dim=1)

        ratio_disc = torch.exp(torch.log(new_action_porbs_disc + 1e-10) - torch.log(old_action_porbs_disc + 1e-10))
        ratio_cont = torch.exp(new_log_probs - old_log_probs)
        ratio_cont = torch.squeeze(ratio_cont)

        clipped_disc = torch.clip(ratio_disc, 1 - self.clip_ratio, 1 + self.clip_ratio)
        clipped_cont = torch.clip(ratio_cont, 1 - self.clip_ratio, 1 + self.clip_ratio)

        policy_loss_disc = -torch.mean(torch.min(ratio_disc * advantages, clipped_disc * advantages))
        policy_loss_cont = -torch.mean(torch.min(ratio_cont * advantages, clipped_cont * advantages))

        entropy_bonus_disc = -torch.mean(torch.sum(new_policy_disc * torch.log(new_policy_disc + 1e-10), dim=1))
        entropy_bonus_cont = torch.mean(torch.sum(log_std_new + 0.5 * (1. + np.log(2 * np.pi)), dim=1))

        objective = -policy_loss_cont - policy_loss_disc + self.c1 * value_loss + self.c2 * (entropy_bonus_disc + entropy_bonus_cont)

        actor_loss = policy_loss_disc + policy_loss_cont - self.c2 * (entropy_bonus_disc + entropy_bonus_cont)

        if self.debug:
            checker(actions_one_hot, 'actions_one_hot-_compute_actor_loss')
            checker(new_policy_disc, 'new_policy_disc-_compute_actor_loss')
            checker(old_policy_disc, 'old_policy_disc-_compute_actor_loss')
            checker(new_action_porbs_disc, 'new_action_porbs_disc-_compute_actor_loss')
            checker(old_action_porbs_disc, 'old_action_porbs_disc-_compute_actor_loss')
            checker(ratio_disc, 'ratio_disc-_compute_actor_loss')
            checker(ratio_cont, 'ratio_cont-_compute_actor_loss')
            checker(policy_loss_disc, 'policy_loss_disc-_compute_actor_loss')
            checker(policy_loss_cont, 'policy_loss_cont-_compute_actor_loss')
            checker(entropy_bonus_disc, 'entropy_bonus_disc-_compute_actor_loss')
            checker(entropy_bonus_cont, 'entropy_bonus_cont-_compute_actor_loss')
            checker(objective, 'objective-_compute_actor_loss')

        return actor_loss


    def _apply_grad(
            self,
            states: torch.Tensor,
            returns: torch.Tensor,
            advantages: torch.Tensor,
            actions_cont: torch.Tensor,
            actions_disc: torch.Tensor,
            old_log_probs: torch.Tensor,
            old_logits: torch.Tensor,
        ):
        """
        Performs a gradient update step for actor and critic networks.

        Args:
            states (torch.Tensor): State batch.
            returns (torch.Tensor): Discounted returns.
            advantages (torch.Tensor): Advantage values.
            actions_cont (torch.Tensor): Continuous actions.
            actions_disc (torch.Tensor): Discrete actions.
            old_log_probs (torch.Tensor): Previous log probabilities.
            old_logits (torch.Tensor): Previous discrete logits.

        Returns:
            torch.Tensor: Total objective value after update.
        """
        values = torch.squeeze(self.critic.forward(states))
        critic_loss = torch.mean(torch.square(returns - values))

        mixed = self.actor.forward(states)
        logits_open, logits_close, mean, log_std = torch.split(mixed, 1, dim=1)
        logits = torch.concat([logits_open, logits_close], dim=1)

        log_probs = self.compute_cont_log_probs(actions_cont, mean, log_std)

        actor_loss = self._compute_actor_loss(
            new_log_probs=log_probs,
            old_log_probs=old_log_probs,

            new_logits_disc=logits,
            old_logits_disc=old_logits,

            log_std_new=log_std,
            actions_disc=actions_disc,

            value_loss=critic_loss,
            advantages=advantages
        )

        objective = actor_loss + self.c1 * critic_loss

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        objective.backward()

        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=self.grad_norm
            )

        self.actor.optimizer.step()
        self.critic.optimizer.step()

        if self.debug:
            checker(critic_loss, 'critic_loss-_apply_grad')
            checker(advantages, 'advantages-_apply_grad')
            checker(log_probs, 'log_probs-_apply_grad')
            checker(returns, 'returns-_apply_grad')
            checker(values, 'values-_apply_grad')
            checker(mixed, 'mixed-_apply_grad')

        return objective


    def update(
        self,
        states: torch.Tensor,
        actions_cont: torch.Tensor,
        actions_disc: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_logits: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        dones: torch.Tensor,
    ) -> List[float]:
        """
        Performs multiple optimization epochs over a batch of experience.

        Args:
            states (torch.Tensor): State batch.
            actions_cont (torch.Tensor): Continuous actions.
            actions_disc (torch.Tensor): Discrete actions.
            old_log_probs (torch.Tensor): Old continuous log probabilities.
            old_logits (torch.Tensor): Old discrete logits.
            values (torch.Tensor): Value estimates.
            rewards (torch.Tensor): Reward sequence.
            returns (torch.Tensor): Discounted returns.
            advantages (torch.Tensor): Advantage values.
            dones (torch.Tensor): Episode termination flags.

        Returns:
            List[float]: Objective values for each optimization epoch.
        """
        objective_values = []

        for _ in range(self.opt_epochs):

            objective = self._apply_grad(
                states=states,
                returns=returns,
                advantages=advantages,
                actions_cont=actions_cont,
                actions_disc=actions_disc,
                old_log_probs=old_log_probs,
                old_logits=old_logits,
            )

            objective_values.append(objective)

        if self.debug:
            checker(advantages, 'advantages-update')
            checker(rewards, 'rewards-update')
            checker(returns, 'returns-update')
            checker(values, 'values-update')
            checker(states, 'states-update')
            checker(dones, 'dones-update')

        return torch.squeeze(torch.stack(objective_values))


def shuffle_and_split_into_mini_batches(*tensors, batch_size) -> list[list[torch.Tensor]]:
    """
    Shuffles tensors and splits them into mini-batches.

    Args:
        *tensors (torch.Tensor): Tensors to shuffle and batch.
        batch_size (int): Number of samples per batch.

    Returns:
        list[list[torch.Tensor]]: List of mini-batches.
    """
    tensors = list(tensors)

    N = tensors[0].shape[0]
    perm = torch.randperm(N, device=DEVICE)

    shuffled = [t[perm] for t in tensors]
    batches = []
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch = [t[start : end] for t in shuffled]
        batches.append(batch)

    for i in range(len(tensors) - 1, -1, -1):
        del tensors[i]

    return batches


def to_tensor(*arrays) -> list[torch.Tensor]:
    """
    Converts lists of arrays or tensors into stacked PyTorch tensors.

    Args:
        *arrays (list): Lists containing numpy arrays or tensors.

    Returns:
        list[torch.Tensor]: Converted tensors.
    """
    tensors = []

    for array in arrays:
        if isinstance(array[0], torch.Tensor):
            t = torch.stack(array, dim=0)
        else:
            t = torch.as_tensor(
                np.array(array),
                device=DEVICE
            )

        array.clear() # DO NOT TOUCH IT IDIOT

        if t.dtype not in (torch.bool, torch.int32, torch.int64):
            t = t.float()

        tensors.append(t)

    return tensors


def create_memory_window(memory_size: int, num_features: int, initial_obs: np.ndarray) -> deque:
    """
    Creates a fixed-size memory window for LSTM input.

    Args:
        memory_size (int): Number of stored observations.
        num_features (int): Number of features per observation.
        initial_obs (np.ndarray): Initial observation.

    Returns:
        deque: Initialized memory window.
    """
    window = deque(maxlen=memory_size)

    for _ in range(memory_size):
        window.append(initial_obs.copy())

    return window


def evaluate_policy(env: gym.Env, agent: PPOAgent, num_episodes: int = 10, memory_size: int = 3) -> List[float]:
    """
    Evaluates the agent's policy over multiple episodes.

    Args:
        env (gym.Env): Evaluation environment.
        agent (PPOAgent): PPO agent.
        num_episodes (int): Number of episodes to evaluate.
        memory_size (int): LSTM memory window size.

    Returns:
        List[float]: Total rewards per episode.
    """
    rewards = []
    agent.actor.eval()
    agent.critic.eval()

    for _ in range(num_episodes):
        obs, _ = env.reset()
        window = create_memory_window(memory_size=memory_size, num_features=env.observation_space.shape[0], initial_obs=obs)

        reward_per_episode = 0.0
        done = False
        while not done:
            _, _, action_cont, action_disc, _ = agent.act(tensor(window))
            obs, r, terminated, truncated, _,  = env.step((action_cont, action_disc))
            done = terminated or truncated
            reward_per_episode += r

        rewards.append(reward_per_episode)

    agent.actor.train()
    agent.critic.train()

    return rewards


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
        env: gym.Env, eval_env: gym.Env | None = None,

        hidden_layers: int = 16,
        hidden_units: int = 32,
        dropout: float = 0.2,

        actor_lr: float | np.float32 = 1e-4,
        critic_lr: float | np.float32 = 5e-4,

        advantage_type: str = 'gae',

        gamma: float | np.float32=0.99,
        lam: float | np.float32 = 0.95,

        clip_ratio: float | np.float32 = 0.2,
        grad_norm: float | None = None,
        opt_epochs: int = 10,

        c1: float | np.float32 = 0.05,
        c2: float | np.float32 = 0.01,

        batch_size: int = 64,
        memory_size: int = 3,
        replay_buffer_size: int = 8096,
        stats_every: int = 1000,
        eval_interval: int = 1000,
        eval_episodes: int = 10,
        total_steps: int = 100_000,

        del_model: bool = False,
        display_stat: bool = True,
        save_path: str = 'models_params',
        debug: bool = False,
        seed: int = 123,
    ) -> np.float32:
    """
    Trains the PPO agent in the given environment.

    Args:
        env (gym.Env): Training environment.
        eval_env (gym.Env | None): Optional evaluation environment.
        hidden_layers (int): Number of LSTM layers.
        hidden_units (int): Number of hidden units per layer.
        dropout (float): Dropout rate.
        actor_lr (float | np.float32): Actor learning rate.
        critic_lr (float | np.float32): Critic learning rate.
        advantage_type (str): Advantage method ('simple' or 'gae').
        gamma (float | np.float32): Discount factor.
        lam (float | np.float32): GAE lambda.
        clip_ratio (float | np.float32): PPO clip ratio.
        grad_norm (float | None): Gradient clipping norm.
        opt_epochs (int): Optimization epochs per update.
        c1 (float | np.float32): Value loss coefficient.
        c2 (float | np.float32): Entropy coefficient.
        batch_size (int): Mini-batch size.
        memory_size (int): LSTM memory size.
        replay_buffer_size (int): Steps before update.
        stats_every (int): Logging interval.
        eval_interval (int): Evaluation interval.
        eval_episodes (int): Evaluation episodes.
        total_steps (int): Total training steps.
        del_model (bool): Deletes model after training.
        display_stat (bool): Displays training statistics.
        save_path (str): Model save path.
        debug (bool): Enables debug mode.
        seed (int): Random seed.

    Returns:
        np.float32: Mean evaluation reward.
    """
    set_seed(seed)

    obs, _ = env.reset(seed=seed)

    last_n_rewards = []
    last_n_losses = []
    total_rewards = []
    total_objective = []
    eval_rewards = []

    ep_lens = []

    batch_actions_disc = []
    batch_actions_cont = []
    batch_rewards = []
    batch_states = []
    batch_values = []
    batch_logits_disc = []
    batch_log_probs = []
    batch_dones = []

    rewards_per_episode = 0.0
    current_buffer_size = 0
    ep_len = 0

    observation_shape = env.observation_space.shape[0]
    num_actions = env.action_space[1].n

    agent = PPOAgent(
        num_features=observation_shape, num_actions=num_actions, cont_actions=1, memory_size=memory_size, hidden_layers=hidden_layers,
        hidden_units=hidden_units, actor_lr=actor_lr, critic_lr=critic_lr, advantage_type=advantage_type, gamma=gamma, lam=lam,
        clip_ratio=clip_ratio, opt_epochs=opt_epochs, c1=c1, c2=c2, dropout=dropout, grad_norm=grad_norm, debug=debug
    )
    agent.critic.train()
    agent.actor.train()

    window = create_memory_window(memory_size=memory_size, num_features=observation_shape, initial_obs=obs)

    best_eval = float('-inf')
    pbar = tqdm(
        range(1, total_steps + 1),
        ncols=80,
        desc='Steps',
        file=sys.stderr,
        dynamic_ncols=False
    )

    for step in pbar:
        ep_len += 1
        batch_states.append(np.array(window, dtype=np.float32, copy=True))

        log_probs, logits_disc, action_cont, action_disc, values = agent.act(tensor(window))

        obs, reward, terminated, truncated, _ = env.step((action_cont, action_disc))
        window.append(obs)

        current_buffer_size += 1
        batch_actions_disc.append(action_disc)
        batch_actions_cont.append(action_cont)
        batch_rewards.append(reward)
        batch_values.append(values)
        batch_logits_disc.append(logits_disc[0])
        batch_log_probs.append(log_probs[0])

        rewards_per_episode += reward

        done = terminated or truncated
        batch_dones.append(done)

        if current_buffer_size >= replay_buffer_size:
            a_c, a_d, s, o_l_p, o_l_d, v, r, d = to_tensor(
                batch_actions_cont, batch_actions_disc,
                batch_states, batch_log_probs,
                batch_logits_disc, batch_values,
                batch_rewards, batch_dones
            )
            rt = agent.discount_reward(r, d)
            ad = agent.get_advantages(r, v, d, s)

            for batch in shuffle_and_split_into_mini_batches(
                s, a_c, a_d, o_l_p, o_l_d, v, r, rt, ad, d,
                batch_size=batch_size
            ):
                objective_values = agent.update(
                    *batch
                ).to('cpu').detach().numpy()
                total_objective.extend(objective_values)
                last_n_losses.extend(objective_values)

            current_buffer_size = 0

        if done:
            obs, _ = env.reset()
            window = create_memory_window(memory_size=memory_size, num_features=observation_shape, initial_obs=obs) # recreate window
            ep_lens.append(ep_len)
            last_n_rewards.append(rewards_per_episode)
            total_rewards.append(rewards_per_episode)
            ep_len = 0.0
            rewards_per_episode = 0.0

        if step % eval_interval == 0 and eval_env is not None:
            rewards = evaluate_policy(eval_env, agent, eval_episodes)
            eval_rewards.extend(rewards)
            avg_reward = np.mean(rewards)

            if display_stat:
                tqdm.write(f'Policy evaluation | Mean rewards {avg_reward:.3f} per {eval_episodes} episodes |')

            if avg_reward > best_eval:
                best_eval = avg_reward

        if step % stats_every == 0:
            avg_objective = 0 if len(last_n_losses) == 0 else np.mean(last_n_losses)

            if display_stat:
                tqdm.write(
                    f'Step {step} | Mean rewards: {np.mean(last_n_rewards):.3f} | '
                    f'Mean objective: {np.mean(avg_objective):.3f} |'
                )

            last_n_rewards = []
            last_n_losses = []

    agent.save(save_path)

    if display_stat:
        plot_statistics(total_rewards, 'Total rewards (train env)', 'Rewards')
        plot_statistics(eval_rewards, 'Total rewards (eval env)', 'Rewards')
        plot_statistics(ep_lens, 'Total episode lengths', 'Lengths')
        plot_statistics(total_objective, 'Objective', 'Objective')

    avg_eval_rewards = evaluate_policy(eval_env, agent, eval_env.num_episodes)

    if del_model:
        del agent
        gc.collect()

    return np.mean(avg_eval_rewards)
