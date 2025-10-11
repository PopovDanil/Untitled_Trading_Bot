from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.pylab import ArrayLike
from tqdm import tqdm

from ..neural_networks.mlp import MLP


class PPOAgent:
    def __init__(
            self, observation_shape: int,
            num_actions: int,
            hidden_shapes: Tuple = (64, 64),
            actor_lr: float | np.float32 = 1e-4,
            critic_lr: float | np.float32 = 5e-4,
            advantage_type: str = 'gae',
            gamma: float | np.float32=0.99,
            lam: float | np.float32 = 0.95,
            clip_ratio: float | np.float32 = 0.2,
            opt_epochs: int = 10,
            c1: float | np.float32 = 0.05,
            c2: float | np.float32 = 0.01
        ):
        self.opt_epochs = opt_epochs
        self.clip_ratio = clip_ratio
        self.num_actions = num_actions
        self.advantage_type = advantage_type

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor = MLP(observation_shape=observation_shape, output_shape=num_actions, hidden_shapes=hidden_shapes, lr=actor_lr)
        self.critic = MLP(observation_shape=observation_shape, output_shape=1, hidden_shapes=hidden_shapes, lr=critic_lr)

        self.c1 = c1
        self.c2 = c2
        self.lam = lam
        self.gamma = gamma

    def save(self, save_path: str = 'models_params') -> None:
        self.actor.model.save(save_path + '/ppo/actor.keras')
        self.critic.model.save(save_path + '/ppo/critic.keras')

    def act(self, observation: np.ndarray | tf.Tensor) -> Tuple[tf.Tensor, ArrayLike | float, tf.Tensor]:
        logits = self.actor.forward(np.atleast_2d(observation))
        action = tf.squeeze(tf.random.categorical(logits, 1)).numpy()
        value = tf.squeeze(self.critic.forward(np.atleast_2d(observation)))
        return logits, action, value

    def discount_reward(self, rewards: tf.Tensor, dones: tf.Tensor, gamma: float | np.float32 = 0.99) -> tf.Tensor:
        result = []
        discounted_sum = 0.0

        for reward, done in zip(tf.reverse(rewards, axis=[0]), tf.reverse(dones, axis=[0])):

            if done:
                discounted_sum = 0.0

            discounted_sum = reward + gamma * discounted_sum
            result.append(discounted_sum)

        return tf.convert_to_tensor(result[::-1], dtype=tf.float32)

    def standardize(self, values) -> tf.Tensor:
        return (values - tf.reduce_mean(values)) / (tf.math.reduce_std(values) + 1e-8)

    def simple_advantages(self, returns, values) -> tf.Tensor:
        advantages = returns - values # type: ignore
        return self.standardize(advantages)

    def gae_advantages(self, rewards, values, dones, states) -> tf.Tensor:
        T = rewards.shape[0]
        advantages = np.zeros(T, dtype=np.float32) # type: ignore
        gae = .0

        next_value = .0
        if not dones[-1]:
            next_value = tf.squeeze(self.critic.forward(np.atleast_2d(states[-1])))

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            next_value = values[t]
            advantages[t] = gae

        advantages = tf.constant(advantages)
        return self.standardize(advantages)

    def get_advantages(self, rewards, values, dones, states) -> tf.Tensor:
        if self.advantage_type == 'simple':
            return self.simple_advantages(rewards, values)
        return self.gae_advantages(rewards, values, dones, states)

    def _compute_loss(self, new_logits, old_logits, actions, values, returns, advantages):
        actions_one_hot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)

        new_policy = tf.nn.softmax(new_logits)
        old_policy = tf.nn.softmax(old_logits)

        new_action_porbs = tf.reduce_sum(actions_one_hot * new_policy, axis=1)
        old_action_porbs = tf.reduce_sum(actions_one_hot * old_policy, axis=1)

        ratio = tf.exp(tf.math.log(new_action_porbs + 1e-10) - tf.math.log(old_action_porbs + 1e-10))
        clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
        value_loss = tf.reduce_mean(tf.square(returns - values))

        entropy_bonus = -tf.reduce_mean(tf.reduce_sum(new_policy * tf.math.log(new_policy + 1e-10), axis=1)) # type: ignore

        objective = -policy_loss + self.c1 * value_loss + self.c2 * entropy_bonus
        actor_loss = policy_loss - self.c2 * entropy_bonus

        return objective, actor_loss, value_loss

    def update(self, states: tf.Tensor, actions: tf.Tensor, old_logits: tf.Tensor, values: tf.Tensor, rewards: tf.Tensor, dones: tf.Tensor) -> List[float]:
        objective_values = []
        advantages = self.get_advantages(rewards, values, dones, states)
        returns = self.discount_reward(rewards, dones, self.gamma)

        for _ in range(self.opt_epochs):
            with tf.GradientTape(persistent=True) as tape:
                logits = self.actor.forward(states)
                values = tf.squeeze(self.critic.forward(states))
                objective, actor_loss, critic_loss = self._compute_loss(logits, old_logits, actions, values, returns, advantages)

            actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
            critic_gradients = tape.gradient(critic_loss, self.critic.model.trainable_variables)

            self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.model.trainable_variables)) # type: ignore
            self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.model.trainable_variables)) # type: ignore
            objective_values.append(objective.numpy())

        return objective_values

def to_tensor(*arrays: List) -> List:
    tensors = []
    for i, array in enumerate(arrays):
        tensor = tf.convert_to_tensor(array, dtype=tf.float32 if i != 0 else tf.int32)
        tensors.append(tensor)
    return tensors


def reset_arrays(arrays: List) -> List:
    for i in range(len(arrays)):
        arrays[i] = []
    return arrays


def evaluate_policy(env: gym.Env, agent: PPOAgent, num_episodes: int = 10) -> List[float]:
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        reward_per_episode = 0.0
        done = False

        while not done:
            _, action, _ = agent.act(obs)
            obs, r, terminated, truncated, _,  = env.step(action)
            done = terminated or truncated
            reward_per_episode += r # type: ignore

        rewards.append(reward_per_episode)

    return rewards


def plot_statistics(data: list, plot_name: str, rolling_window: int = 50):
    episodes = np.arange(1, len(data) + 1)

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 6))

    plt.plot(episodes, data, color="lightcoral", alpha=0.6, label="Raw")

    if len(data) >= rolling_window:
        rolling = np.convolve(data, np.ones(rolling_window)/rolling_window, mode="valid")
        plt.plot(episodes[rolling_window-1:], rolling, color="red", linewidth=2.5, label=f"Rolling mean ({rolling_window})")

    plt.title(f"Training Progress - {plot_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Mean Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/ppo/{plot_name.replace(' ', '')}_ppo.png", dpi=150)
    plt.close()


def set_seed(seed: int = 123) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(
        env: gym.Env, eval_env: gym.Env,
        hidden_shapes: Tuple = (64, 64),

        actor_lr: float | np.float32 = 1e-4,
        critic_lr: float | np.float32 = 5e-4,

        advantage_type: str = 'gae',

        gamma: float | np.float32=0.99,
        lam: float | np.float32 = 0.95,

        clip_ratio: float | np.float32 = 0.2,
        opt_epochs: int = 10,

        c1: float | np.float32 = 0.05,
        c2: float | np.float32 = 0.01,

        batch_size: int = 64,
        stats_every: int = 1000,
        eval_interval: int = 1000,
        eval_episodes: int = 10,
        total_steps: int = 100_000,

        save_path: str = 'models_params',
        seed: int = 123
    ):

    set_seed(seed)

    obs, _ = env.reset(seed=seed)

    last_n_rewards = []
    last_n_losses = []
    total_rewards = []
    total_losses = []
    eval_rewards = []
    ep_lens = []

    batch_actions = []
    batch_rewards = []
    batch_states = []
    batch_values = []
    batch_logits = []
    batch_dones = []

    rewards_per_episode = 0.0
    current_batch_size = 0
    ep_len = 0

    observation_shape = env.observation_space.shape[0] # type: ignore
    num_actions = env.action_space.n # type: ignore

    agent = PPOAgent(
        observation_shape, num_actions, hidden_shapes, actor_lr,
        critic_lr, advantage_type, gamma, lam, clip_ratio,
        opt_epochs, c1, c2
    )

    best_eval = float('-inf')

    for step in tqdm(range(1, total_steps + 1), ncols=80, desc='Steps'):
        ep_len += 1
        batch_states.append(obs)

        logits, action, values = agent.act(obs)

        obs, reward, terminated, truncated, _ = env.step(action) # type: ignore

        current_batch_size += 1
        batch_actions.append(action)
        batch_rewards.append(reward)
        batch_values.append(values)
        batch_logits.append(logits.numpy()[0])

        rewards_per_episode += reward # type: ignore

        done = terminated or truncated
        batch_dones.append(done)

        if current_batch_size >= batch_size:
            a, s, o_l, v, r, d = to_tensor(batch_actions, batch_states, batch_logits, batch_values, batch_rewards,  batch_dones)
            objective_values = agent.update(s, a, o_l, v, r, d)
            total_losses.extend(objective_values)
            last_n_losses.extend(objective_values)
            batch_actions, batch_states, batch_logits, batch_values, batch_rewards,  batch_dones = [], [], [], [], [], []
            current_batch_size = 0

        if done:
            obs, _ = env.reset()
            ep_lens.append(ep_len)
            last_n_rewards.append(rewards_per_episode)
            total_rewards.append(rewards_per_episode)
            ep_len = 0.0
            rewards_per_episode = 0.0

        if step % eval_interval == 0:
            rewards = evaluate_policy(eval_env, agent, eval_episodes)

            eval_rewards.extend(rewards)
            avg_reward = np.mean(rewards)

            tqdm.write(f'Policy evaluation | Mean rewards {avg_reward:.3f} per {eval_episodes} episodes |')

            if avg_reward > best_eval:
                best_eval = avg_reward
                agent.save(save_path)

        if step % stats_every == 0:
            tqdm.write(f'Step {step} | Mean rewards: {np.mean(last_n_rewards):.3f} | Mean objective: {np.mean(last_n_losses):.3f} |')
            last_n_rewards = []
            last_n_losses = []

    plot_statistics(total_rewards, 'Total rewards per episode')
    plot_statistics(ep_lens, 'Total episode lengths')
    plot_statistics(total_losses, 'Objective')


# if __name__ == '__main__':

#     env = gym.make('CartPole-v1')
#     eval_env = gym.make('CartPole-v1')

#     train(
#         env, eval_env,
#         hidden_shapes=(64, 64),
#         actor_lr=2e-4,
#         critic_lr=5e-4,
#         advantage_type='gae',
#         gamma=0.99,
#         lam=0.95,
#         clip_ratio=0.2,
#         opt_epochs=4,
#         c1=0.5,
#         c2=0.001,
#         batch_size=256,
#         stats_every=1000,
#         eval_interval=1000,
#         eval_episodes=10,
#         total_steps=50000,
#         save_path='models_params',
#         seed=123
#     )