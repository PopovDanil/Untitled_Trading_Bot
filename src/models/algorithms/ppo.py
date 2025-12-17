import gc
import os
from collections import deque
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
from models.neural_networks.lstm import LSTM
from settings import DIR
from tqdm import tqdm

DEBUG = True


class RecourseMonitor:
    def __init__(self):
        super().__init__()
        self.batch_count = 0

    def get_info(self):
        self.batch_count += 1
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        print(f"Batch {self.batch_count}: "
                f"CPU: {cpu_percent}%, "
                f"RAM: {memory.percent}%")

        if tf.config.experimental.list_physical_devices('GPU'):
            gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
            print(f"GPU Memory: {gpu_stats['current'] / 1024**2:.0f}MB")


class PPOAgent:
    def __init__(
            self, num_features: int,
            num_actions: int,
            cont_actions: int,
            memory_size: int = 3,
            hidden_layers: int = 16,
            hidden_units: int = 32,
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
        self.memory_size = memory_size
        self.num_actions = num_actions
        self.cont_actions = cont_actions
        self.num_features = num_features
        self.advantage_type = advantage_type

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor = LSTM(timestamps=memory_size, features=num_features, output_shape=3, hidden_layers=hidden_layers, hidden_units=hidden_units, lr=actor_lr)
        self.critic = LSTM(timestamps=memory_size, features=num_features, output_shape=1, hidden_layers=hidden_layers, hidden_units=hidden_units, lr=critic_lr)

        self.c1 = c1
        self.c2 = c2
        self.lam = lam
        self.gamma = gamma

        self.EPS = 1e-5
        self.LOG_STD_MIN = -10.0
        self.LOG_STD_MAX = 2.0

        self.recourse_monitor = RecourseMonitor()

    def save(self, save_path: str = 'models_params') -> None:
        intermediate_path = os.path.join('models', 'algorithms')
        self.actor.save(path=os.path.join(DIR, intermediate_path, save_path, 'ppo', 'actor.keras'))
        self.critic.save(path=os.path.join(DIR, intermediate_path, save_path, 'ppo', 'critic.keras'))

    def act(self, observation: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        mixed = self.actor.forward(observation.reshape(1, *observation.shape))
        logits_disc, mean, log_std = tf.split(mixed, 3, axis=1)

        action_disc = tf.squeeze(tf.random.categorical(logits_disc, 1)).numpy()

        action_cont, cont_log_probs, _, _ = self.compute_cont_action(mean=mean, log_std=log_std)
        action_cont = action_cont.numpy()

        value = tf.squeeze(self.critic.forward(observation.reshape(1, *observation.shape)))

        return cont_log_probs, logits_disc, action_cont, action_disc, value

    @tf.function
    def compute_cont_log_probs(self, actions: np.ndarray | tf.Tensor, mean: tf.Tensor, log_std: tf.Tensor) -> tf.Tensor:

        actions = tf.reshape(actions, shape=(actions.shape[0], 1))

        log_std_clipped = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = tf.math.exp(log_std) + self.EPS

        actions_clipped = tf.clip_by_value(actions, -1.0 + self.EPS, 1.0 - self.EPS)

        u = tf.math.atanh(actions_clipped)

        quad = ((u - mean) ** 2) / (std ** 2) + self.EPS

        cont_log_probs = -0.5 * tf.reduce_sum(quad + 2.0 * log_std_clipped + tf.math.log(2 * np.pi), axis=1, keepdims=True)
        jacobian = tf.math.log(1.0 - actions_clipped ** 2 + self.EPS)
        cont_log_probs = cont_log_probs - jacobian

        # tf.debugging.check_numerics(cont_log_probs, 'NaNs were detected in compute_cont_log_probs')

        return cont_log_probs

    @tf.function
    def compute_cont_action(self, mean: tf.Tensor, log_std: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # tf.debugging.check_numerics(mean, message='NaNs were detected in compute_cont_action')
        # tf.debugging.check_numerics(log_std, message='NaNs were detected in compute_cont_action')

        log_std_clipped = tf.clip_by_value(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = tf.math.exp(log_std) + self.EPS

        z = tf.random.normal(shape=(self.cont_actions,))
        u = mean + std * z

        action_cont = tf.squeeze(tf.tanh(u))

        cont_log_probs = -0.5 * tf.reduce_sum((z ** 2) + 2.0 * log_std_clipped + tf.math.log(2 * np.pi), axis=1, keepdims=True)
        jacobian = tf.math.log(1.0 - action_cont ** 2 + self.EPS)
        cont_log_probs = cont_log_probs - jacobian

        tf.debugging.check_numerics(cont_log_probs, message='NaNs were detected in compute_cont_action')

        return action_cont, cont_log_probs, mean, std

    @tf.function
    def discount_reward(self, rewards: tf.Tensor, dones: tf.Tensor, gamma: float | np.float32 = 0.99) -> tf.Tensor:
        rews = tf.reverse(rewards, axis=[0])
        dns = tf.reverse(dones, axis=[0])

        def worker(carry, x):
            reward, done = x
            carry = reward + gamma * carry * tf.where(done, 0.0, 1.0)
            return carry

        discounted = tf.scan(
            fn=worker,
            elems=(rews, dns),
            initializer=tf.constant(0.0)
        )

        return tf.reverse(discounted, axis=[0])

    @tf.function
    def standardize(self, values) -> tf.Tensor:
        return (values - tf.reduce_mean(values)) / (tf.math.reduce_std(values) + 1e-8)

    @tf.function
    def simple_advantages(self, returns, values) -> tf.Tensor:
        advantages = returns - values
        return self.standardize(advantages)

    @tf.function
    def gae_advantages(self, rewards, values, dones, states):

        next_values = tf.concat([values[1:], tf.expand_dims(values[-1], 0)], axis=0)
        deltas = rewards + self.gamma * next_values * (1.0 - tf.cast(dones, tf.float32)) - values

        def body(carry, x):
            delta, done = x
            carry = delta + self.gamma * self.lam * (1.0 - tf.cast(done, tf.float32)) * carry
            return carry

        adv = tf.scan(
            fn=body,
            elems=(tf.reverse(deltas, [0]), tf.reverse(dones, [0])),
            initializer=tf.constant(0.0),
        )
        adv = tf.reverse(adv, [0])

        return self.standardize(adv)

    @tf.function
    def get_advantages(self, rewards, values, dones, states) -> tf.Tensor:
        if self.advantage_type == 'simple':
            return self.simple_advantages(rewards, values)
        return self.gae_advantages(rewards, values, dones, states)

    @tf.function
    def _compute_loss(self, new_log_probs, old_log_probs, new_logits_disc, old_logits_disc, log_std_new, actions_disc, values, returns, advantages):

        # Only for DEBUGGING
        tf.debugging.check_numerics(new_log_probs, 'new_log_probs contains NaNs')
        tf.debugging.check_numerics(old_log_probs, 'old_log_probs contains NaNs')
        tf.debugging.check_numerics(log_std_new, 'log_std_new contains NaNs')

        actions_one_hot = tf.one_hot(actions_disc, self.num_actions, dtype=tf.float32)

        new_policy_disc = tf.nn.softmax(new_logits_disc)
        old_policy_disc = tf.nn.softmax(old_logits_disc)

        new_action_porbs_disc = tf.reduce_sum(actions_one_hot * new_policy_disc, axis=1)
        old_action_porbs_disc = tf.reduce_sum(actions_one_hot * old_policy_disc, axis=1)

        ratio_disc = tf.exp(tf.math.log(new_action_porbs_disc + 1e-10) - tf.math.log(old_action_porbs_disc + 1e-10))

        ratio_cont = tf.exp(new_log_probs - old_log_probs)
        ratio_cont = tf.squeeze(ratio_cont)

        clipped_disc = tf.clip_by_value(ratio_disc, 1 - self.clip_ratio, 1 + self.clip_ratio)
        clipped_cont = tf.clip_by_value(ratio_cont, 1 - self.clip_ratio, 1 + self.clip_ratio)

        policy_loss_disc = -tf.reduce_mean(tf.minimum(ratio_disc * advantages, clipped_disc * advantages))
        policy_loss_cont = -tf.reduce_mean(tf.minimum(ratio_cont * advantages, clipped_cont * advantages))

        value_loss = tf.reduce_mean(tf.square(returns - values))

        entropy_bonus_disc = -tf.reduce_mean(tf.reduce_sum(new_policy_disc * tf.math.log(new_policy_disc + 1e-10), axis=1))
        entropy_bonus_cont = -tf.reduce_mean(tf.reduce_sum(log_std_new + 0.5 * np.log(2 * np.pi * np.e), axis=1))

        objective = -policy_loss_cont - policy_loss_disc + self.c1 * value_loss + self.c2 * (entropy_bonus_disc + entropy_bonus_cont)

        disc_actor_loss = policy_loss_disc - self.c2 * entropy_bonus_disc
        cont_actor_loss = policy_loss_cont - self.c2 * entropy_bonus_cont
        actor_loss = disc_actor_loss + cont_actor_loss

        return objective, actor_loss, value_loss

    @tf.function
    def _apply_grad(
            self,
            states: tf.Tensor,
            returns: tf.Tensor,
            advantages: tf.Tensor,
            actions_cont: tf.Tensor,
            actions_disc: tf.Tensor,
            old_log_probs: tf.Tensor,
            old_logits: tf.Tensor,
            values: tf.Tensor,
        ):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            mixed = self.actor.forward(states)
            logits, mean, log_std = tf.split(mixed, 3, axis=1)

            tf.debugging.check_numerics(mixed, 'NaNs were detected in update')

            log_probs = self.compute_cont_log_probs(actions_cont, mean, log_std)

            values = tf.squeeze(self.critic.forward(states))

            objective, actor_loss, critic_loss = self._compute_loss(
                new_log_probs=log_probs,
                old_log_probs=old_log_probs,

                new_logits_disc=logits,
                old_logits_disc=old_logits,

                log_std_new=log_std,
                actions_disc=actions_disc,

                values=values,
                returns=returns,
                advantages=advantages
            )

        actor_gradients = tape1.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = tape2.gradient(critic_loss, self.critic.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return objective

    def update(
        self,
        states: tf.Tensor,
        actions_cont: tf.Tensor,
        actions_disc: tf.Tensor,
        old_log_probs: tf.Tensor,
        old_logits: tf.Tensor,
        values: tf.Tensor,
        rewards: tf.Tensor,
        dones: tf.Tensor,
        debug: bool = False,
    ) -> List[float]:

        if DEBUG:
            print('Monitoring before UPDATE STARTED')
            self.recourse_monitor.get_info()

        objective_values = []
        returns = self.discount_reward(rewards, dones, self.gamma)
        advantages = self.get_advantages(rewards, values, dones, states)

        for epoch in range(self.opt_epochs):
            # print(f'Optimization epoch {epoch}')
            # self.recourse_monitor.get_info()

            objective = self._apply_grad(
                states=states,
                returns=returns,
                advantages=advantages,
                actions_cont=actions_cont,
                actions_disc=actions_disc,
                old_log_probs=old_log_probs,
                old_logits=old_logits,
                values=values
            )

            objective_values.append(objective)

            # if debug:
            #     print('Monitoring BEFORE garbage collected in UPDATE')
            #     self.recourse_monitor.get_info()

            # del tape1
            # del tape2
            # del actor_gradients
            # del critic_gradients
            # gc.collect()

            # if debug:
            #     print('Monitoring AFTER garbage collected in UPDATE')
            #     self.recourse_monitor.get_info()

        return tf.squeeze(tf.stack(objective_values))


def to_tensor(*arrays: List) -> List:
    tensors = []
    for array in arrays:

        tensor = tf.stack(array)

        if not tensor.dtype.is_integer and not tensor.dtype.is_bool:
            tensor = tf.cast(tensor, dtype=tf.float32)

        tensors.append(tensor)

    return tensors


def create_memory_window(memory_size: int, num_features: int, initial_obs: np.ndarray) -> deque:
    window = deque(maxlen=memory_size)

    for _ in range(memory_size):
        window.append(np.zeros(
            shape=(num_features,),
            dtype=np.float32,
        ))
    window.append(initial_obs)

    return window


def evaluate_policy(env: gym.Env, agent: PPOAgent, num_episodes: int = 10, memory_size: int = 3) -> List[float]:
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        window = create_memory_window(memory_size=memory_size, num_features=env.observation_space.shape[0], initial_obs=obs)

        reward_per_episode = 0.0
        done = False
        while not done:
            _, _, action_cont, action_disc, _ = agent.act(np.array(window))
            obs, r, terminated, truncated, _,  = env.step((action_cont, action_disc))
            done = terminated or truncated
            reward_per_episode += r

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
    plt.savefig(os.path.join(os.path.dirname(DIR), 'plots', 'ppo', f'{plot_name.replace(' ', '_')}'), dpi=200)
    plt.close()


def set_seed(seed: int = 123) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(
        env: gym.Env, eval_env: gym.Env | None = None,

        hidden_layers: int = 16,
        hidden_units: int = 32,

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
        memory_size: int = 3,
        stats_every: int = 1000,
        eval_interval: int = 1000,
        eval_episodes: int = 10,
        total_steps: int = 100_000,

        del_model: bool = False,
        display_stat: bool = True,
        save_path: str = 'models_params',
        seed: int = 123,
    ) -> Tuple[np.float32]:

    set_seed(seed)

    obs, _ = env.reset(seed=seed)

    last_n_rewards = []
    last_n_losses = []
    total_rewards = []
    total_losses = []
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
    current_batch_size = 0
    ep_len = 0

    observation_shape = env.observation_space.shape[0]
    num_actions = env.action_space[1].n

    agent = PPOAgent(
        num_features=observation_shape, num_actions=num_actions, cont_actions=1, memory_size=memory_size, hidden_layers=hidden_layers,
        hidden_units=hidden_units, actor_lr=actor_lr, critic_lr=critic_lr, advantage_type=advantage_type, gamma=gamma, lam=lam,
        clip_ratio=clip_ratio, opt_epochs=opt_epochs, c1=c1, c2=c2
    )
    window = create_memory_window(memory_size=memory_size, num_features=observation_shape, initial_obs=obs)

    best_eval = float('-inf')

    for step in tqdm(range(1, total_steps + 1), ncols=80, desc='Steps'):
        ep_len += 1
        batch_states.append(np.array(window, dtype=np.float32, copy=True))

        log_probs, logits_disc, action_cont, action_disc, values = agent.act(np.array(window))

        obs, reward, terminated, truncated, _ = env.step((action_cont, action_disc))
        window.append(obs)

        current_batch_size += 1
        batch_actions_disc.append(action_disc)
        batch_actions_cont.append(action_cont)
        batch_rewards.append(reward)
        batch_values.append(values)
        batch_logits_disc.append(logits_disc[0])
        batch_log_probs.append(log_probs[0])

        rewards_per_episode += reward

        done = terminated or truncated
        batch_dones.append(done)

        if current_batch_size >= batch_size:
            a_c, a_d, s, o_l_p, o_l_d, v, r, d = to_tensor(
                batch_actions_cont, batch_actions_disc,
                batch_states, batch_log_probs,
                batch_logits_disc, batch_values,
                batch_rewards, batch_dones
            )

            objective_values = agent.update(s, a_c, a_d, o_l_p, o_l_d, v, r, d,
                                debug=False).numpy()
            total_losses.extend(objective_values)
            last_n_losses.extend(objective_values)

            batch_actions_cont.clear()
            batch_actions_disc.clear()
            batch_logits_disc.clear()
            batch_log_probs.clear()
            batch_rewards.clear()
            batch_states.clear()
            batch_values.clear()
            batch_dones.clear()

            current_batch_size = 0

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
                agent.save(save_path)

        if step % stats_every == 0:
            if display_stat:
                tqdm.write(f'Step {step} | Mean rewards: {np.mean(last_n_rewards):.3f} | Mean objective: {np.mean(last_n_losses):.3f} |')
            last_n_rewards = []
            last_n_losses = []

    if display_stat:
        plot_statistics(total_rewards, 'Total rewards per episode')
        plot_statistics(ep_lens, 'Total episode lengths')
        plot_statistics(total_losses, 'Objective')

    if del_model:
        del agent
        gc.collect()

    return np.mean(total_losses), np.mean(total_rewards)
