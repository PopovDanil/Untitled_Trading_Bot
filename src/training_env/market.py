import random
from typing import List, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class Market(gym.Env):
    def __init__(self, training_files: List[str], initial_cash: np.float32, slippage: np.float32, broker_fee: np.float32, lam: float = 0.01, reward_scaler: float = 100.0, seed: int = 123):
        super().__init__()
        self.files = training_files

        random.seed(seed)

        # observations - open, high, low, close, volume (already standardized)
        self._get_observations(files=training_files)

        # open, high, low, close, volume, position, current_step, remaining_time, cash
        self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.observation_shape + 4,), dtype=np.float32)

        # go long - >0, short - <0, hold = 0, value - amount of current cash/futures to spend, second - 0 - wait, 1 - close position
        self.action_space = spaces.Tuple([
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            spaces.Discrete(2)
        ])

        self.profitable_episodes = 0
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.pnl = 0.0
        self.qty = 0.0

        self.slippage = slippage
        self.fee = broker_fee
        self.lam = lam # punishment scaler for holding outside position

        self.max_reward = 100.0
        self.min_reward = -100.0
        self.reward_scaler = reward_scaler # used to avoid to small rewards due to cash increase/decrease

        self.position = 0 # 0 - waiting, 1 - long, -1 - short
        self.entry_price = 0.0

        self.current_price = self.observations[0].iloc[0]['close']
        self.previous_price = self.current_price
        self.current_episode = 0
        self.current_step = -1

    def _reorder_obs(self):
        random.shuffle(self.observations)

    def _get_observations(self, files: List[str]):
        self.observations = []
        for file in files:
            data = pd.read_csv(file, index_col=0)
            self.observations.append(data)

        # exclude close_raw
        self.observation_shape = self.observations[0].shape[1] - 1
        self.episode_len = self.observations[0].shape[0]
        self.num_episodes = len(self.observations)

        # exclude close_raw
        self.data_columns = self.observations[0].columns.to_list()
        self.data_columns.remove('close_raw')

    def _get_next_obs(self) -> Tuple[np.ndarray, bool]:
        previous = self.current_step
        obs = np.zeros(shape=(self.observation_shape), dtype=np.float32)
        self.current_step = (self.current_step + 1) % self.episode_len

        done = False
        remaining_steps = 0
        position = self.position

        if (self.current_step == 0 and previous > 0) or self.current_episode >= self.num_episodes:
            done = True
            position = 0
            self.current_step = -1
            self.current_episode += 1

            if self.current_episode >= self.num_episodes:
                self._reorder_obs()
                self.current_episode = 0
        else:
            remaining_steps = self.episode_len - self.current_step
            obs = self.observations[self.current_episode].loc[self.current_step, self.data_columns].values

        # Normalize them
        norm_step = self.current_step / self.episode_len
        norm_remaining = remaining_steps / self.episode_len

        obs = np.hstack([obs, np.array([position, norm_step, norm_remaining])], dtype=np.float32)

        return obs, done

    def step(self, action: Tuple) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:

        pnl = 0.0
        ratio, close = action

        current_obs, done = self._get_next_obs() # CASH NOT ADDED
        if done: close = True

        # third element - close
        self.current_price = self.observations[self.current_episode].iloc[self.current_step]['close_raw']

        reward = (self.position * (self.current_price - self.previous_price)) / self.initial_cash * self.reward_scaler
        truncated = done
        terminated = done
        info = {}

        if close and self.position == 0:
            reward -= 1

        if close == 1:

            pnl = self.position * (self.current_price - self.entry_price) * self.qty
            pnl = pnl * (1 - self.fee) if pnl > 0 else pnl * (1 + self.fee)

            # avoid 0 division
            reward += (pnl / (self.cash + 1e-6)) * self.reward_scaler
            self.cash += pnl

            if self.cash <= 0:
                truncated = True
                terminated = True
                done = True
                reward = -10

            if done:
                profit_ratio = self.cash / self.initial_cash * 100
                if profit_ratio > 0: self.profitable_episodes += 1

                info = {
                    'profit_ratio': profit_ratio,
                    'profitable_episodes': self.profitable_episodes,
                    'last_step': self.current_step
                }

            self.position = 0
            self.qty = 0.0
        elif ratio != 0 and self.position != 0:
            reward = -1
        elif ratio == 0 and self.position == 0:
            reward -= self.lam
        else:
            if ratio > 0:
                self.entry_price = self.current_price * (1 + self.slippage)
                self.position = 1
            else:
                self.entry_price = self.current_price * (1 - self.slippage)
                self.position = -1
            self.qty = abs(ratio) * self.cash * (1 - self.fee) / self.current_price

            self.cash -= abs(ratio) * self.cash

            if self.cash == 0:
                reward -= 2

        # Update the price
        self.previous_price = self.current_price

        current_obs = np.hstack([current_obs, np.array([self.cash / self.initial_cash])], dtype=np.float32) # Normalized cash was added

        # Clip reward
        reward = np.clip(reward, self.min_reward, self.max_reward)
        return current_obs, reward, truncated, terminated, info

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        self.cash = self.initial_cash
        self.pnl = 0.0
        self.qty = 0.0

        self.position = 0
        self.entry_price = 0.0

        obs, done = self._get_next_obs()
        if done:
            obs, _ = self._get_next_obs()

        self.current_price = self.observations[self.current_episode].iloc[0]['close_raw']
        self.previous_price = self.current_price

        obs = np.hstack([obs, np.array([1])], dtype=np.float32)
        return obs, {}
