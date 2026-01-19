import random

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from settings import market_settings


class Market(gym.Env):
    """
    Simulates Stock market. Includes slippage and broker fee.
    """
    def __init__(
        self,
        training_files: list[str],
        features: list[str],
        initial_cash: np.float32,
        slippage: np.float32,
        broker_fee: np.float32,
        lam: float = 0.1,
        seed: int = 42
        ) -> None:
        """
        Initializes a new market instance.

        Args:
            training_files (list[str]): list of paths to train data.
            features (list[str]): list of features used during training.
            initial_cash (np.float32): starting amount of money (dollars).
            slippage (np.float32): slippage.
            broker_fee (np.float32): broker's fee.
            lam (float, optional): punishment scaler for holding outside position. Defaults to 0.01.
            reward_scaler (float, optional): scaling coefficient. Defaults to 100.0.
            seed (int, optional): random seed. Defaults to 123.
        """
        random.seed(seed)
        super().__init__()

        self.files = training_files
        self.features = features
        self._get_observations(files=training_files)

        # Describes properties of training data
        self.observation_space = spaces.Box(
            low=float('-inf'),
            high=float('inf'),
            shape=(self.observation_shape + 3,),
            dtype=np.float32
        )

        # Hybrid action space meaning:
        # First component - continuous
        # go long - >0,
        # short - <0,
        # hold = 0,
        # value - amount of current cash/futures to spend
        # -----------------------------------------------
        # Second component - discrete
        # 0 - wait,
        # 1 - close position
        self.action_space = spaces.Tuple([
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            spaces.Discrete(2)
        ])

        self.profitable_episodes = 0
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.pnl = 0.0 # Profit and Loss
        self.qty = 0.0 # Quantity

        self.slippage = slippage
        self.fee = broker_fee
        self.lam = lam

        # calculated by looking at code
        self.max_reward = 7
        self.min_reward = -7

        self.ratio_threshold = 0.1 # used to avoid noise during training
        self.position = 0 # 0 - waiting, 1 - long, -1 - short
        self.entry_price = 0.0

        self.current_price = self.observations.iloc[0]['close_raw']
        self.last_not_zero_price = self.current_price # to avoid accidental division by zero

        self.previous_price = self.current_price
        self.current_episode = 0
        self.current_step = -1

        self.zero = 0
        self.one = 0


    def _adjust_initial_cash(self):
        # Somnitelno, no okey
        surplus = 1 if self.current_step == -1 else 0
        if self.initial_cash / self.observations.iloc[self.current_step + surplus]['close_raw'] < 10:
            self.initial_cash = self.observations.iloc[self.current_step + surplus]['close_raw'] * 10
            self.cash = self.initial_cash


    def _reorder_obs(self) -> None:
        """
        Randomly shuffles training files.
        """
        random.shuffle(self.files)


    def _load_file(self, file: str) -> pd.DataFrame:
        """
        Loads df from .csv file and resets index column.

        Args:
            file (str): path.

        Returns:
            pd.DataFrame: df with selected features.
        """
        data = pd.read_csv(file, index_col=0)
        data = data.reset_index()
        return data[self.features]


    def _get_observations(self, files: list[str]) -> None:
        """
        Initially loads the first training df and set episode length, number of training
        examples, feature names (second time).

        Args:
            files (list[str]): list of files.
        """
        self.observations = self._load_file(files[0])

        # exclude close_raw
        self.observation_shape = self.observations.shape[1] - 1
        self.episode_len = self.observations.shape[0]
        self.num_episodes = len(files)

        # find min and max prices for penalty normalization
        self.min_price_per_episode = self.observations['close_raw'].min()
        self.max_price_per_episode = self.observations['close_raw'].max()
        self.price_amplitude = self.max_price_per_episode - self.min_price_per_episode

        # exclude close_raw
        self.data_columns = self.observations.columns.to_list()
        self.data_columns.remove('close_raw')


    def _get_next_obs(self) -> tuple[np.ndarray, bool]:
        """
        Get the next observation in current episode.

        Returns:
            tuple[np.ndarray, bool]: observation data and stopping flag.
        """
        previous = self.current_step
        obs = np.zeros(shape=(self.observation_shape), dtype=np.float32)
        self.current_step = (self.current_step + 1) % self.episode_len

        done = False
        position = self.position

        if (self.current_step == 0 and previous > 0) or self.current_episode >= self.num_episodes:
            done = True
            position = 0
            self.current_step = -1
            self.current_episode += 1
            self.observations = self._load_file(self.files[self.current_episode])

            if self.current_episode >= self.num_episodes - 1:
                self._reorder_obs()
                self.current_episode = 0

            self._adjust_initial_cash()
        else:
            obs = self.observations.loc[self.current_step, self.data_columns].values

        # Normalize them: 74.5 is mean of 0-149, 43.8816... is std
        norm_step = (self.current_step - 74.5) / 43.88

        obs = np.hstack([obs, np.array([position, norm_step])], dtype=np.float32)

        # print(f'Current file - {self.current_episode} | obs - {obs} |')

        return obs, done


    def step(self, action: tuple) -> tuple[np.ndarray, float, bool, bool, dict]:
        pnl = 0.0

        ratio, close = action
        ratio = np.clip(ratio, -1, 1)

        zero_cash = False

        if close == 0:
            self.zero += 1
        else:
            self.one += 1

        current_obs, done = self._get_next_obs() # CASH NOT ADDED
        if done: close = True

        # third element - close
        self.current_price = self.observations.iloc[self.current_step]['close_raw']
        self.last_not_zero_price = self.current_price if self.current_price != 0 else self.last_not_zero_price

        # Instant reward: position - {-1, 0, 1} -> -1 and price goes up = penalty, 1 and price goes down = penalty.
        # Waiting (0) = nothing. Price price_amplitude guarantees reward in range [-1, 1]
        reward = (self.position * (self.current_price - self.previous_price)) / self.price_amplitude

        truncated = done
        terminated = done
        info = {}

        if close and self.position == 0:
            reward -= 1 # penalty for closing not yet opened contract

        elif close == 1:

            pnl = self.position * (self.current_price - self.entry_price) * self.qty
            pnl = pnl * (1 - self.fee) if pnl > 0 else pnl * (1 + self.fee)

            info.update({
                'pnl': pnl
            })

            # avoid 0 division
            reward += np.clip((pnl / (self.cash + 1e-6)), -2, 10) # idk
            self.cash += pnl

            if self.cash <= 0:
                if market_settings.train:
                    self.cash = self.initial_cash
                    zero_cash = True
                else:
                    truncated = True
                    terminated = True
                    done = True
                    reward = self.min_reward

            if done:
                profit_ratio = self.cash / self.initial_cash * 100
                if profit_ratio > 0: self.profitable_episodes += 1

                info.update({
                    'profit_ratio': profit_ratio,
                    'profitable_episodes': self.profitable_episodes,
                    'last_step': self.current_step
                })

            self.position = 0
            self.qty = 0.0
        elif abs(ratio) > self.ratio_threshold and self.position != 0:
            reward -= 0.5
        elif abs(ratio) <= self.ratio_threshold  and self.position == 0:
            reward -= self.lam # penalty for waiting outside the position
        else:
            if ratio > 0:
                self.entry_price = self.current_price * (1 + self.slippage)
                self.position = 1
            else:
                self.entry_price = self.current_price * (1 - self.slippage)
                self.position = -1

            if self.current_price <= 0:
                self.current_price = self.last_not_zero_price

            self.qty = abs(ratio) * self.cash * (1 - self.fee) / self.current_price

            self.cash -= abs(ratio) * self.cash

            if self.cash == 0:
                reward -= 0.5

        # Update the price
        self.previous_price = self.current_price

        # TODO: Use standardization
        current_obs = np.hstack([current_obs, np.array([self.cash / self.initial_cash])], dtype=np.float32) # Normalized cash was added

        # Clip and scale reward
        reward = np.clip(reward, self.min_reward, self.max_reward) / self.max_reward
        if zero_cash:
            reward = self.min_reward

        return current_obs, reward, truncated, terminated, info


    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        # print(self.zero, self.one)
        self.cash = self.initial_cash
        self.pnl = 0.0
        self.qty = 0.0

        self.position = 0
        self.entry_price = 0.0

        obs, done = self._get_next_obs()
        if done:
            obs, _ = self._get_next_obs()

        self.current_price = self.observations.iloc[0]['close_raw']
        self.previous_price = self.current_price

        obs = np.hstack([obs, np.array([1])], dtype=np.float32)
        return obs, {}
