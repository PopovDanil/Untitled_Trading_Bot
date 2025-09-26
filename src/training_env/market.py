from gymnasium import spaces
import gymnasium as gym
import pandas as pd
import numpy as np


class Market(gym.Env):
    def __init__(self, path_to_training_data: str, initial_cash: np.float32):
        super().__init__()

        self.path_to_data = path_to_training_data
        self.data = pd.read_csv(path_to_training_data)

        max_value = self.data.max(numeric_only=True).max()
        min_value = self.data.min(numeric_only=True).min()

        self.observation_space = spaces.Box(low=min_value, high=max_value, shape=self.data.shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.stocks = 0.0
        self.cash = initial_cash

        self.current_date = self.data.head(1).index

    def step(self, action: float):
        # action -> float [-1, 1]
        # return - data[i]

        # good:
        # 1. 
        pass

    def reset(self, seed: int):
        pass

