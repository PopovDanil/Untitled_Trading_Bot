from gymnasium import spaces
import gymnasium as gym
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Tuple

class Market(gym.Env):
    def __init__(self, path_to_training_data: str, initial_cash: np.float32, slippage: np.float32, broker_fee: np.float32):
        super().__init__()

        self.path_to_data = path_to_training_data
        self.data = pd.read_csv(path_to_training_data)
        # observations - open,high,low,close,volume

        # open, high, low, close, volume + additional, + position, current_step, remaining_time
        self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'), shape=self.data.shape[1] + 3, dtype=np.float32)

        # go long - >0, short - <0, hold = 0, value - amount of current cash/futures to spend, second - 0 - wait, 1 - close position
        self.action_space = spaces.Tuple([
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            spaces.Discrete(2)
        ])

        self.cash = initial_cash
        self.pnl = 0.0

        self.slippage = slippage
        self.fee = broker_fee

        self.position = 0 # 0 - waiting, 1 - long, -1 - short
        self.entry_price = 0.0

        self.current_price = data[]
        self.current_step = 0

    def step(self, action: Tuple):
        # action - [float, int]


    def reset(self, seed: int):
        pass

    def show_plot(self) -> None:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=self.data['date'],
                    open=self.data['open'],
                    high=self.data['high'],
                    low=self.data['low'],
                    close=self.data['close']
                )
            ]
        )

        fig.update_layout(
            title='G&P 500 Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )

        fig.show()

m = Market('/home/danil/Documents/ML/Project/Untitled_Trading_Bot/data/extracted/tatasteel_session289.csv', initial_cash=0.0)
m.show_plot()

