from gymnasium import spaces
import gymnasium as gym
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class Market(gym.Env):
    def __init__(self, path_to_training_data: str, initial_cash: np.float32):
        super().__init__()

        self.path_to_data = path_to_training_data
        self.data = pd.read_csv(path_to_training_data)

        max_value = self.data.max(numeric_only=True).max()
        min_value = self.data.min(numeric_only=True).min()

        self.observation_space = spaces.Box(low=min_value, high=max_value, shape=self.data.shape, dtype=np.float32) # add current cash, volume of stocks
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.stocks = 0.0
        self.cash = initial_cash

        self.current_date = self.data.head(1).index

    def step(self, action: float):
        # action -> float [-1, 1]
        # return - data[i]

        # Ides:
        # For selling
        # 1. Check previous and future prices to give immediate reward
        # 2. For the first one - add the same price, for the last - give
        #
        pass

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

