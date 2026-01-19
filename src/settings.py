import os

import torch
from pydantic_settings import BaseSettings

DIR = os.getcwd()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentSettings(BaseSettings):
    plot_path: str = os.path.join('plots', 'ppo')

class MarketSettings(BaseSettings):
    train: bool = False


agent_settings = AgentSettings()
market_settings = MarketSettings()