import numpy as np
import pandas as pd
from typing import List


class Data_Extractor:
    def __init__(self):
        pass

    def __calculate_volatility(self, data, window_size: int) -> pd.Series:

        if isinstance(data, np.ndarray):
            df = pd.DataFrame({'Close': data.copy()})
        else:
            df = data.copy(deep=True)

        return np.abs(df['Close'].shift(-window_size) - df['Close']) / df['Close']


    def __extract_intervals(self, data: pd.DataFrame, min_volatility: float, window_size: int = 10, pre_session_size: int = 90, after_session_interval: int = 60, scaling_factor: float = 1.5) -> np.ndarray:
        chosen_sessions = []

        volatility = self.__calculate_volatility(data, window_size)
        potential_sessions = volatility[volatility >= min_volatility]

        for end, session_vol in potential_sessions.items():
            start = end - pre_session_size

            if start < 0 or end + after_session_interval + 1 > data.shape[0]:
                continue

            session_closes = data['Close'].iloc[start : end].values
            filter = np.mean(self.__calculate_volatility(session_closes, window_size))

            if filter < (session_vol / scaling_factor):
                chosen_sessions.append(data.iloc[start : end + after_session_interval].values)

        # drop first column with dates
        return np.array(chosen_sessions)[:, :, 1:]

    def extractor_pipeline(self, files: List[str], min_volatility: float = 0.0015) -> None:

        extracted = []

        for file in files:
            try:
                data = pd.read_csv(file)
                extracted.extend(self.__extract_intervals(data, min_volatility))

            except Exception as e:
                print(f'Sorry, something went wrong: {e}. Proceeding with the other files.')

        extracted = np.array(extracted)

        np.savez('data/raw/ext', data=extracted)

ext = Data_Extractor()
ext.extractor_pipeline(files=['/home/danil/Documents/ML/Project/Untitled_Trading_Bot/data/training.csv'])