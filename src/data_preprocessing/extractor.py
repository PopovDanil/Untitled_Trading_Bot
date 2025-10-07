import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm


# TODO: add scaling per ticket

class Data_Extractor:
    def __init__(self,
            saving_path: str = 'data/extracted/',
            window_size: int = 10,
            pre_session_size: int = 90,
            after_session_interval: int = 60,
            min_volatility: float = 0.03,
            scaling_factor: float = 1.5
        ):
        self.saving_path = saving_path
        self.window_size = window_size
        self.pre_session_size = pre_session_size
        self.after_session_interval = after_session_interval
        self.min_volatility = min_volatility
        self.scaling_factor = scaling_factor

    def __calculate_volatility(self, data, *args, **kwargs) -> pd.Series:

        if isinstance(data, np.ndarray):
            df = pd.DataFrame({'Close': data.copy()})
        else:
            df = data.copy(deep=True)

        close = df['Close'] if 'Close' in df.columns else df['close']
        return np.abs(close.shift(-self.window_size) - close) / close


    def __extract_intervals(self, data: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        chosen_sessions = []

        volatility = self.__calculate_volatility(data, self.window_size)
        potential_sessions = volatility[volatility >= self.min_volatility]

        for end, session_vol in potential_sessions.items():
            start = end - self.pre_session_size

            if start < 0 or end + self.after_session_interval + 1 > data.shape[0]:
                continue

            # TODO: remove KOSTYLY
            if 'Close' in data.columns:
                session_closes = data['Close'].iloc[start : end].values
            else:
                session_closes = data['close'].iloc[start : end].values

            filter = np.mean(self.__calculate_volatility(session_closes, self.window_size))

            if filter < (session_vol / self.scaling_factor):
                print(data.iloc[start : end + self.after_session_interval])
                chosen_sessions.append(data.iloc[start : end + self.after_session_interval].values)

        # drop first column with dates
        try:
            result = np.array(chosen_sessions)[:, :, 1:]
            return result
        except:
            return np.array([])

    def extract_data(self, files: List[str], *args, **kwargs) -> str:

        extracted = []

        for file in tqdm(files):
            data = pd.read_csv(file)
            extracted.extend(self.__extract_intervals(data, self.min_volatility))

        extracted = np.array(extracted)

        file_name = self.saving_path + 'extracted.npz'

        np.savez(file_name, data=extracted)

        return file_name


d = Data_Extractor()
d.extract_data(['/home/danil/Documents/ML/Project/Untitled_Trading_Bot/data/train/TATASTEEL_5minute.csv'])