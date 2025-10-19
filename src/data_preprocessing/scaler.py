from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import file_exists, get_files, join_paths


class Data_Scaler:
    def __init__(self, ticker: str, saving_path: str = 'data/ready/', scaler_name: str = 'Std'):
        self.ticker = ticker
        self.saving_path = saving_path
        self.scaler_name = scaler_name
        self.scalers_path = 'data/scalers/'
        self.scaler_path = join_paths(self.scalers_path, ticker + ".joblib")
        self.files = None

        if not file_exists(self.scalers_path, ticker + ".joblib"):
            self.use_existing = False
            self.scaler = {
                'MinMax': MinMaxScaler(),
                'Std': StandardScaler()
            }[scaler_name]
        else:
            self.use_existing = True
            self.scaler = self.__load_scaler()

    def __load_scaler(self) -> StandardScaler | MinMaxScaler | None:
        scaler = None
        try:
            scaler = joblib.load(self.scaler_path)
        except Exception as e:
            print(f"Error while loading the scaler: {e}")

        return scaler

    def __fit_scaler_(self, data, *args, **kwargs) -> np.ndarray:
        self.scaler.fit(data)
        joblib.dump(self.scaler, self.scaler_path)

    def __collect_data(self, path_to_data: str):
        pattern = self.ticker + "*.csv"
        self.files = get_files(path_to_data, pattern)

        data = pd.DataFrame()
        for file in self.files:
            df = pd.read_csv(file, index_col=0).drop(columns=['date'])
            data = pd.concat([data, df], axis=0)

        return data.values

    def __transform_zeros(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = []
        index = list(data.index)
        for i in range(len(index)):
            idx = index[i]
            prev = index[i] if i == 0 else index[i - 1]
            nxt = index[i] if i == len(index) - 1 else index[i + 1]
            if data[idx] == 0.0:
                new_data.append(data[prev] / 2 + data[nxt] / 2)
            else:
                new_data.append(data[idx])

        return new_data

    def __transform(self, to_transform: pd.DataFrame | pd.Series = None, save: bool = True) -> np.ndarray | None:

        read_files = True
        to_return = None
        if self.files is None:
            read_files = False
            self.files = [to_transform]

        for file in self.files:
            df = pd.read_csv(file, index_col=0).drop(columns=['date']) if read_files else file

            data = self.scaler.transform(df.values)
            data = pd.DataFrame(data=data, columns=df.columns)
            data['close_raw'] = df['close'].values # self.__transform_zeros(df['close'])

            if save:
                file_to_save = join_paths(self.saving_path, f'{self.ticker}_{str(uuid4())}.csv')
                data.to_csv(file_to_save)
            to_return = data

        return  to_return

    def scale_data(self, df: pd.DataFrame | pd.Series = None, path_to_data: str = 'data/extracted/', save: bool = True) -> None | np.ndarray:
        data = self.__collect_data(path_to_data) if df is None else df

        try:
            if not self.use_existing:
                self.__fit_scaler_(data)
            scaled = self.__transform(data, save)

            return scaled
        except ValueError as e:
            print(f'An error occurred: {e}')
