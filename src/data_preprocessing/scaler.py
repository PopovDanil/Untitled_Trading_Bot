import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import file_exists, join_paths, get_files
import joblib


class Data_Scaler:
    def __init__(self, ticker: str, saving_path: str = 'data/ready/', scaler_name: str = 'Std'):
        self.ticker = ticker
        self.saving_path = saving_path
        self.scaler_name = scaler_name
        self.scalers_path = 'data/scalers/'
        self.scaler_path = join_paths(self.scalers_path, ticker + ".joblib")

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
            df = pd.read_csv(file)
            df = df.drop(columns=['date'])
            data = pd.concat([data, df], axis=0)

        return data.values

    def __transform(self, data, *args, **kwargs):
        scaled = []

        for file in self.files:
            df = pd.read_csv(file, index_col=0)
            df = df.drop(columns=['date'])
            scaled.append(self.scaler.transform(df.values))

        return np.array(scaled)

    def scale_data(self, path_to_data: str = 'data/extracted/'):
        data = self.__collect_data(path_to_data)

        try:
            if not self.use_existing:
                self.__fit_scaler_(data)
            scaled = self.__transform(data)

            file_name = join_paths(self.saving_path, self.ticker + '.npz')

            np.savez(file_name, data=scaled)

            return file_name
        except ValueError as e:
            print(f'An error occurred: {e}')

d = Data_Scaler(ticker='zyduslife_session122', saving_path='data')
d.scale_data()