import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


# TODO: add golb to find ready files

class Data_Scaler:
    def __init__(self, saving_path: str = 'data/ready/', scaler_name: str = 'Std'):
        self.saving_path = saving_path
        self.scaler_name = scaler_name
        self.scaler = {
            'MinMax': MinMaxScaler(),
            'Std': StandardScaler()
        }[scaler_name]

    def __fit_scaler_(self, data, *args, **kwargs) -> np.ndarray:
        self.scaler.fit(data)
        joblib.dump(self.scaler, 'data/scaler/conf.joblib')

    def transform(self, data, *args, **kwargs):
        scaled = []

        if len(data.shape) <= 2:
            return self.scaler.transform(data)

        for idx in range(data.shape[0]):
            scaled.append(self.scaler.transform(data[idx]))

        return np.array(scaled)

    def scale_data(self, path_to_data: str = 'data/extracted/'):
        data = np.load(path_to_data, allow_pickle=True)['data']

        data_stacked = np.concat([data[idx] for idx in range(data.shape[0])], axis=0)
        self.__fit_scaler_(data_stacked)
        scaled = self.transform(data)

        file_name = self.saving_path + 'ready.npz'

        np.savez(file_name, data=scaled)

        return file_name
