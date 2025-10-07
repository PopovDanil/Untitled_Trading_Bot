import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .utils import file_exists, join_paths
import joblib


# TODO: add golb to find ready files
# TODO: Add scaler per future, since the prices are very
class Data_Scaler:
    def __init__(self, ticket_name: str, saving_path: str = 'data/ready/', scaler_name: str = 'Std'):
        self.ticket = ticket_name
        self.saving_path = saving_path
        self.scaler_name = scaler_name
        self.scaler_path = join_paths(self.scalers_path, ticket_name + ".joblib")
        self.scalers_path = 'data/scalers/'

        if not file_exists(self.scalers_path, ticket_name + ".joblib"):
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

        if not self.use_existing:
            self.__fit_scaler_(data_stacked)
        scaled = self.transform(data)

        file_name = join_paths(self.saving_path, self.ticket + '.npz')

        np.savez(file_name, data=scaled)

        return file_name
