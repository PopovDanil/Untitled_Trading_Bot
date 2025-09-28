import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


class Data_Normalizer:
    def __init__(self, normalizer_name: str):
        self.normalizer_name = normalizer_name
        self.normalizer = {
            'MinMax': MinMaxScaler(),
            'Std': StandardScaler()
        }[normalizer_name]

    def __fit_scaler_(self, data) -> np.ndarray:
        self.normalizer.fit(data)
        joblib.dump(self.normalizer, 'data/scaler/conf.joblib')

    def transform(self, data):
        normalized = []

        if len(data.shape) <= 2:
            return self.normalizer.transform(data)

        for idx in range(data.shape[0]):
            normalized.append(self.normalizer.transform(data[idx]))

        return np.array(normalized)

    def normalizer_pipeline(self, path_to_data: str = 'data/ready/'):
        data = np.load(path_to_data, allow_pickle=True)['data']

        data_stacked = np.concat([data[idx] for idx in range(data.shape[0])], axis=0)
        self.__fit_scaler_(data_stacked)
        normalized = self.transform(data)

        np.savez('data/ready/ext', data=normalized)


d = Data_Normalizer('Std')
d.normalizer_pipeline('/home/danil/Documents/ML/Project/Untitled_Trading_Bot/data/raw/ext.npz')