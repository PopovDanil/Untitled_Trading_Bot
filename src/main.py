from data_preprocessing.pipeline import update_training_dataset, preprocess_dataset


if __name__ == '__main__':
    # update_training_dataset(period='1mo')
    preprocess_dataset(directory='data/train')