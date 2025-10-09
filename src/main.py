from data_preprocessing.pipeline import update_training_dataset, preprocess_dataset, compress_to_one


if __name__ == '__main__':
    # compress_to_one('data/ready')
    # update_training_dataset(period='1mo')
    preprocess_dataset(directory='data/train')