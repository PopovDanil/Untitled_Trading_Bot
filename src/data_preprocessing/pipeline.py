import numpy as np

from .collector import Data_Collector
from .extractor import Data_Extractor
from .scaler import Data_Scaler
from .utils import (
    daterange,
    discard_files,
    extract_ticker_name,
    get_files,
    get_past_datetime,
    join_paths,
)

futures_tickers = [
    "ES=F",   # E-Mini S&P 500
    "YM=F",   # Mini Dow Jones
    "NQ=F",   # Nasdaq-100
    "RTY=F",  # Russell 2000
    "ZB=F",   # U.S. Treasury Bond Futures
    "ZN=F",   # 10-Year T-Note Futures
    "ZF=F",   # 5-Year U.S. Treasury Note Futures
    "ZT=F",   # 2-Year T-Note Futures
    "GC=F",   # Gold Futures
    "MGC=F",  # Micro Gold Futures
    "SI=F",   # Silver Futures
    "SIL=F",  # Micro Silver Futures
    "PL=F",   # Platinum Futures
    "HG=F",   # Copper Futures
    "PA=F",   # Palladium Futures
]

def compress_to_one(path: str, on_delete: bool = False) -> None:
    files = get_files(path, pattern='*.npz')
    result = []

    for file in files:
        arr = np.load(file, allow_pickle=True)['data']

        if result == []:
            result = arr
        else:
            result = np.vstack([result, arr])

    if on_delete:
        discard_files(files)

    np.savez(join_paths(path, 'ready'), data=result)


def update_training_dataset(period: str = '1mo') -> None:
    # Get dates for data collection
    start, end = get_past_datetime(period=period)

    # Collect futures
    for ticker in futures_tickers:

        # Initialize perprocessers
        extractor = Data_Extractor(min_volatility=0.03, scaling_factor=1.5)
        scaler = Data_Scaler(ticker=ticker)

        collected_data_files = []

        for date_s, date_e in daterange(start, end):
            file_name = f'data/raw/{ticker}_{str(date_s)}-{str(date_e)}.csv'

            try:
                collector = Data_Collector(file_save_to=file_name, stock_name=ticker, start=date_s, end=date_e)
                _, empty = collector.collect_data()

                if not empty: collected_data_files.append(file_name)
            except Exception:
                # TODO: add something
                pass

        extractor.extract_data(files=collected_data_files)
        scaler.scale_data(path_to_data=extractor.saving_path)

        # Clean the raw data storage
        discard_files(collected_data_files)

    print('Your beautiful data is ready!')


# cut first n rows in the list, and add fresh
def merge_new_and_old():
    pass


def preprocess_dataset(directory: str = 'data/train') -> None:
    files = get_files(dir=directory)

    for file in files:
        extractor = Data_Extractor(min_volatility=0.06, scaling_factor=5.0)
        scaler = Data_Scaler(ticker=extract_ticker_name(file))
        extractor.extract_data(files=[file])
        scaler.scale_data(path_to_data=extractor.saving_path)
