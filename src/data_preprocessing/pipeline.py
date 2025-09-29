from .collector import Data_Collector
from .extractor import Data_Extractor
from .scaler import Data_Scaler
from .utils import get_past_datetime, daterange


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


def get_data(period: str = '4y') -> str:
    # Get dates for data collection
    start, end = get_past_datetime(period=period)

    # Initialize perprocessers
    extractor = Data_Extractor()
    scaler = Data_Scaler()

    collected_data_file_names = []

    # Collect futures
    for ticket in futures_tickers:
        for date_s, date_e in daterange(start, end):
            file_name = f'data/raw/{str(date_s)}-{str(date_e)}_{ticket}.csv'

            try:
                collector = Data_Collector(file_save_to=file_name, stock_name=ticket, start=date_s, end=date_e)
                collector.collect_data()

                collected_data_file_names.append(file_name)
            except Exception as e:
                pass

    file = extractor.extract_data(files=collected_data_file_names)
    file = scaler.scale_data(file)

    print(f'Your beautiful data is ready! You can find it here: {file}')

    return file