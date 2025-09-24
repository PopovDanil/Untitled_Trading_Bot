import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from utils import get_past_datetime, validate_dates
import pandas_datareader.data as reader
import pandas_ta as ta


# TODO: Add artificial download of at least 26 additional records to handle NaN in MACD index

class DataCollector:
    def __init__(self, stock_name: str = '^GSPC'):
        self.stock_name = stock_name

    # Adjust shapes of main df and additional column
    def __broadcast_data(self, dst: pd.DataFrame, src: pd.DataFrame, column_name: str, **kwargs) -> None:
        values = []

        # Convert strings to dates
        dst_dates = [index.to_pydatetime().date() for index in dst.index]
        src_dates = [index.to_pydatetime().date() for index in src.index]
        src_indexes = [index for index in src.index] # Just for the beauty

        # Pointers to the current instances
        dst_i = 0
        src_i = 0

        # We need to assign value from `src` to the corresponding row in `dst`
        # If current date is between two times, where the statistics was measured,
        # We need to assign it the left bound, else move our boundaries to the right
        while dst_i < len(dst_dates) and src_i < len(src_dates) - 1:
            if src_dates[src_i] <= dst_dates[dst_i] < src_dates[src_i + 1]:
                values.append(src.loc[src_indexes[src_i]])
                dst_i += 1
            else:
                src_i += 1

        # Check the rest, if the last index was never reached
        if dst_i <= (len(dst_dates) - 1):
            value = src.loc[src_indexes[-1]]
            residual = len(dst_dates) - dst_i
            values.extend([value] * residual)

        dst[column_name] = values

    # Add SMA, RSI, MACD
    def __add_micro_indexes(self, data: pd.DataFrame, sma_length: int = 20, rsi_length: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9, **kwargs) -> None:
        # SMA - Simple Moving Average - average over the window of desired length
        data['SMA' + str(sma_length)] = ta.sma(data['Close'], length=sma_length)

        # RSI - Relative Strength Index - defines trends' power and probability of changes
        data['RSI' + str(rsi_length)] = ta.rsi(data['Close'], length=rsi_length)

        # MACD - Moving Average Convergence/Divergence - shows price oscitation
        MACD = ta.macd(data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        for col in MACD.columns:
            data[col] = MACD[col]


    # Fix missing values -> substitute mean. Better approach is
    # To substitute mean of previous and next values, but it will
    # Not greatly affect the performance, since, obviously, not
    # Many great increases/decreases often happen in economy, so
    # All the values will be approximately the same
    def __remove_missing(self, data: pd.DataFrame, **kwargs) -> None:
        for column in data.columns:
            mean = data[column].mean()
            data.fillna({column: mean}, inplace=True)


    # Default stock name - ^GSPC -- S&P 500
    def __collect_data(self, stock_name: str = '^GSPC', period: str = '1mo', interval: str = '1d', start: date | None = None, end: date | None = None, **kwargs) -> pd.DataFrame:

        # Encoder for macro indicators
        macro_indicators = {
            'CPIAUCSL': 'CPI',          # Inflation              per month
            'UNRATE': 'UnemplRate',     # Unemployment Rate      per month
            'FEDFUNDS': 'FedFundsRate', # Federal funds rate     per month
            'GDP': 'GDP',               # GDP                    per 3 months
            'DGS10': '10Y_Treasury',    # 10-year bond yield     per 3 days
            'VIXCLS': 'VIX',            # VIX index              per 3 days
        }

        # Download latest OHLCV (or desired interval chosen by `start` and `end`)
        # OHLCV - Open, High, Low, Close, Volume
        if start is not None and end is not None:
            start, end = validate_dates(start, end, timedelta(days=93))
            data = yf.download(tickers=stock_name, start=start, end=end, multi_level_index=False)
        else:
            start, end = validate_dates(*get_past_datetime(period=period), timedelta(days=93))
            data = yf.download(tickers=stock_name, period=period, interval=interval,  multi_level_index=False)

        if data is None:
            raise RuntimeError('No such stock data')

        # Download Macroeconomics Indicators
        macro = {
            'CPIAUCSL': reader.DataReader(name='CPIAUCSL', data_source='fred', start=start, end=end),
            'UNRATE': reader.DataReader(name='UNRATE', data_source='fred', start=start, end=end),
            'FEDFUNDS': reader.DataReader(name='FEDFUNDS', data_source='fred', start=start, end=end),
            'GDP': reader.DataReader(name='GDP', data_source='fred', start=start, end=end),
            'DGS10': reader.DataReader(name='DGS10', data_source='fred', start=start, end=end),
            'VIXCLS': reader.DataReader(name='VIXCLS', data_source='fred', start=start, end=end),
        }

        # Sometimes FRED does not collect statistics for long time, so validation is required
        # If some indicator is missing, we should use the last one
        for indicator in macro.keys():
            if macro[indicator].empty:
                macro[indicator] = reader.DataReader(name=indicator, data_source='fred').tail(1)

            # Add to the main table and adjust shapes
            self.__broadcast_data(data, macro[indicator][indicator], macro_indicators[indicator])


        return data


    # Main pipeline
    def data_collection_pipeline(self, path: str = 'data', **kwargs) -> pd.DataFrame:
        # Collect fresh data from FRED and Yahoo Finance
        data = self.__collect_data(**kwargs)

        # Add Micro Indexes: SMA, RSI, MACD
        self.__add_micro_indexes(data, **kwargs)

        # Handle missing values
        self.__remove_missing(data, **kwargs)

        # Save
        data.to_csv(path + '/training.csv')

        return data


collector = DataCollector()
data = collector.data_collection_pipeline(period='1y')
print(data)