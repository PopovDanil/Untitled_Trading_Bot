import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from .utils import get_past_datetime, validate_dates
import pandas_datareader.data as reader
import pandas_ta as ta
from typing import Tuple

# TODO: Fix args for pipeline
# TODO: Add comments

# TODO: I really need to stick to one stock? If there is only one optimal strategy, so it should be uniform
# TODO: Split into train/test/validation
# TODO: Add PER


class Data_Collector:
    def __init__(
            self, stock_name: str = '^GSPC',
            file_save_to: str = 'data/raw/',
            period: str = '1d',
            interval: str = '1m',
            start: date | None = None,
            end: date | None = None,
            add_macro: bool = False,
            add_micro: bool = False,
            sma_length: int = 20,
            rsi_length: int = 14,
            macd_fast: int = 12,
            macd_slow: int = 26,
            macd_signal: int = 9,
        ):
        self.stock_name = stock_name    # The main stock or future name for collector
        self.file_save_to = file_save_to  # Path to save downloaded and generated data
        self.period = period            # Period to download data. Valid: 1d, 5d, 1mo, 3mo, 6mo, 1y
        self.interval= interval         # The frequency of data observations. Valid: 1m, 1d, 2d
        self.start = start              # If you want to get data about specific dates. Begin of collection
        self.end = end                  # If you want to get data about specific dates. End of collection
        self.add_macro = add_macro      # Include macro indicators
        self.add_micro = add_micro      # Include micro indicators
        self.sma_length = sma_length    # Simple mean average window size
        self.rsi_length = rsi_length    # RSI window size
        self.macd_fast = macd_fast      # MACD
        self.macd_slow = macd_slow      # MACD
        self.macd_signal = macd_signal  # MACD
        print(self.start, self.end)

    # Adjust shapes of main df and additional column
    def __broadcast_data(self, dst: pd.DataFrame, src: pd.DataFrame, column_name: str, *args, **kwargs) -> None:
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
    def __add_micro_indexes(self, data: pd.DataFrame, *args, **kwargs) -> None:
        # SMA - Simple Moving Average - average over the window of desired length
        data['SMA' + str(self.sma_length)] = ta.sma(data['Close'], length=self.sma_length)

        # RSI - Relative Strength Index - defines trends' power and probability of changes
        data['RSI' + str(self.rsi_length)] = ta.rsi(data['Close'], length=self.rsi_length)

        # MACD - Moving Average Convergence/Divergence - shows price oscitation
        MACD = ta.macd(data['Close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        for col in MACD.columns:
            data[col] = MACD[col]


    # Fix missing values -> substitute mean. Better approach is
    # To substitute mean of previous and next values, but it will
    # Not greatly affect the performance, since, obviously, not
    # Many great increases/decreases often happen in economy, so
    # All the values will be approximately the same
    def __remove_missing(self, data: pd.DataFrame, *args, **kwargs) -> None:
        for column in data.columns:
            mean = data[column].mean()
            data.fillna({column: mean}, inplace=True)


    # Downloads the main data + 26 days before the interval to
    # Avoid NaN in MACD
    def __download_main_data(self, *args, **kwargs) -> pd.DataFrame | None:

        # Valid periods based on the yf documentation
        transitions = {
            '1d': '5d',
            '5d': '1mo',
            '1mo': '3mo',
            '3mo': '6mo',
            '6mo': '1y',
            '1y': '2y'
        }

        valid_period = transitions[self.period]

        data = yf.download(tickers=self.stock_name, period=valid_period, interval=self.interval,  multi_level_index=False)

        return data

    # Default stock name - ^GSPC -- S&P 500
    def __collect_data(self, *args, **kwargs) -> pd.DataFrame:

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
        if self.start is not None and self.end is not None:
            # This part is needed if you want to download data per minutes
            delta = timedelta(days=120 if self.interval[-1] != 'm' else 1)
            start, end = validate_dates(self.start, self.end, delta)

            data = yf.download(tickers=self.stock_name, start=start, end=end, interval=self.interval, multi_level_index=False)

            # This will ensure that GDP was measured
            start, end = validate_dates(start, end, timedelta(days=120))
        else:
            start, end = validate_dates(*get_past_datetime(period=self.period), timedelta(days=120))
            data = self.__download_main_data(tickers=self.stock_name, period=self.period, interval=self.interval)

        if data is None:
            raise RuntimeError('No such stock data')

        if self.add_macro:
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

    # Since during data collection we have downloaded more then we initially wanted
    # We need to crop it
    def __crop_to_desired_dates(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        start = self.start
        end = self.end

        # Check if exact dates were specified, in other case - define them from period
        if start is None or end is None:
            start, end = get_past_datetime(period=self.period)

        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        # Cast dates to avoid errors
        df.index = pd.to_datetime(df.index, utc=True)

        filtered_df = df.loc[(df.index >= start) & (df.index <= end)]

        # To follow same style in case of minutes
        filtered_df.index.rename('Date', inplace=True)

        return filtered_df

    # Main pipeline
    def collect_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        # Collect fresh data from FRED and Yahoo Finance
        # If something goes wrong, there is no need to continue, so we can terminate

        # Big bro don't use error handling
        data = self.__collect_data(**kwargs)

        # Add Micro Indexes: SMA, RSI, MACD
        if self.add_micro:
            self.__add_micro_indexes(data, **kwargs)

        # Handle missing values
        self.__remove_missing(data, **kwargs)

        # Extract desired period
        cropped = self.__crop_to_desired_dates(data, **kwargs)

        # Save
        if not cropped.empty:
            cropped.to_csv(self.file_save_to)

        return cropped, cropped.empty
