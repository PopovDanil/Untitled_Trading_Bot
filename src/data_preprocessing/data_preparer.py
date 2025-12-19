from datetime import date, timedelta
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pandas_datareader.data as reader
import pandas_ta as ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from .utils import (
    extract_ticker_name,
    file_exists,
    get_files,
    get_past_datetime,
    join_paths,
    validate_dates,
)


class Data_Collector:
    def __init__(
        self,
        ticker: str = '^GSPC',
        period: str = '1d',
        interval: str = '1m',
        start: date | None = None,
        end: date | None = None,
        ) -> None:
        """
        Initializes data downloader for specified ticker.

        Args:
            ticker (str, optional): Ticker name. Defaults to '^GSPC'.
            period (str, optional): Time period (from current time) for data collection. Defaults to '1d'.
            interval (str, optional): Interval between observations. Defaults to '1m'.
            start (date | None, optional): For specific dates. Requires end specification. Defaults to None.
            end (date | None, optional):For specific dates. Requires start specification. Defaults to None.
        """
        self.ticker = ticker
        self.period = period
        self.interval= interval
        self.start = start
        self.end = end

    def __download_main_data(self) -> pd.DataFrame | None:
        """
        Downloads ticker observations for specified period (or dates) + 26 additional observations for MACD.

        Raises:
            RuntimeError: Observations ware not downloaded.

        Returns:
            pd.DataFrame | None: Observations in case of success.
        """

        if self.start is not None and self.end is not None:
            # This part is needed if you want to download data per minutes
            delta = timedelta(days=120 if self.interval[-1] != 'm' else 1)
            start, end = validate_dates(self.start, self.end, delta)

            data = yf.download(tickers=self.stock_name, start=start, end=end, interval=self.interval, multi_level_index=False)

            # This will ensure that GDP was measured
            start, end = validate_dates(start, end, timedelta(days=122))
        else:
            start, end = validate_dates(*get_past_datetime(period=self.period), timedelta(days=120))
            data = self.__download_main_data(tickers=self.stock_name, period=self.period, interval=self.interval)

        transitions = {
            '1d': '5d',
            '5d': '1mo',
            '1mo': '3mo',
            '3mo': '6mo',
            '6mo': '1y',
            '1y': '2y'
        }

        valid_period = transitions[self.period]

        data = yf.download(tickers=self.ticker, period=valid_period, interval=self.interval,  multi_level_index=False)

        if data is None:
            raise RuntimeError("No data for the specified dates.")

        return data

    def __download_macro_indicators(self) -> pd.DataFrame:

        # Encoder for macro indicators
        macro_indicators = {
            'CPIAUCSL': 'CPI',          # Inflation              per month
            'UNRATE': 'UnemplRate',     # Unemployment Rate      per month
            'FEDFUNDS': 'FedFundsRate', # Federal funds rate     per month
            'GDP': 'GDP',               # GDP                    per 3 months
            'DGS10': '10Y_Treasury',    # 10-year bond yield     per 3 days
            'VIXCLS': 'VIX',            # VIX index              per 3 days
        }

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

    def __format_df(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.reset_index(drop=False)
        new_data = {
            'date':   data['Date'],
            'open':   data['Open'],
            'high':   data['High'],
            'low':    data['Low'],
            'close':  data['Close'],
            'volume': data['Volume'],
        }

        new_df = pd.DataFrame(new_data)
        return new_df

class Data_Collector_1:
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
        ) -> None:
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
        cropped = self.__format_df(cropped)

        # Save
        if not cropped.empty:
            cropped.to_csv(self.file_save_to)

        return cropped, cropped.empty


class Data_Extractor:
    def __init__(self,
            saving_path: str = 'data/extracted/',
            window_size: int = 10,
            pre_session_size: int = 90,
            after_session_interval: int = 60,
            min_volatility: float = 0.03,
            scaling_factor: float = 1.5
        ):
        self.saving_path = saving_path
        self.window_size = window_size
        self.pre_session_size = pre_session_size
        self.after_session_interval = after_session_interval
        self.min_volatility = min_volatility
        self.scaling_factor = scaling_factor

    def __calculate_volatility(self, data, *args, **kwargs) -> pd.Series | np.ndarray:

        if isinstance(data, np.ndarray):
            df = pd.DataFrame({'Close': data.copy()})
        else:
            df = data.copy(deep=True)

        close = df['Close'] if 'Close' in df.columns else df['close']
        return np.abs(close.shift(-self.window_size) - close) / close


    def __extract_intervals(self, data: pd.DataFrame, ticker: str, *args, **kwargs) -> None:
        chosen_sessions = []

        volatility = self.__calculate_volatility(data, self.window_size)
        potential_sessions = volatility[volatility >= self.min_volatility]

        for end, session_vol in potential_sessions.items():
            start = end - self.pre_session_size

            if start < 0 or end + self.after_session_interval + 1 > data.shape[0]:
                continue

            if 'Close' in data.columns:
                session_closes = data['Close'].iloc[start : end].values
            else:
                session_closes = data['close'].iloc[start : end].values

            filter = np.mean(self.__calculate_volatility(session_closes, self.window_size))

            if filter < (session_vol / self.scaling_factor):
                chosen_sessions.append(data.iloc[start : end + self.after_session_interval])

        for idx, session in enumerate(chosen_sessions):
            if not session.empty:
                path = join_paths(self.saving_path, f'{ticker}_session{idx + 1}.csv')
                session.to_csv(path)

    def __drop_zero_price(self, data: pd.DataFrame) -> None:
        zeros = data[data['close'] <= 0.0]
        data.drop(zeros.index, inplace=True)

    def extract_data(self, files: List[str], *args, **kwargs) -> None:

        for file in tqdm(files):
            data = pd.read_csv(file)
            ticker = extract_ticker_name(file)
            self.__drop_zero_price(data)
            self.__extract_intervals(data, ticker, self.min_volatility)


class Data_Scaler:
    def __init__(self, ticker: str, saving_path: str = 'data/ready/', scaler_name: str = 'Std'):
        self.ticker = ticker
        self.saving_path = saving_path
        self.scaler_name = scaler_name
        self.scalers_path = 'data/scalers/'
        self.scaler_path = join_paths(self.scalers_path, ticker + ".joblib")
        self.files = None

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
            df = pd.read_csv(file, index_col=0).drop(columns=['date'])
            data = pd.concat([data, df], axis=0)

        return data.values

    def __transform_zeros(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = []
        index = list(data.index)
        for i in range(len(index)):
            idx = index[i]
            prev = index[i] if i == 0 else index[i - 1]
            nxt = index[i] if i == len(index) - 1 else index[i + 1]
            if data[idx] == 0.0:
                new_data.append(data[prev] / 2 + data[nxt] / 2)
            else:
                new_data.append(data[idx])

        return new_data

    def __transform(self, to_transform: pd.DataFrame | pd.Series = None, save: bool = True) -> np.ndarray | None:

        read_files = True
        to_return = None
        if self.files is None:
            read_files = False
            self.files = [to_transform]

        for file in self.files:
            df = pd.read_csv(file, index_col=0).drop(columns=['date']) if read_files else file

            data = self.scaler.transform(df.values)
            data = pd.DataFrame(data=data, columns=df.columns)
            data['close_raw'] = df['close'].values # self.__transform_zeros(df['close'])

            if save:
                file_to_save = join_paths(self.saving_path, f'{self.ticker}_{str(uuid4())}.csv')
                data.to_csv(file_to_save)
            to_return = data

        return  to_return

    def scale_data(self, df: pd.DataFrame | pd.Series = None, path_to_data: str = 'data/extracted/', save: bool = True) -> None | np.ndarray:
        data = self.__collect_data(path_to_data) if df is None else df

        try:
            if not self.use_existing:
                self.__fit_scaler_(data)
            scaled = self.__transform(data, save)

            return scaled
        except ValueError as e:
            print(f'An error occurred: {e}')

