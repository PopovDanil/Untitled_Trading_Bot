import os
import random

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from utils import split_period_into_days
from yfinance import download

tickers = [
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


class Collector:
    """
    Collects data either form file (passing path to the desired .csv) or by downloading from yfinace.
    """
    def __init__(
        self,
        ticker: str,
        period: str = '59d',
        ) -> None:
        """
        Initializes downloader instance for specified ticker.

        Args:
            ticker (str): ticker name
            period (str, optional): period (from current moment) for data searching.
                Must be in format 'Xd'. Defaults to '59d'.
        """
        self.ticker = ticker
        self.period = period


    def download_main_data(self) -> pd.DataFrame:
        """
        Downloads 1 minute data for each day in the period.

        Returns:
            pd.DataFrame: stacked dataframe with observations. Date is index column.
        """
        required_days = split_period_into_days(self.period)

        data = pd.DataFrame()
        for (start, end) in required_days:
            df = download(tickers=self.ticker, start=start, end=end, interval='1m', multi_level_index=False, progress=False)
            if df is None:
                print(f'Missing data for {start}!')
                continue

            df['date'] = [date.to_pydatetime() for date in df.index]

            data = pd.concat([data, df], axis=0)

        data = data.reset_index(drop=True)
        date = data['date']
        data = data.drop(columns=['date'])
        data.index = date

        return data


    def load_from_file(self, file: str) -> pd.DataFrame:
        """
        Loads observations from .csv file.

        Args:
            file (str): path.

        Returns:
            pd.DataFrame: stacked dataframe with observations. Date is index column.
        """
        df = pd.read_csv(file)
        df = df.set_index('date')
        return df


    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats columns and index: brings columns to lowercase and formats index to YYYY-MM-DD HH:MM.

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: formatted dataframe. Date is index column.
        """
        new_df = pd.DataFrame()

        for column in df.columns:
            formatted_name = column.lower()

            if formatted_name not in ['date', 'open', 'high', 'low', 'close', 'volume']:
                continue

            new_df[formatted_name] = df[column]

        return new_df


    def collect(self, file_path: str | None = None) -> pd.DataFrame:
        """
        Loads data. Either downloads (file_path is None) or loads from file.

        Args:
            file_path (str | None, optional): path to data if downloading is skipped. Defaults to None.

        Returns:
            pd.DataFrame: loaded data.
        """
        df = None
        if file_path is not None:
            df = self.load_from_file(file=file_path)
        else:
            df = self.download_main_data()

        return self._format_df(df)


class Preprocessor:
    """
    Fills missing values, log-scales, adds micro indexes.
    """
    def __init__(
            self,
            sma_length: int = 20,
            rsi_length: int = 14,
            macd_fast: int = 12,
            macd_slow: int = 26,
            macd_signal: int = 9
        ) -> None:
        """
        Initializes instance

        Args:
            sma_length (int, optional): Simple mean average window size. Defaults to 20.
            rsi_length (int, optional):  RSI window size. Defaults to 14.
            macd_fast (int, optional): MACD fast. Defaults to 12.
            macd_slow (int, optional): MACD slow. Defaults to 26.
            macd_signal (int, optional): MACD signal. Defaults to 9.
        """
        self.sma_length = sma_length
        self.rsi_length = rsi_length
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal


    def _fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filles missing values with mean per feature.

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: dataframe.
        """
        filled = df.copy()

        for column in df.columns:
            if column == 'date':
                continue

            mean = df[column].mean()
            filled[column] = filled[column].fillna(mean)

        return filled


    def _add_micro_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds micro indexes (RSI, SMA, MACD):
            SMA - Simple Moving Average - average over the window of desired length.
            RSI - Relative Strength Index - defines trends' power and probability of changes.
            MACD - Moving Average Convergence/Divergence - shows price oscitation.

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: modified dataframe.
        """
        data = df.copy()

        data['sma' + str(self.sma_length)] = ta.sma(data['close'], length=self.sma_length)

        data['rsi' + str(self.rsi_length)] = ta.rsi(data['close'], length=self.rsi_length)

        macd = ta.macd(data['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        for col in macd.columns:
            data[col.lower()] = macd[col]

        return data


    def _apply_log_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log-scaling for value stability. Adds log-returns.

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: modified dataframe.
        """
        data = df.copy()

        data['log_close'] = np.log(data['close'])
        data['log_high'] = np.log(data['high'])
        data['log_low'] = np.log(data['low'])
        data['log_volume'] = np.log(data['volume'] + 1) # to avoid -inf

        data['log_return'] = np.log(df['close'] / df['close'].shift(1))

        return data


    def _drop_incomplete(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops incomplete rows (edge missing values after micro indexes windows).

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: modified dataframe.
        """
        return df.dropna()


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combined operations.

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: modified dataframe.
        """
        result = self._fill_na(df)
        result = self._apply_log_scaling(result)
        result = self._add_micro_indexes(result)
        result = self._drop_incomplete(result)
        return result


class Scaler:
    """
    Normalizes features.
    """
    def __init__(self, ticker: str) -> None:
        """
        Initializes instance.

        Args:
            ticker (str): ticker name
        """
        self.ticker = ticker
        self.scaler_path = os.path.join('data', 'scalers', ticker + '.joblib')

        # Check if we have already used scaler for ticker. If no - initialize a new one.
        if not os.path.exists(self.scaler_path):
            self.use_existing = False
            self.scaler = StandardScaler()
        else:
            self.use_existing = True
            self.scaler = self._load_scaler() # load existing


    def _load_scaler(self) -> StandardScaler:
        """
        Loads found scaler.

        Raises:
            RuntimeError: joblib failure.

        Returns:
            StandardScaler: loaded scaler.
        """
        scaler = None
        try:
            scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Error while loading the scaler: {e}")

        return scaler


    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """
        Fits new scaler.

        Args:
            df (pd.DataFrame): dataframe.
        """
        self.scaler.fit(df)
        joblib.dump(self.scaler, self.scaler_path)


    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies normalization.

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: normalized dataframe.
        """
        df_without_dates = df.reset_index(drop=True)

        # Can be skipped, since in pipeline scaler will be pre-fitted.
        if not self.use_existing:
            self._fit_scaler(df_without_dates)

        df_std = self.scaler.transform(df_without_dates)
        df_std = pd.DataFrame(data=df_std, columns=df_without_dates.columns)

        df_std.index = df.index
        df_std['close_raw'] = df['close']

        return df_std


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        API.

        Args:
            df (pd.DataFrame): dataframe.

        Returns:
            pd.DataFrame: dataframe.
        """
        result = self._normalize(df)
        return result


class Extractor:
    def __init__(self,
            window_size: int = 10,
            pre_session_size: int = 90,
            after_session_interval: int = 60,
            min_volatility: float = 0.05,
            scaling_factor: float = 5
        ) -> None:
        self.window_size = window_size
        self.pre_session_size = pre_session_size
        self.after_session_interval = after_session_interval
        self.min_volatility = min_volatility
        self.scaling_factor = scaling_factor


    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        return abs(close.shift(-self.window_size) - close) / close


    def _extract_intervals(self, data: pd.DataFrame) -> list[pd.DataFrame]:
        df = data.reset_index(drop=True)

        chosen_sessions = []
        last_used = float('-inf')

        volatility = self._calculate_volatility(df)
        potential_sessions = volatility[volatility >= self.min_volatility]

        for (end, session_vol) in potential_sessions.items():

            start = end - self.pre_session_size
            final_end = end + self.after_session_interval

            if start < 0 or final_end >= df.shape[0] or end <= last_used:
                continue

            session_closes = df.iloc[start : end]

            filter = np.mean(self._calculate_volatility(session_closes))

            if filter < (session_vol / self.scaling_factor):
                chosen_sessions.append(df.iloc[start : final_end])
                last_used = final_end

        return chosen_sessions


    def transform(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        result = self._extract_intervals(df)
        return result


class Splitter:
    def __init__(self, ticker: str, save: bool = True, test_size: float = 0.2, random_state: int = 1) -> None:
        self.ticker = ticker
        self.save = save
        self.test_size = test_size
        random.seed(random_state)


    def _split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df_ = df.reset_index(drop=True)
        df_['date'] = pd.to_datetime(df.index)

        result = {}
        for day, observations in df_.groupby(df_['date'].dt.date):
            observations = observations.set_index('date')
            result[str(day)] = observations

            if self.save:
                observations.to_csv(f'./data/tmp/{self.ticker}_{day}.csv')

        return result


    def _train_test_split(self, splitted: dict) -> tuple[dict, dict]:
        observations = list(splitted.keys())
        random.shuffle(observations)

        test_size = int(len(observations) * self.test_size)

        test_keys = observations[:test_size]
        train_keys = observations[test_size:]

        train = {k: splitted[k] for k in train_keys}
        test = {k: splitted[k] for k in test_keys}

        return train, test


    def transform(self, df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        result = self._split(df)
        train, test = self._train_test_split(result)
        return train, test


class Pipeline:
    def __init__(self, ticker: str, file_path: str | None = None) -> None:
        self.ticker = ticker
        self.file_path = file_path

        self.collector = Collector(ticker=self.ticker, period='10d')
        self.preprocessor = Preprocessor()
        self.splitter = Splitter(ticker=self.ticker)
        self.scaler = Scaler(ticker=self.ticker)
        self.extractor = Extractor()


    def _fit_scaler(self, df: dict[str, pd.DataFrame]) -> None:
        stacked = pd.DataFrame()

        for date in df.keys():
            stacked = pd.concat([stacked, df[date]], axis=0)

        self.scaler.transform(stacked)


    def process_data(self) -> None:
        df = self.collector.collect(self.file_path)

        df.to_csv('./data/after_collection.csv')

        df = self.preprocessor.transform(df)

        df.to_csv('./data/after_preprocessor.csv')

        train, test = self.splitter.transform(df)

        self._fit_scaler(train)

        for i in range(2):
            src = train if i == 1 else test
            name = 'train' if i == 1 else 'test'

            for day in src.keys():
                obs = src[day]
                obs_std = self.scaler.transform(obs)
                high_volatile_intervals = self.extractor.transform(obs_std)

                for idx, interval in enumerate(high_volatile_intervals):
                    interval.to_csv(f'./data/{name}/{self.ticker}_{day}_{idx}.csv')