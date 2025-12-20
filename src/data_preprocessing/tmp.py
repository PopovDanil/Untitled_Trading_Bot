import os

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from utils import split_period_into_days
from yfinance import download


# to one format
class Collector:
    def __init__(
        self,
        ticker: str,
        period: str = '1d',
        ) -> None:

        self.ticker = ticker
        self.period = period


    def _download_main_data(self) -> pd.DataFrame:
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


    def _load_from_file(self, file: str) -> pd.DataFrame:
        return pd.read_csv(file)


    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats columns and index: brings columns to lowercase and formats index to YYYY-MM-DD HH:MM.

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        new_df = pd.DataFrame()

        for column in df.columns:
            formatted_name = column.lower()

            if formatted_name not in ['date', 'open', 'high', 'low', 'close', 'volume']:
                continue

            new_df[formatted_name] = df[column]

        return new_df


class Preprocessor:
    def __init__(
            self,
            sma_length: int = 20,
            rsi_length: int = 14,
            macd_fast: int = 12,
            macd_slow: int = 26,
            macd_signal: int = 9
        ) -> None:
        self.sma_length = sma_length    # Simple mean average window size
        self.rsi_length = rsi_length    # RSI window size
        self.macd_fast = macd_fast      # MACD
        self.macd_slow = macd_slow      # MACD
        self.macd_signal = macd_signal  # MACD


    def _fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        filled = df.copy()

        for column in df.columns:
            if column == 'date':
                continue

            mean = df[column].mean()
            filled[column] = filled[column].fillna(mean)

        return filled


    def _add_micro_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        # SMA - Simple Moving Average - average over the window of desired length
        data['sma' + str(self.sma_length)] = ta.sma(data['close'], length=self.sma_length)

        # RSI - Relative Strength Index - defines trends' power and probability of changes
        data['rsi' + str(self.rsi_length)] = ta.rsi(data['close'], length=self.rsi_length)

        # MACD - Moving Average Convergence/Divergence - shows price oscitation
        macd = ta.macd(data['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        for col in macd.columns:
            data[col.lower()] = macd[col]

        return data


    def _apply_log_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data['log_close'] = np.log(data['close'])
        data['log_high'] = np.log(data['high'])
        data['log_low'] = np.log(data['low'])
        data['log_volume'] = np.log(data['volume'] + 1) # to avoid -inf

        data['log_return'] = np.log(df['close'] / df['close'].shift(1))

        return data


    def _drop_incomplete(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()


class Scaler:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.scaler_path = os.path.join('data', 'scalers', ticker + '.joblib')

        if not os.path.exists(self.scaler_path):
            self.use_existing = False
            self.scaler = StandardScaler()
        else:
            self.use_existing = True
            self.scaler = self._load_scaler()

    def _load_scaler(self) -> StandardScaler:
        scaler = None
        try:
            scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Error while loading the scaler: {e}")

        return scaler


    def _fit_scaler(self, df: pd.DataFrame) -> None:
        self.scaler.fit(df)
        joblib.dump(self.scaler, self.scaler_path)


    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df_without_dates = df.reset_index(drop=True)

        if not self.use_existing:
            self._fit_scaler(df_without_dates)

        df_std = self.scaler.transform(df_without_dates)
        df_std = pd.DataFrame(data=df_std, columns=df_without_dates.columns)

        df_std.index = df.index
        df_std['close_raw'] = df['close']

        return df_std


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


d = Collector(ticker='^GSPC', period='4d')
df = d._download_main_data()
df = d._format_df(df)
df.to_csv('./data/formatted.csv')

p = Preprocessor()
df = p._fill_na(df)
df = p._add_micro_indexes(df)
df = p._apply_log_scaling(df)
df = p._drop_incomplete(df)

df.to_csv('./data/m.csv')

s = Scaler(ticker='^GSPC')
df = s._normalize(df)

e = Extractor()
dfs = e._extract_intervals(df)

for i, df in enumerate(dfs):
    df.to_csv(f'./data/ext/ext{i}.csv')