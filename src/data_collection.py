import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from utils import get_past_datetime, validate_dates
from datetime import datetime
import pandas_datareader.data as reader
import pandas_ta as ta


# TODO: preprocessing stage - handling missing values, normalization

# Adjust shapes of main df and additional column
def broadcast_data(dst: pd.DataFrame, src: pd.DataFrame, column_name: str) -> None:
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
def add_micro_indexes(data: pd.DataFrame, sma_length: int = 20, rsi_length: int = 14, macd_fast: int = 12, macd_low: int = 26, macd_signal: int = 0) -> None:
    # SMA - Simple Moving Average - average over the window of desired length
    data['SMA' + str(sma_length)] = ta.sma(data['Close'], length=sma_length)

    # RSI - Relative Strength Index - defines trends' power and probability of changes
    data['RSI' + str(rsi_length)] = ta.rsi(data['Close'], length=rsi_length)

    # MACD
    raise NotImplemented


# Default stock name - ^GSPC -- S&P 500
def collect_data(stock_name: str = '^GSPC', period: str = '1mo', interval: str = '1d', start: date | None = None, end: date | None = None) -> pd.DataFrame:

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
        broadcast_data(data, macro[indicator][indicator], macro_indicators[indicator])


    return data


data = collect_data()
data.to_csv('data/debug.csv')
