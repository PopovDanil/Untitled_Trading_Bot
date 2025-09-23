import yfinance as yf
import pandas as pd
from datetime import date
from utils import get_past_datetime
import pandas_datareader.data as reader


# TODO:
# Check the periods, not all data available per day -> broadcast it
# Different shapes -> broadcast

# Default stock name - ^GSPC -- S&P 500
def collect_data(stock_name: str = '^GSPC', period: str = '1mo', interval: str = '1d', start: date | None = None, end: date | None = None) -> pd.DataFrame:

    # Download latest OHLCV (or desired interval chosen by `start` and `end`)
    # OHLCV - Open, High, Low, Close, Volume
    if start is not None and end is not None:
        data = yf.download(tickers=stock_name, start=start, end=end)
    else:
        start, end = get_past_datetime(period=period)
        data = yf.download(tickers=stock_name, period=period, interval=interval)

    if data is None:
        raise RuntimeError('No such stock data')

    print(data)

    # Download Macroeconomics Indicators
    macro = {
        'CPI': reader.DataReader(name='CPIAUCSL', data_source='fred', start=start, end=end),          # Inflation
        'Unemployment': reader.DataReader(name='UNRATE', data_source='fred', start=start, end=end),   # Unemployment Rate
        'FedFundsRate': reader.DataReader(name='FEDFUNDS', data_source='fred', start=start, end=end), # Federal funds rate
        'GDP': reader.DataReader(name='GDP', data_source='fred', start=start, end=end),               # GDP
        '10Y_Treasury': reader.DataReader(name='DGS10', data_source='fred', start=start, end=end),    # 10-year bond yield
        'VIX': reader.DataReader(name='VIXCLS', data_source='fred', start=start, end=end),            # VIX index
    }

    for key in macro.keys():
        print(key, macro[key])

    macro = pd.DataFrame(data=macro)
    print(macro.head(10))

    return macro


collect_data(period='1y')
# data = yf.Ticker('^GSPC').funds_data

# data.history(period='5d', interval='1m').to_csv('sp500.csv')
# print(data.description)
# text = ''
# for key in data.info.keys():
#     text += f'{key}: {data.info[key]}\n'

# with open('example.txt', 'w') as f:
#     f.write(text)

