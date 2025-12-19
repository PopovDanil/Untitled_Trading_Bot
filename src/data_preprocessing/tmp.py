
import pandas as pd
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

            df['date'] = [date.to_pydatetime().date() for date in df.index]
            df = df.reset_index(drop=True)

            data = pd.concat([data, df], axis=0)

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

        new_df.index = new_df['date']
        return new_df


d = Collector(ticker='^GSPC', period='1d')
df = d._load_from_file('C:\\Users\\popov\\Documents\\Projects\\Project\\Untitled_Trading_Bot\\data\\train\\ADANIENSOL_5minute.csv')
print(df.head(5))
df = d._format_df(df)
df.to_csv('./data/formatted.csv')