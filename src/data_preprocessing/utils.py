import glob
import os
from datetime import date, datetime, timedelta
from typing import Tuple


# Valid 1h, 1d, 1mo, 1y
def get_past_datetime(period: str = '1mo') -> Tuple[date, date]:

    deltas = {
        '1h': timedelta(hours=1),
        '1d': timedelta(days=1),
        '1mo': timedelta(days=31),
        '1y': timedelta(days=365),
        '2y': timedelta(days=2 * 365),
        '3y': timedelta(days=3 * 365),
        '4y': timedelta(days=4 * 365),
        '5y': timedelta(days=5 * 365),
    }

    end = date.today()
    start = end - deltas[period]

    return start, end


# Adjust to minimal period of data collection
def validate_dates(start: date | str, end: date | str, min_difference: timedelta) -> Tuple[date, date]:
    if isinstance(start, str):
        start = datetime.strptime(start, '%Y-%m-%d').date()
    if isinstance(end, str):
        end = datetime.strptime(end, '%Y-%m-%d').date()

    difference = end - start
    if difference < min_difference:
        start = end - min_difference

    return start, end


# Range Dates
def daterange(start: date, end: date):
    days = int((end - start).days)

    for n in range(days - 1):
        yield start + timedelta(days=n), start + timedelta(days=n+1)


def discard_files(files: list[str]) -> None:
    for file in files:
        try:
            os.remove(file)
        except:
            pass


def get_files(dir: str, pattern: str = '*.csv') -> list[str]:
    files = glob.glob(os.path.join(dir, pattern))
    return files

def file_exists(path: str, file: str) -> bool:
    file = os.path.join(path, file)
    return os.path.exists(file)

def join_paths(path: str, file: str) -> str:
    return os.path.join(path, file)

def extract_ticker_name(name: str) -> str:
    file_name = name.split('/')[-1]
    return file_name.split('_')[0].lower().strip()