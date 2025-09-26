from datetime import timedelta, date, datetime
from typing import Tuple


# Valid 1h, 1d, 1mo, 1y
def get_past_datetime(period: str = '1mo') -> Tuple[date, date]:

    deltas = {
        '1h': timedelta(hours=1),
        '1d': timedelta(days=1),
        '1mo': timedelta(days=31),
        '1y': timedelta(days=365),
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
