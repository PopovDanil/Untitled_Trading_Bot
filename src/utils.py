from datetime import timedelta, date
from typing import Tuple


# Valid 1h, 1d, 1mo, 1y
def get_past_datetime(period: str = '1m') -> Tuple[date, date]:

    deltas = {
        '1h': timedelta(hours=1),
        '1d': timedelta(days=1),
        '1mo': timedelta(days=31),
        '1y': timedelta(days=365),
    }

    end = date.today()
    start = end - deltas[period]

    return start, end
