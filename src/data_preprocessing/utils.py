from datetime import datetime, timedelta


def split_period_into_days(period: str = '1d') -> list[tuple[datetime, datetime]]:
    """
    Accepts period in format "_d", where _ represents desired number of days. Returns
    the splitted period (from current day) + 2 days.

    Args:
        period (str, optional): desired period in days. Defaults to '1d'.

    Returns:
        list[tuple[datetime, datetime]]: splitted period of days in format YYYY-MM-DD
            each day is represented as pair (start, end)
    """
    end = datetime.now().date()

    required_days = int(period[:-1]) + 1
    start = end - timedelta(days=required_days)

    days = []

    current = start + timedelta(days=1)
    previous = start
    while current <= end:
        days.append((previous, current))
        previous = current
        current += timedelta(days=1)

    return days
