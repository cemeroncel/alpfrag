"""Helper functions to save data."""

import datetime


def get_timestamp() -> str:
    """Returns the current timestamp in the YYYYMMDDTHHMMSS format.

    Returns
    -------
    str
        Current timestamp in the YYYYMMDDHHMMSS format.

    """

    return datetime.datetime.today().strftime('%Y%m%d%H%M%S')
