import datetime

import pandas as pd

from batch import prepare_data


def dt(hour: int, minute: int, second: int = 0) -> datetime.datetime:
    return datetime.datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    categorical_columns = ['PULocationID', 'DOLocationID']
    all_columns = categorical_columns + ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
    initial_data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1))
    ]
    actual_df = prepare_data(
        pd.DataFrame(initial_data, columns=all_columns), categorical_columns
    )
    expected_df = pd.DataFrame([
        ('-1', '-1', dt(1, 1), dt(1, 10), 9),
        ('1', '1', dt(1, 2), dt(1, 10), 8)
    ], columns=all_columns + ['duration'])
    assert (actual_df == expected_df).all().all()
