import datetime

import pandas as pd

from batch import get_input_path, get_storage_options


def dt(hour: int, minute: int, second: int = 0) -> datetime.datetime:
    return datetime.datetime(2023, 1, 1, hour, minute, second)


def prepare_data():
    categorical_columns = ['PULocationID', 'DOLocationID']
    all_columns = categorical_columns + ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
    initial_data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1))
    ]
    df = pd.DataFrame(initial_data, columns=all_columns)
    input_path = get_input_path(2023, 1)
    df.to_parquet(
        input_path, engine='pyarrow', compression=None, index=False, storage_options=get_storage_options()
    )


if __name__ == '__main__':
    prepare_data()
