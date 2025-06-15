import argparse
import pickle

import pandas as pd


def read_data(categorical: list[str], filename: str) -> pd.DataFrame:
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--month', required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    year, month = args.year, args.month
    print(f'{year = }, {month = }')

    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04}-{month:02}.parquet'
    df = read_data(categorical, url)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Prediction mean: {y_pred.mean().item():2f}')
    print(f'Prediction std: {y_pred.std().item():2f}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = df[['ride_id']].copy()
    df_result['predicted_duration'] = y_pred

    output_file = f'yellow_tripdata_{year:04}-{month:02}.parquet'

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__  == '__main__':
    main()
