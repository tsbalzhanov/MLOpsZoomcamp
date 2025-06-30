#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import sys

import pandas as pd


def get_storage_options() -> dict:
    storage_options = {'client_kwargs': {}}
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url is not None:
        storage_options['client_kwargs']['endpoint_url'] = s3_endpoint_url
    return storage_options


def get_input_path(year: int, month: int) -> str:
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year: int, month: int) -> str:
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(input_path: str) -> pd.DataFrame:
    return pd.read_parquet(input_path, storage_options=get_storage_options())


def save_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_parquet(
        output_path, engine='pyarrow', compression=None, index=False, storage_options=get_storage_options()
    )


def prepare_data(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def main(year: int, month: int) -> None:
    input_path = get_input_path(year, month)
    output_path = get_output_path(year, month)
    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_path)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_path)


if __name__ == '__main__':
    year, month = int(sys.argv[1]), int(sys.argv[2])
    main(year, month)
