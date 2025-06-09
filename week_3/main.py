import datetime
import pathlib

import mlflow
import mlflow.entities
import pandas as pd
import prefect
import requests
import sklearn.feature_extraction
import sklearn.linear_model
import yaml


@prefect.task
def download_rides(result_dir: pathlib.Path, date: datetime.date) -> pathlib.Path:
    if date.day != 1:
        raise ValueError('Date\'s day should be 1st')
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{date.strftime("%Y-%m")}.parquet'
    file_name = url.rsplit('/', 1)[-1]
    assert file_name is not None
    response = requests.get(url)
    response.raise_for_status()
    result_path = result_dir / file_name
    result_path.write_bytes(response.content)
    return result_path


@prefect.task
def count_parquet_num_entries(initial_rides_path: pathlib.Path) -> int:
    return len(pd.read_parquet(initial_rides_path))


@prefect.task
def prepare_data(result_dir: pathlib.Path, initial_rides_path: pathlib.Path) -> pathlib.Path:
    result_path = result_dir / initial_rides_path.name
    trips = pd.read_parquet(initial_rides_path)
    trips['duration'] = trips.tpep_dropoff_datetime - trips.tpep_pickup_datetime
    trips.duration = trips.duration.dt.total_seconds() / 60
    trips = trips[(trips.duration >= 1) & (trips.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    trips[categorical] = trips[categorical].astype(str)
    trips.to_parquet(result_path)
    return result_path


@prefect.task
def train_model(
    prepared_rides_path: pathlib.Path
) -> tuple[sklearn.feature_extraction.DictVectorizer, sklearn.linear_model.LinearRegression, mlflow.entities.Run]:
    feature_cols = ['PULocationID', 'DOLocationID']
    target_col = 'duration'
    selected_cols = feature_cols + [target_col]
    trips = pd.read_parquet(prepared_rides_path)[selected_cols]
    features_dict = trips[feature_cols].to_dict('records')
    mlflow.autolog()
    with mlflow.start_run() as ml_run:
        vectorizer = sklearn.feature_extraction.DictVectorizer()
        linear_model = sklearn.linear_model.LinearRegression()
        linear_model.fit(vectorizer.fit_transform(features_dict), trips[target_col])
    return vectorizer, linear_model, ml_run


@prefect.task
def print_model_size(ml_run: mlflow.entities.Run) -> None:
    model_info_uri = f'{ml_run.info.artifact_uri}/model/MLmodel'
    model_info_text = mlflow.artifacts.load_text(model_info_uri)
    model_info = yaml.safe_load(model_info_text)
    print(f'Model size in bytes: {model_info["model_size_bytes"]}')


@prefect.flow
def main_flow(data_dir: pathlib.Path, date: datetime.date) -> None:
    print(f'Prefect version: {prefect.__version__}')

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('yellow-taxi-linear')

    initial_data_dir = data_dir / 'initial_data'
    initial_data_dir.mkdir(exist_ok=True)
    initial_rides_path = download_rides(initial_data_dir, date)
    print(f'Number of records in initial data: {count_parquet_num_entries(initial_rides_path)}')

    preraped_data_dir = data_dir / 'prepared_data'
    preraped_data_dir.mkdir(exist_ok=True)
    prepared_rides_path = prepare_data(preraped_data_dir, initial_rides_path)
    print(f'Number of records in prepared data: {count_parquet_num_entries(prepared_rides_path)}')

    vectorizer, linear_model, ml_run = train_model(prepared_rides_path)
    print(f'Linear model intercept: {linear_model.intercept_.item():.2f}')

    print_model_size(ml_run)


def main() -> None:
    main_flow.serve()


if __name__ == '__main__':
    main()
