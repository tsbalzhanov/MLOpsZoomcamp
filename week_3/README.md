Initialize venv with `uv sync`

Run local mlflow server with `uv run mlflow server --backend-store-uri "sqlite:///data/mlflow/store.db" --default-artifact-root data/mlflow/artifacts`

Run local prefect server with `uv run prefect server start`

Serve prefect flow with `uv run main.py`

Launch new prefect flow run with `uv run prefect deployment run 'main-flow/main-flow' -p data_dir=data -p date="2023-03-01"`
