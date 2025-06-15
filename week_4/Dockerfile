FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

COPY main.py Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

ENTRYPOINT ["python", "main.py"]
