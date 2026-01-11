FROM apache/airflow:2.9.3-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-airflow.txt /requirements-airflow.txt

RUN chown airflow: /requirements-airflow.txt

USER airflow

RUN pip install --no-cache-dir -r /requirements-airflow.txt