from __future__ import annotations

import os
import time
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

import mlflow
from mlflow.tracking import MlflowClient

sys.path.append("/opt/airflow/src")

from pipeline import (
    load_iris_splits,
    drift_report,
    is_drifted,
    train_pycaret_and_log,
)

TARGET = os.environ.get("TARGET")
PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD"))
DRIFT_STRENGTH = float(os.environ.get("DRIFT_STRENGTH"))
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
REG_MODEL_NAME = os.environ.get("REG_MODEL_NAME")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def check_drift(**context):
    train_df, current_df = load_iris_splits(drift_strength=DRIFT_STRENGTH)
    rep = drift_report(train_df, current_df, target=TARGET)
    triggered = is_drifted(rep, psi_threshold=PSI_THRESHOLD)

    ti = context["ti"]
    ti.xcom_push(key="drift_triggered", value=triggered)
    ti.xcom_push(key="drift_report_json", value=rep.to_json(orient="records"))

    return triggered


def branch(**context):
    triggered = context["ti"].xcom_pull(key="drift_triggered")

    return "retrain" if triggered else "no_retrain"


def retrain(**context):
    train_df, _ = load_iris_splits(drift_strength=DRIFT_STRENGTH)
    info = train_pycaret_and_log(train_df, target=TARGET, experiment_name=MLFLOW_EXPERIMENT)

    # cохраняем именно model_uri, который вернул pipeline.py
    ti = context["ti"]
    ti.xcom_push(key="model_uri", value=info["model_uri"])  # "runs:/<run_id>/model"
    ti.xcom_push(key="run_id", value=info["run_id"])

    return info


def _wait_for_artifact(run_id: str, artifact_path: str = "model", timeout_s: int = 30):
    client = MlflowClient()
    deadline = time.time() + timeout_s
    last_paths = None

    while time.time() < deadline:
        arts = client.list_artifacts(run_id)
        paths = [a.path for a in arts]
        last_paths = paths
        if artifact_path in paths:
            return True
        time.sleep(1)

    raise RuntimeError(
        f"Artifact '{artifact_path}' not found in run={run_id} after {timeout_s}s. "
        f"Top-level artifacts: {last_paths}"
    )


def register_in_mlflow_alias(**context):
    ti = context["ti"]
    model_uri = ti.xcom_pull(key="model_uri")
    run_id = ti.xcom_pull(key="run_id")

    if not model_uri or not run_id:
        raise RuntimeError(f"Missing XComs: model_uri={model_uri}, run_id={run_id}")

    _wait_for_artifact(run_id, artifact_path="model", timeout_s=30)

    client = MlflowClient()

    # регистрируем именно по model_uri (не пересобираем строку вручную)
    mv = mlflow.register_model(model_uri=model_uri, name=REG_MODEL_NAME)

    # alias
    client.set_registered_model_alias(REG_MODEL_NAME, "staging", str(mv.version))

    ti.xcom_push(key="registered_version", value=str(mv.version))
    ti.xcom_push(key="registered_alias", value="staging")

    print(f"[MLFLOW] Registered {REG_MODEL_NAME} v{mv.version} from {model_uri}")
    print(f"[MLFLOW] Alias 'staging' -> v{mv.version}")

    return {"model": REG_MODEL_NAME, "version": str(mv.version), "alias": "staging"}


with DAG(
    dag_id="drift_retrain_ab_pipeline",
    start_date=days_ago(1),
    schedule="@daily",
    catchup=False,
    tags=["ml", "drift", "ab"],
) as dag:
    start = EmptyOperator(task_id="start")
    t_check = PythonOperator(task_id="check_drift", python_callable=check_drift)
    t_branch = BranchPythonOperator(task_id="branch_on_drift", python_callable=branch)

    no_retrain = EmptyOperator(task_id="no_retrain")

    t_retrain = PythonOperator(task_id="retrain", python_callable=retrain)
    t_register_alias = PythonOperator(
        task_id="register_in_mlflow_alias",
        python_callable=register_in_mlflow_alias,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> t_check >> t_branch
    t_branch >> no_retrain >> end
    t_branch >> t_retrain >> t_register_alias >> end
