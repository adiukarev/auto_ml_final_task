from __future__ import annotations

import os
import time
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

import mlflow
from mlflow.tracking import MlflowClient

sys.path.append("/opt/airflow/src")

import pipeline

TARGET = os.environ.get("TARGET")
PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD"))
DRIFT_STRENGTH = float(os.environ.get("DRIFT_STRENGTH"))
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
REG_MODEL_NAME = os.environ.get("REG_MODEL_NAME")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
DRIFT_REPORT_OUT = os.environ.get("DRIFT_REPORT_OUT")
AB_REQUEST_LOG = os.environ.get("AB_REQUEST_LOG")
AB_EVAL_REPORT = os.environ.get("AB_EVAL_REPORT")


def _check_drift(**context):
    train_df, current_df = pipeline.load_iris_splits(drift_strength=DRIFT_STRENGTH)
    rep = pipeline.drift_report(train_df, current_df, target=TARGET)

    try:
        ds = context.get("ds") or "unknown_date"
        dated_path = DRIFT_REPORT_OUT.replace("drift_latest", f"drift_{ds}")
        pipeline.save_drift_report(rep, DRIFT_REPORT_OUT)
        pipeline.save_drift_report(rep, dated_path)
    except Exception as e:
        print(f"[WARN] Failed to save drift report: {e}")

    triggered = pipeline.is_drifted(rep, psi_threshold=PSI_THRESHOLD)

    ti = context["ti"]
    ti.xcom_push(key="drift_triggered", value=triggered)
    ti.xcom_push(key="drift_report_json", value=rep.to_json(orient="records"))

    return triggered


def _branch(**context):
    triggered = context["ti"].xcom_pull(key="drift_triggered")

    return "retrain" if triggered else "no_retrain"


def _retrain(**context):
    train_df, _ = pipeline.load_iris_splits(drift_strength=DRIFT_STRENGTH)
    info = pipeline.train_pycaret_and_log(train_df, target=TARGET, experiment_name=MLFLOW_EXPERIMENT)

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


def _register_in_mlflow_alias(**context):
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

    try:
        client.get_model_version_by_alias(REG_MODEL_NAME, "prod")
    except Exception:
        client.set_registered_model_alias(REG_MODEL_NAME, "prod", str(mv.version))
        print(f"[MLFLOW] Alias 'prod' was missing. Bootstrapped to v{mv.version}.")

    ti.xcom_push(key="registered_version", value=str(mv.version))
    ti.xcom_push(key="registered_alias", value="staging")

    print(f"[MLFLOW] Registered {REG_MODEL_NAME} v{mv.version} from {model_uri}")
    print(f"[MLFLOW] Alias 'staging' -> v{mv.version}")

    return {"model": REG_MODEL_NAME, "version": str(mv.version), "alias": "staging"}


def _ab_evaluate_and_promote(**context):
    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}

    def _get_int(key: str, default: int) -> int:
        v = conf.get(key)
        if v is None:
            v = os.environ.get(key)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _get_float(key: str, default: float) -> float:
        v = conf.get(key)
        if v is None:
            v = os.environ.get(key)
        try:
            return float(v)
        except Exception:
            return float(default)

    log_path = conf.get("AB_REQUEST_LOG") or AB_REQUEST_LOG
    report_path = conf.get("AB_EVAL_REPORT") or AB_EVAL_REPORT

    report = pipeline.evaluate_ab_and_promote(
        log_path=log_path,
        model_name=REG_MODEL_NAME,
        prod_alias=conf.get("AB_PROD_ALIAS", "prod"),
        staging_alias=conf.get("AB_STAGING_ALIAS", "staging"),
        min_labeled_rows=_get_int("AB_MIN_LABELED", 30),
        min_delta_acc=_get_float("AB_MIN_DELTA_ACC", 0.0),
        alpha=_get_float("AB_ALPHA", 0.05),
        report_path=report_path,
    )

    print(f"[AB] Evaluation report: {report}")
    return report



with DAG(
    dag_id="drift_retrain_ab_pipeline",
    start_date=days_ago(1),
    schedule="@daily",
    catchup=False,
    tags=["ml", "drift", "ab"],
) as dag:
    start = EmptyOperator(task_id="start")
    t_check = PythonOperator(task_id="check_drift", python_callable=_check_drift)
    t_branch = BranchPythonOperator(task_id="branch_on_drift", python_callable=_branch)

    no_retrain = EmptyOperator(task_id="no_retrain")

    t_retrain = PythonOperator(task_id="retrain", python_callable=_retrain)
    t_register_alias = PythonOperator(task_id="register_in_mlflow_alias", python_callable=_register_in_mlflow_alias)

    # Evaluate A/B and (if successful) promote staging ->> prod.
    t_ab_eval = PythonOperator(
        task_id="ab_evaluate_and_promote", 
        python_callable=_ab_evaluate_and_promote, 
        trigger_rule=TriggerRule.ALL_DONE
    )

    end = EmptyOperator(task_id="end", trigger_rule="all_done")

    start >> t_check >> t_branch
    t_branch >> no_retrain
    t_branch >> t_retrain >> t_register_alias

    [no_retrain, t_register_alias] >> t_ab_eval >> end
