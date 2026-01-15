import os
import tempfile
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import ks_2samp

import mlflow
from mlflow.tracking import MlflowClient

from mlflow.models.signature import infer_signature

def _safe_log_dict_as_json(d: dict, artifact_path: str, filename: str = "report.json") -> None:
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, filename)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(p, artifact_path=artifact_path)
    except Exception:
        pass


def _log_basic_dataset_profile(df: pd.DataFrame, target: str) -> None:
    try:
        mlflow.log_param("n_rows", int(df.shape[0]))
        mlflow.log_param("n_cols", int(df.shape[1]))
        mlflow.log_param("target", target)

        vc = df[target].value_counts(dropna=False)
        for i, (k, v) in enumerate(vc.items()):
            if i >= 20:
                break
            mlflow.log_param(f"target_value_{i}", str(k))
            mlflow.log_param(f"target_count_{i}", int(v))

        na = df.isna().sum()
        mlflow.log_param("n_missing_total", int(na.sum()))
        mlflow.log_param("n_missing_cols", int((na > 0).sum()))
    except Exception:
        pass


def _try_log_pycaret_plots(final_model) -> None:
    try:
        from pycaret.classification import plot_model
    except Exception:
        return

    plots = [
        ("confusion_matrix", "confusion_matrix"),
        ("auc", "roc_auc"),
        ("pr", "precision_recall"),
        ("feature", "feature_importance"),
    ]

    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        try:
            os.chdir(td)
            for plot, name in plots:
                try:
                    plot_model(final_model, plot=plot, save=True)
                    for fn in os.listdir(td):
                        if fn.lower().endswith(".png"):
                            mlflow.log_artifact(os.path.join(td, fn), artifact_path=f"plots/{name}")
                            try:
                                os.remove(os.path.join(td, fn))
                            except Exception:
                                pass
                except Exception:
                    continue
        finally:
            os.chdir(cwd)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_drift_report(report: pd.DataFrame, out_path: str) -> str:
    _ensure_dir(out_path)
    report.to_csv(out_path, index=False)
    
    return out_path


def load_iris_splits(seed_train: int = 42, seed_current: int = 7, drift_strength: float = 0.25):
    df = load_iris(as_frame=True).frame.copy()
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})

    train_df = df.sample(frac=0.7, random_state=seed_train).reset_index(drop=True)
    current_df = df.sample(frac=0.7, random_state=seed_current).reset_index(drop=True)

    current_df["sepal length (cm)"] = current_df["sepal length (cm)"] + drift_strength

    cols = [c for c in train_df.columns if c != "target"] + ["target"]

    return train_df[cols], current_df[cols]


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty:
        return 0.0

    edges = expected.quantile(np.linspace(0, 1, bins + 1)).values
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def ks_pvalue(expected: pd.Series, actual: pd.Series) -> float:
    expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty:
        return 1.0
    
    return float(ks_2samp(expected, actual).pvalue)


def drift_report(train_df: pd.DataFrame, current_df: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    feats = [c for c in train_df.columns if c != target]
    rows = []
    for f in feats:
        rows.append({
            "feature": f,
            "psi": psi(train_df[f], current_df[f]),
            "ks_pvalue": ks_pvalue(train_df[f], current_df[f]),
            "train_mean": float(np.nanmean(train_df[f])),
            "current_mean": float(np.nanmean(current_df[f])),
            "mean_shift": float(np.nanmean(current_df[f]) - np.nanmean(train_df[f])),
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def is_drifted(report: pd.DataFrame, psi_threshold: float = 0.2) -> bool:
    return bool((report["psi"] > psi_threshold).any())


def train_pycaret_and_log(train_df: pd.DataFrame, target: str, experiment_name: str) -> dict:
    from pycaret.classification import setup, compare_models, finalize_model, pull

    mlflow.set_experiment(experiment_name)

    X = train_df.drop(columns=[target])
    input_example = X.head(3)

    with mlflow.start_run(run_name="pycaret_automl") as run:
        setup(
            data=train_df,
            target=target,
            session_id=42,
            fold=5,
            verbose=False,
            html=False,
            log_experiment=False,
        )

        _log_basic_dataset_profile(train_df, target)
        mlflow.log_param("pycaret_session_id", 42)
        mlflow.log_param("pycaret_fold", 5)
        mlflow.log_param("pycaret_sort", "F1")

        best = compare_models(sort="F1")
        final = finalize_model(best)


        try:
            from pycaret.classification import predict_model
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            holdout = predict_model(final)
            pred_col = "Label" if "Label" in holdout.columns else None
            if pred_col is not None and target in holdout.columns:
                y_true = holdout[target].astype(int).to_numpy()
                y_pred = holdout[pred_col].astype(int).to_numpy()

                mlflow.log_metric("holdout_accuracy", float(accuracy_score(y_true, y_pred)))
                mlflow.log_metric("holdout_precision_macro", float(precision_score(y_true, y_pred, average="macro", zero_division=0)))
                mlflow.log_metric("holdout_recall_macro", float(recall_score(y_true, y_pred, average="macro", zero_division=0)))
                mlflow.log_metric("holdout_f1_macro", float(f1_score(y_true, y_pred, average="macro", zero_division=0)))

            _try_log_pycaret_plots(final)
        except Exception:
            pass


        leaderboard = pull()

        if leaderboard is not None and len(leaderboard) > 0:
            top = leaderboard.iloc[0]

            metric_map = {
                "Accuracy": "accuracy",
                "AUC": "roc_auc",
                "Recall": "recall",
                "Precision": "precision",
                "F1": "f1",
                "Kappa": "kappa",
                "MCC": "mcc",
            }

            for col, mlflow_name in metric_map.items():
                if col in leaderboard.columns:
                    val = top[col]
                    try:
                        val = float(val)
                        if np.isfinite(val):
                            mlflow.log_metric(mlflow_name, val)
                    except Exception:
                        pass

            if "Model" in leaderboard.columns:
                mlflow.log_param("best_model", str(top["Model"]))
        else:
            mlflow.log_param("leaderboard_empty", True)

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            leaderboard.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, artifact_path="reports")

        signature = None
        try:
            y_sig = final.predict(input_example)
            signature = infer_signature(input_example, y_sig)
        except Exception:
            pass

        mlflow.sklearn.log_model(
            sk_model=final,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        client = MlflowClient()
        artifact_paths = [a.path for a in client.list_artifacts(run.info.run_id)]
        if "model" not in artifact_paths:
            raise RuntimeError(
                f"MLflow run {run.info.run_id} has no 'model' artifact. Got: {artifact_paths}"
            )

        return {"run_id": run.info.run_id, "model_uri": f"runs:/{run.info.run_id}/model"}


def register_model(model_uri: str, model_name: str, alias: str = "staging") -> dict:
    client = MlflowClient()

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(model_name, alias, str(mv.version))

    return {"name": model_name, "version": mv.version, "alias": alias}


def evaluate_ab_and_promote(
    log_path: str,
    model_name: str,
    prod_alias: str = "prod",
    staging_alias: str = "staging",
    min_labeled_rows: int = 30,
    min_delta_acc: float = 0.0,
    alpha: float = 0.05,
    report_path: str | None = None,
) -> dict:
    from scipy.stats import chi2_contingency
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    started_run = False
    if mlflow.active_run() is None:
        try:
            mlflow.start_run(run_name="ab_evaluation")
            started_run = True
        except Exception:
            started_run = False

    try:
        mlflow.log_param("ab_log_path", log_path)
        mlflow.log_param("ab_model_name", model_name)
        mlflow.log_param("ab_prod_alias", prod_alias)
        mlflow.log_param("ab_staging_alias", staging_alias)
        mlflow.log_param("ab_min_labeled_rows", int(min_labeled_rows))
        mlflow.log_param("ab_min_delta_acc", float(min_delta_acc))
        mlflow.log_param("ab_alpha", float(alpha))
    except Exception:
        pass

    if not os.path.exists(log_path):
        return {
            "status": "no_data",
            "detail": f"Log file not found: {log_path}",
            "promoted": False,
        }

    df = pd.read_csv(log_path)
    if df.empty:
        return {"status": "no_data", "detail": "Empty log", "promoted": False}

    if "label" not in df.columns:
        return {"status": "no_labels", "detail": "Column 'label' missing", "promoted": False}

    df = df.dropna(subset=["label", "prediction", "alias"]).copy()
    if df.empty:
        return {"status": "no_labels", "detail": "No labeled rows", "promoted": False}

    def _to_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    df["y_true"] = df["label"].map(_to_int)
    df["y_pred"] = df["prediction"].map(_to_int)
    df = df.dropna(subset=["y_true", "y_pred"]).copy()
    if df.empty:
        return {"status": "no_labels", "detail": "Labels are not numeric", "promoted": False}

    df_prod = df[df["alias"] == prod_alias]
    df_stg = df[df["alias"] == staging_alias]

    if len(df_prod) < min_labeled_rows or len(df_stg) < min_labeled_rows:
        return {
            "status": "not_enough_data",
            "detail": f"Need at least {min_labeled_rows} labeled rows per variant",
            "counts": {"prod": int(len(df_prod)), "staging": int(len(df_stg))},
            "promoted": False,
        }

    def _metrics(d: pd.DataFrame) -> dict:
        y_t = d["y_true"].astype(int).to_numpy()
        y_p = d["y_pred"].astype(int).to_numpy()

        return {
            "n": int(len(d)),
            "accuracy": float(accuracy_score(y_t, y_p)),
            "precision_macro": float(precision_score(y_t, y_p, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_t, y_p, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        }

    m_prod = _metrics(df_prod)
    m_stg = _metrics(df_stg)

    prod_correct = int((df_prod["y_true"] == df_prod["y_pred"]).sum())
    prod_wrong = int(len(df_prod) - prod_correct)
    stg_correct = int((df_stg["y_true"] == df_stg["y_pred"]).sum())
    stg_wrong = int(len(df_stg) - stg_correct)

    chi2, p_value, _, _ = chi2_contingency([[prod_correct, prod_wrong], [stg_correct, stg_wrong]])

    promoted = False
    decision = "keep_prod"

    if (m_stg["accuracy"] >= m_prod["accuracy"] + min_delta_acc) and (p_value < alpha):
        client = MlflowClient()
        stg_mv = client.get_model_version_by_alias(model_name, staging_alias)
        client.set_registered_model_alias(model_name, prod_alias, str(stg_mv.version))
        promoted = True
        decision = "promote_staging_to_prod"

    report = {
        "status": "ok",
        "decision": decision,
        "promoted": promoted,
        "p_value": float(p_value),
        "chi2": float(chi2),
        "metrics": {"prod": m_prod, "staging": m_stg},
        "counts": {"prod": int(len(df_prod)), "staging": int(len(df_stg))},
    }

    if report_path:
        _ensure_dir(report_path)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- persist report to MLflow
    try:
        _safe_log_dict_as_json(report, artifact_path="ab_eval")
        if report_path and os.path.exists(report_path):
            mlflow.log_artifact(report_path, artifact_path="ab_eval")
        # metrics (flat)
        if "metrics_prod" in report:
            for k,v in report["metrics_prod"].items():
                if isinstance(v,(int,float)):
                    mlflow.log_metric(f"ab_prod_{k}", float(v))
        if "metrics_staging" in report:
            for k,v in report["metrics_staging"].items():
                if isinstance(v,(int,float)):
                    mlflow.log_metric(f"ab_staging_{k}", float(v))
        for k in ["delta_acc", "p_value"]:
            if k in report and isinstance(report[k],(int,float)):
                mlflow.log_metric(f"ab_{k}", float(report[k]))
        mlflow.log_param("ab_promoted", bool(report.get("promoted", False)))
    except Exception:
        pass

    if started_run:
        try:
            mlflow.end_run()
        except Exception:
            pass

    return report


def run_pipeline(
    psi_threshold: float = 0.2,
    drift_strength: float = 0.25,
    target: str = "target",
    experiment: str = "ab-automl",
    reg_model_name: str = "IrisClassifier",
) -> dict:
    train_df, current_df = load_iris_splits(drift_strength=drift_strength)
    rep = drift_report(train_df, current_df, target=target)
    triggered = is_drifted(rep, psi_threshold=psi_threshold)

    out = {
        "drift_triggered": triggered,
        "drift_top": rep.head(5).to_dict(orient="records"),
    }

    if not triggered:
        return out

    info = train_pycaret_and_log(train_df, target=target, experiment_name=experiment)
    reg = register_model(info["model_uri"], reg_model_name, alias="staging")
    out.update({"train": info, "register": reg})

    return out
