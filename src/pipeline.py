import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import ks_2samp

import mlflow
from mlflow.tracking import MlflowClient


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
            # иначе PyCaret создаёт свои run'ы и всё едет
            log_experiment=False,
        )

        best = compare_models(sort="F1")
        final = finalize_model(best)

        leaderboard = pull()
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            leaderboard.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, artifact_path="reports")

        # модель -> артефакт model
        mlflow.sklearn.log_model(
            sk_model=final,
            artifact_path="model",
            input_example=input_example,
        )

        # Контрольная проверка: "model" реально появился
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
