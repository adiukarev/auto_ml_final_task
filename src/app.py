import os
import time
import json
import csv
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
REG_MODEL_NAME = os.environ["REG_MODEL_NAME"]
DEFAULT_SPLIT_B = float(os.environ.get("AB_SPLIT_B"))
CONFIG_PATH = os.environ.get("ROUTER_CONFIG")
LOG_PATH = os.environ.get("REQUEST_LOG")

_model_cache: Dict[str, Any] = {}
_model_meta_cache: Dict[str, Dict[str, Any]] = {}


def _safe_makedirs(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_router_config() -> Dict[str, Any]:
    cfg = {"ab_split_b": DEFAULT_SPLIT_B}
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                if isinstance(data, dict) and "ab_split_b" in data:
                    cfg["ab_split_b"] = float(data["ab_split_b"])
    except Exception:
        pass
    return cfg


_router_cfg = _load_router_config()


def _ensure_log() -> None:
    _safe_makedirs(LOG_PATH)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "ts_utc",
                    "request_id",
                    "user_id",
                    "variant",
                    "alias",
                    "model",
                    "model_version",
                    "features_json",
                    "label",
                    "prediction",
                    "latency_ms",
                ]
            )


def _get_model_version_by_alias(alias: str) -> Optional[str]:
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(REG_MODEL_NAME, alias)
        return str(mv.version)
    except Exception:
        return None


def _load_model_by_alias(alias: str) -> Tuple[Any, Optional[str]]:
    key = f"{REG_MODEL_NAME}@{alias}"
    if key in _model_cache:
        meta = _model_meta_cache.get(key, {})
        return _model_cache[key], meta.get("version")

    uri = f"models:/{REG_MODEL_NAME}@{alias}"
    print("Loading:", uri)
    model = mlflow.pyfunc.load_model(uri)
    _model_cache[key] = model
    _model_meta_cache[key] = {"version": _get_model_version_by_alias(alias)}
    return model, _model_meta_cache[key].get("version")


def _select_variant(user_id: int, split_b: float) -> str:
    bucket = user_id % 100
    return "B" if bucket < int(split_b * 100) else "A"


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/config")
def get_config():
    return jsonify(
        {
            "ab_split_b": float(_router_cfg.get("ab_split_b", DEFAULT_SPLIT_B)),
            "config_path": CONFIG_PATH,
            "log_path": LOG_PATH,
            "model": REG_MODEL_NAME,
            "aliases": {"A": "prod", "B": "staging"},
        }
    )


@app.post("/config")
def set_config():
    payload = request.get_json(force=True) or {}
    if "ab_split_b" not in payload:
        return jsonify({"error": "bad_request", "detail": "missing field ab_split_b"}), 400

    try:
        v = float(payload["ab_split_b"])
    except Exception:
        return jsonify({"error": "bad_request", "detail": "ab_split_b must be a number"}), 400

    if not (0.0 <= v <= 1.0):
        return jsonify({"error": "bad_request", "detail": "ab_split_b must be in [0, 1]"}), 400

    _router_cfg["ab_split_b"] = v
    _safe_makedirs(CONFIG_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(_router_cfg, f, ensure_ascii=False, indent=2)

    return jsonify({"status": "ok", "ab_split_b": v})


@app.post("/predict")
def predict():
    t0 = time.time()
    _ensure_log()

    payload = request.get_json(force=True) or {}
    request_id = str(payload.get("request_id") or f"req-{int(time.time() * 1000)}")
    user_id = int(payload.get("user_id", 0))
    features = payload.get("features") or {}
    label = payload.get("label")

    if not isinstance(features, dict):
        return jsonify({"error": "bad_request", "detail": "`features` must be an object"}), 400

    split_b = float(_router_cfg.get("ab_split_b", DEFAULT_SPLIT_B))
    variant = _select_variant(user_id=user_id, split_b=split_b)

    alias = "staging" if variant == "B" else "prod"

    try:
        model, model_version = _load_model_by_alias(alias)
        pred = model.predict(pd.DataFrame([features]))
        prediction = pred[0].item() if hasattr(pred[0], "item") else pred[0]
    except MlflowException as e:
        return (
            jsonify({"error": "model_not_available", "detail": str(e), "model": REG_MODEL_NAME, "alias": alias}),
            503,
        )
    except Exception as e:
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

    latency_ms = int((time.time() - t0) * 1000)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                datetime.utcnow().isoformat(),
                request_id,
                user_id,
                variant,
                alias,
                REG_MODEL_NAME,
                model_version,
                json.dumps(features, ensure_ascii=False),
                label,
                prediction,
                latency_ms,
            ]
        )

    return jsonify(
        {
            "request_id": request_id,
            "variant": variant,
            "alias": alias,
            "model": REG_MODEL_NAME,
            "model_version": model_version,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "ab_split_b": split_b,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
