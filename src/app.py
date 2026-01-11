import os
import time
import json
import csv
from datetime import datetime
from typing import Dict, Any

import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from mlflow.exceptions import MlflowException

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]   
REG_MODEL_NAME = os.environ["REG_MODEL_NAME"]      
AB_SPLIT_B = float(os.environ.get("AB_SPLIT_B"))
LOG_PATH = os.environ.get("REQUEST_LOG")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model_cache: Dict[str, Any] = {}

def load_model_by_alias(alias: str):
    key = f"{REG_MODEL_NAME}@{alias}"
    if key in _model_cache:
        return _model_cache[key]

    uri = f"models:/{REG_MODEL_NAME}@{alias}"
    print("Loading:", uri)
    model = mlflow.pyfunc.load_model(uri)
    _model_cache[key] = model
    return model


def ensure_log() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_utc", "user_id", "variant", "alias", "model", "features_json", "prediction"])


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    t0 = time.time()
    ensure_log()

    payload = request.get_json(force=True) or {}
    user_id = int(payload.get("user_id", 0))
    features = payload.get("features") or {}

    if not isinstance(features, dict):
        return jsonify({"error": "bad_request", "detail": "`features` must be an object"}), 400

    variant = "B" if (user_id % 100) < int(AB_SPLIT_B * 100) else "A"
    stage = "Production" if variant == "A" else "Staging"
    alias = "prod" if stage == "Production" else "staging"

    try:
        model = load_model_by_alias(alias)
        pred = model.predict(pd.DataFrame([features]))
        prediction = pred[0].item() if hasattr(pred[0], "item") else pred[0]
    except MlflowException as e:
        return jsonify({"error": "model_not_available", "detail": str(e), "model": REG_MODEL_NAME, "alias": alias}), 503
    except Exception as e:
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                datetime.utcnow().isoformat(),
                user_id,
                variant,
                alias,
                REG_MODEL_NAME,
                json.dumps(features, ensure_ascii=False),
                prediction,
            ]
        )

    return jsonify(
        {
            "variant": variant,
            "alias": alias,
            "model": REG_MODEL_NAME,
            "prediction": prediction,
            "latency_ms": int((time.time() - t0) * 1000),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT")), debug=False)
