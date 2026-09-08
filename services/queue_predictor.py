"""
Loads the trained queue wait-time model and turns a request into a
prediction. See models/queue_prediction/train.py for how the model is
trained, and datasets/README.md for the training data.
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from schemas.queue import QueueRequest

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "queue_prediction" / "model.pkl"

_artifact: dict | None = None


def _load_artifact() -> dict:
    global _artifact
    if _artifact is None:
        _artifact = joblib.load(_MODEL_PATH)
    return _artifact


def _time_to_minutes(time_str: str) -> int:
    hours, minutes = time_str.split(":")
    return int(hours) * 60 + int(minutes)


def _build_features(request: QueueRequest, artifact: dict) -> pd.DataFrame:
    row = {
        "current_queue_length": request.current_queue_length,
        "average_consultation_minutes": request.average_consultation_minutes,
        "minutes_since_midnight": _time_to_minutes(request.time_of_day),
        "emergency_patients_ahead": request.emergency_patients_ahead,
    }
    for day in artifact["days"]:
        row[f"day_{day}"] = 1 if request.day_of_week == day else 0
    return pd.DataFrame([row])[artifact["feature_columns"]]


def _confidence(request: QueueRequest, artifact: dict) -> Literal["low", "medium", "high"]:
    # Reflects how well the training data actually supports this input,
    # not a made-up number: how close current_queue_length and
    # average_consultation_minutes fall to the training distribution.
    dist = artifact["distribution"]
    in_range = 0
    for field in ("current_queue_length", "average_consultation_minutes"):
        bounds = dist[field]
        if bounds["p5"] <= getattr(request, field) <= bounds["p95"]:
            in_range += 1
    if in_range == 2:
        return "high"
    if in_range == 0:
        return "low"
    return "medium"


def predict(request: QueueRequest) -> dict:
    artifact = _load_artifact()
    features = _build_features(request, artifact)
    estimated_wait_minutes = max(0, round(float(artifact["model"].predict(features)[0])))
    queue_position = request.current_queue_length + 1
    confidence = _confidence(request, artifact)

    explanation = (
        "Based on current queue length and average consultation time for this doctor, "
        "adjusted for time of day and day of week."
    )
    if request.emergency_patients_ahead:
        explanation += (
            f" {request.emergency_patients_ahead} emergency patient(s) ahead increase the wait."
        )

    return {
        "estimated_wait_minutes": estimated_wait_minutes,
        "queue_position": queue_position,
        "confidence": confidence,
        "explanation": explanation,
    }
