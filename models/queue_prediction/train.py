"""
Trains the queue wait-time prediction model.

Usage (from the AI/ directory):
    python -m models.queue_prediction.train

Pipeline:
1. Load datasets/queue_synthetic.csv (see datasets/README.md for how it's generated —
   synthetic data, standing in for real hospital queue data we don't have).
2. Feature engineering: day_of_week -> one-hot, time_of_day "HH:MM" -> minutes since
   midnight.
3. Real train/test split (80/20) — this is a genuine evaluation, not a lookup table.
4. Baseline: LinearRegression, to confirm the pipeline works end-to-end.
5. Production model: XGBoost regressor, per the plan's suggested tech.
6. Evaluate both on the held-out test set (MAE, RMSE, R^2) and print the comparison,
   so it's defensible in the graduation report.
7. Persist the XGBoost model + feature metadata (column order, day categories, and
   training-distribution percentiles used for the `confidence` bucketing at inference
   time) to model.pkl.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

_HERE = Path(__file__).resolve().parent
_DATASET_PATH = _HERE.parent.parent / "datasets" / "queue_synthetic.csv"
_MODEL_PATH = _HERE / "model.pkl"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FEATURE_COLUMNS = [
    "current_queue_length",
    "average_consultation_minutes",
    "minutes_since_midnight",
    "emergency_patients_ahead",
] + [f"day_{d}" for d in DAYS]


def _time_to_minutes(time_str: str) -> int:
    hours, minutes = time_str.split(":")
    return int(hours) * 60 + int(minutes)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["current_queue_length"] = df["current_queue_length"]
    features["average_consultation_minutes"] = df["average_consultation_minutes"]
    features["minutes_since_midnight"] = df["time_of_day"].apply(_time_to_minutes)
    features["emergency_patients_ahead"] = df["emergency_patients_ahead"]
    for day in DAYS:
        features[f"day_{day}"] = (df["day_of_week"] == day).astype(int)
    return features[FEATURE_COLUMNS]


def _report(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, R^2={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    df = pd.read_csv(_DATASET_PATH)
    X = build_features(df)
    y = df["actual_wait_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_metrics = _report("LinearRegression (baseline)", y_test, baseline.predict(X_test))

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    xgb_metrics = _report("XGBoost (production model)", y_test, model.predict(X_test))

    distribution = {
        "current_queue_length": {
            "p5": float(X_train["current_queue_length"].quantile(0.05)),
            "p95": float(X_train["current_queue_length"].quantile(0.95)),
        },
        "average_consultation_minutes": {
            "p5": float(X_train["average_consultation_minutes"].quantile(0.05)),
            "p95": float(X_train["average_consultation_minutes"].quantile(0.95)),
        },
    }

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "days": DAYS,
        "distribution": distribution,
        "metrics": {"baseline_linear_regression": baseline_metrics, "xgboost": xgb_metrics},
    }
    joblib.dump(artifact, _MODEL_PATH)
    print(f"Saved model + metadata to {_MODEL_PATH}")


if __name__ == "__main__":
    main()
