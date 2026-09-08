"""
Generates the synthetic outpatient queue dataset used to train the
wait-time prediction model. There is no real hospital queue data
available for this project, so scenarios are generated instead — this
script is the documentation of exactly how, and it is fully reproducible
(fixed random seed). See datasets/README.md for why and how this is used.

Dynamics modeled:
- current_queue_length, emergency_patients_ahead: Poisson-distributed
  arrivals (queues form from a stream of essentially-random arrivals).
- average_consultation_minutes: normally distributed around a typical
  appointment length.
- actual_wait_minutes (the training target): driven by queue composition
  (regular + emergency patients ahead) times consultation length, adjusted
  by a time-of-day effect (busier around midday) and a day-of-week effect
  (heavier Monday backlog, lighter weekends), plus Gaussian noise scaled
  to the base wait — real clinics don't run exactly on schedule.

Run directly to (re)generate datasets/queue_synthetic.csv:
    python datasets/generate_synthetic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 5000
OUTPUT_PATH = Path(__file__).resolve().parent / "queue_synthetic.csv"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Monday absorbs weekend backlog; weekends are quieter.
DAY_MULTIPLIER = {
    "Monday": 1.15,
    "Tuesday": 1.05,
    "Wednesday": 1.0,
    "Thursday": 1.0,
    "Friday": 1.05,
    "Saturday": 0.85,
    "Sunday": 0.8,
}

CLINIC_OPEN_MINUTES = 8 * 60  # 08:00
CLINIC_CLOSE_MINUTES = 18 * 60  # 18:00
PEAK_HOUR = 13.0  # clinics tend to run behind schedule around midday


def _time_of_day_multiplier(minutes_since_midnight: np.ndarray) -> np.ndarray:
    hour = minutes_since_midnight / 60.0
    return 1.0 + 0.25 * np.exp(-((hour - PEAK_HOUR) ** 2) / (2 * 3.0**2))


def generate(n_samples: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    current_queue_length = rng.poisson(lam=4, size=n_samples).clip(0, 20)
    average_consultation_minutes = rng.normal(loc=12, scale=3, size=n_samples).clip(5, 30)
    emergency_patients_ahead = rng.poisson(lam=0.3, size=n_samples).clip(0, 5)

    day_of_week = rng.choice(DAYS, size=n_samples)
    minutes_since_midnight = rng.uniform(CLINIC_OPEN_MINUTES, CLINIC_CLOSE_MINUTES, size=n_samples)

    day_mult = np.array([DAY_MULTIPLIER[d] for d in day_of_week])
    time_mult = _time_of_day_multiplier(minutes_since_midnight)

    base_wait = (current_queue_length + emergency_patients_ahead) * average_consultation_minutes
    noise = rng.normal(loc=0, scale=np.maximum(base_wait * 0.15, 3), size=n_samples)

    actual_wait_minutes = np.clip(base_wait * day_mult * time_mult + noise, 0, None)

    time_of_day = [f"{int(m // 60):02d}:{int(m % 60):02d}" for m in minutes_since_midnight]

    return pd.DataFrame(
        {
            "current_queue_length": current_queue_length,
            "average_consultation_minutes": average_consultation_minutes.round(1),
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "emergency_patients_ahead": emergency_patients_ahead,
            "actual_wait_minutes": actual_wait_minutes.round(1),
        }
    )


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
