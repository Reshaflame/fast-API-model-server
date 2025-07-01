"""
Unified preprocessing utilities for FastAPI inference & on‑the‑fly retraining.

✓ Mirrors the feature‑engineering logic of the training server.
✓ Reads metadata & frequency dictionaries from app/data/.
✓ Guarantees output column order exactly matches expected_features.json.

Main entry‑points
-----------------
preprocess_df(df: pd.DataFrame) -> pd.DataFrame
preprocess_records(records: List[dict]) -> torch.Tensor  # ready for model.forward()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np  # noqa: F401 – pandas uses NumPy under the hood
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# 📂  Locate repo data directory  (…/app/data)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# 📑  Static metadata — *must* be present (copied from training repo)
# ---------------------------------------------------------------------------
with (DATA_DIR / "expected_features.json").open("r") as f:
    EXPECTED_FEATURES: List[str] = json.load(f)

# Optional helpers — load silently if provided
AUTH_CATS: Dict[str, int] = {}
COMP_FREQ: Dict[str, float] = {}
USER_FREQ: Dict[str, float] = {}

for filename, var in [
    ("auth_categories.json", "AUTH_CATS"),
    ("comp_freq.json", "COMP_FREQ"),
    ("user_freq.json", "USER_FREQ"),
]:
    path = DATA_DIR / filename
    if path.exists():
        with path.open("r") as f:
            globals()[var] = json.load(f)

# ---------------------------------------------------------------------------
# 🏗️  Feature‑engineering helpers
# ---------------------------------------------------------------------------

def add_utc_hour_column(df: pd.DataFrame, ts_col: str = "time") -> None:
    """Append an integer [0‑23] column derived from epoch‑seconds column *time*."""
    if ts_col not in df.columns:
        return
    utc = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    df["utc_hour"] = utc.dt.hour.fillna(0).astype("int16")


def _freq_encode(series: pd.Series, mapping: Dict[str, float]) -> pd.Series:
    return (
        series.astype(str)
        .map(mapping)         # NaN for unseen keys
        .fillna(0.0)
        .astype("float32")
    )


def add_freq_encoding(df: pd.DataFrame) -> None:
    """Adds pc_freq & user_freq columns if mappings present."""
    if COMP_FREQ and "pc" in df.columns:
        df["pc_freq"] = _freq_encode(df["pc"], COMP_FREQ)
    if USER_FREQ and "user" in df.columns:
        df["user_freq"] = _freq_encode(df["user"], USER_FREQ)


def label_encode_auth(df: pd.DataFrame) -> None:
    if AUTH_CATS and "auth_type" in df.columns:
        df["auth_type_enc"] = (
            df["auth_type"].astype(str).map(AUTH_CATS).fillna(0).astype("int16")
        )

# ---------------------------------------------------------------------------
# 🎛️  Public API
# ---------------------------------------------------------------------------

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame whose columns == EXPECTED_FEATURES (float32)."""
    df = df.copy()

    # --- feature engineering parity with training server ---
    add_utc_hour_column(df, "time")
    add_freq_encoding(df)
    label_encode_auth(df)

    # Fill any remaining NaNs – works for mixed dtypes
    df = df.fillna(0.0)

    # Ensure every expected column exists
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # Re‑order & cast
    df = df[EXPECTED_FEATURES]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    return df.astype("float32")


def preprocess_records(records: List[Dict[str, Any]]) -> torch.Tensor:
    """Shortcut helper – accepts list‑of‑dicts (FastAPI payload) and returns tensor."""
    if not records:
        raise ValueError("records list is empty")

    df = pd.DataFrame(records)
    processed = preprocess_df(df)
    # (B, F) float32 tensor, ready for model.forward()
    return torch.from_numpy(processed.values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 🔬  Quick self‑test (run `python -m app.utils.preprocess_utils`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = [{
        "time": 1609459200,  # 2021‑01‑01 00:00 UTC
        "pc": "PC‑01",
        "user": "alice",
        "auth_type": "Kerberos",
    }]
    tensor = preprocess_records(sample)
    print("Tensor shape:", tensor.shape)
    print("First row slice:", tensor[0, :10])
