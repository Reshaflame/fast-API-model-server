import pandas as pd
import numpy as np
import json
import torch

with open("data/auth_categories.json") as f:
    AUTH_CATEGORIES = json.load(f)

with open("data/user_freq.json") as f:
    USER_FREQ = json.load(f)

with open("data/comp_freq.json") as f:
    COMP_FREQ = json.load(f)

def preprocess_batch(json_data: list[dict], expected_features_path="data/expected_features.json"):
    df = pd.DataFrame(json_data)

    # Convert time column to integer seconds if needed
    if df["time"].dtype == object:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["time"] = (df["time"].astype("int64") // 1e9).astype(int)

    # Frequency-encode
    df["src_user_freq"] = df["src_user"].map(USER_FREQ).fillna(0)
    df["dst_user_freq"] = df["dst_user"].map(USER_FREQ).fillna(0)
    df["src_comp_freq"] = df["src_user"].map(COMP_FREQ).fillna(0)
    df["dst_comp_freq"] = df["dst_user"].map(COMP_FREQ).fillna(0)

    # Encode categoricals
    for col in AUTH_CATEGORIES:
        df[col] = df[col].astype(str)
        allowed = set(AUTH_CATEGORIES[col])
        df[col] = df[col].where(df[col].isin(allowed), other='OTHER')
        df[col] = pd.Categorical(df[col], categories=AUTH_CATEGORIES[col] + ['OTHER'])

    df = pd.get_dummies(df, columns=AUTH_CATEGORIES.keys(), dummy_na=False)

    # Align features
    with open(expected_features_path) as f:
        expected = json.load(f)

    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[expected]

    tensor = torch.tensor(df.to_numpy(), dtype=torch.float32).unsqueeze(1)  # [B, 1, F]
    return tensor
