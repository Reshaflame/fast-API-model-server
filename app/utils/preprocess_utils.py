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
        # Try to parse timestamps from string (ISO format or otherwise)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["time"] = df["time"].view("int64") // 1_000_000_000
    else:
        # Already numeric (int/float), just cast safely
        df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)

    df["time"] = df["time"].clip(lower=0)  # remove invalid negative timestamps

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
    
    print("📈 Sample preprocessed row:")
    print(df.head(3))
    
    df = df.astype(np.float32)

    # === Shape for GRU: [B, 1, F]
    # === Shape for LSTM+RNN: [B, 10, F] ← simulate a flat sequence
    tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    sequence_length = 10
    padded_tensor = tensor.unsqueeze(1).repeat(1, sequence_length, 1)  # [B, 10, F]

    return padded_tensor
