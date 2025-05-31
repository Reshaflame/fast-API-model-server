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
    # Try datetime parsing first (for ISO strings); fallback to numeric
    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    numeric_time = pd.to_numeric(df["time"], errors="coerce")

    # Combine: Use datetime if parsed, else numeric
    df["time"] = np.where(parsed_time.notna(),
                        parsed_time.view("int64") // 1_000_000_000,
                        numeric_time.fillna(0).astype("int64"))

    # Clip to avoid negatives
    df["time"] = df["time"].clip(lower=0)


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
