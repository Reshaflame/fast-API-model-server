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
    
    # === Ensure expected columns exist ===
    for col in ["src_user", "dst_user", "src_computer", "dst_computer"]:
        if col not in df.columns:
            df[col] = ""
    
    # === Raw time (as int, same as training) ===
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype("int64")
    df["time"] = df["time"].clip(lower=0)

    # === Frequency-encode (no log or scaling) ===
    df["src_user_freq"] = df["src_user"].map(USER_FREQ).fillna(0)
    df["dst_user_freq"] = df["dst_user"].map(USER_FREQ).fillna(0)
    df["src_comp_freq"] = df["src_computer"].map(COMP_FREQ).fillna(0)
    df["dst_comp_freq"] = df["dst_computer"].map(COMP_FREQ).fillna(0)

    # === Categorical handling (just like training) ===
    for col in AUTH_CATEGORIES:
        df[col] = df[col].astype(str)
        allowed = set(AUTH_CATEGORIES[col])
        df[col] = df[col].where(df[col].isin(allowed), other="OTHER")
        df[col] = pd.Categorical(df[col], categories=AUTH_CATEGORIES[col] + ["OTHER"])

    # === One-hot ===
    df = pd.get_dummies(df, columns=AUTH_CATEGORIES.keys(), dummy_na=False)

    # === Align with expected features (pad missing one-hot columns) ===
    with open(expected_features_path) as f:
        expected = json.load(f)

    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[expected]

    print("📈 Preprocessed sample row:")
    print(df.head(3))

    df = df.astype(np.float32)

    # === Reshape to sequence [B, 10, F] like in training ===
    tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    padded_tensor = tensor.unsqueeze(1).repeat(1, 10, 1)  # [B, 10, F]

    return padded_tensor
