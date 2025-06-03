import pandas as pd
import torch
from app.services.predictor import GRU_MODEL, LSTM_MODEL, ISO_MODEL, MLP_HEAD, USE_MLP
import json
import numpy as np

FILE_PATH = "data/labeled_data/chunks/chunk_0_labeled.csv"
DEBUG_EXPORT_PATH = "debug/debug_sample_from_chunk.csv"

def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df

def preview_raw_data(df):
    print("🧾 First 5 raw rows:")
    print(df.head())
    print("\n📊 Columns:", df.columns.tolist())

def export_debug_sample(df):
    debug_sample = df.head(100)
    debug_sample.to_csv(DEBUG_EXPORT_PATH, index=False)
    print(f"💾 Saved first 100 rows to {DEBUG_EXPORT_PATH} for inspection.")

def run_all_models(input_tensor, raw_matrix, row_ids):
    with torch.no_grad():
        gru_out = GRU_MODEL(input_tensor).squeeze()
        lstm_out = LSTM_MODEL(input_tensor).squeeze()

        gru_scores = (1 - gru_out).tolist()
        lstm_scores = (1 - lstm_out).tolist()

    try:
        iso_scores = ISO_MODEL.decision_function(raw_matrix)
    except Exception as e:
        print(f"⚠️ ISO model failed: {e}")
        iso_scores = [0.0] * len(row_ids)

    print(f"✅ GRU sample scores: {gru_scores[:3]}")
    print(f"✅ LSTM sample scores: {lstm_scores[:3]}")
    print(f"✅ ISO sample scores: {iso_scores[:3]}")

    # Ensemble logic
    if USE_MLP:
        print("🤖 Using MLP ensemble...")
        input_scores = torch.tensor(
            list(zip(gru_scores, lstm_scores, iso_scores)),
            dtype=torch.float32
        )
        with torch.no_grad():
            mlp_preds = MLP_HEAD(input_scores).squeeze()
        ensemble_preds = mlp_preds.tolist()
    else:
        print("⚖️ Using manual weights...")
        W_GRU, W_LSTM, W_ISO = 0.4167, 0.4167, 0.1666
        ensemble_preds = [
            W_GRU * g + W_LSTM * l + W_ISO * i
            for g, l, i in zip(gru_scores, lstm_scores, iso_scores)
        ]

    results = [
        {"row_id": rid, "anomaly": pred > 0.5, "score": round(pred, 4)}
        for rid, pred in zip(row_ids, ensemble_preds)
    ]
    return results

def summarize(results):
    total = len(results)
    anomalies = sum(1 for r in results if r["anomaly"])
    print(f"\n🔍 Final Summary:\nTotal Rows: {total}\nAnomalies Detected: {anomalies}\nNormal: {total - anomalies}")
    print("\n🧪 Sample Results:")
    for r in results[:5]:
        print(f"{r['row_id']}: anomaly={r['anomaly']} (score={r['score']})")

if __name__ == "__main__":
    print("🚀 Loading preprocessed chunk...")
    df = load_csv(FILE_PATH)
    preview_raw_data(df)
    export_debug_sample(df)

    # === Feature alignment ===
    df_features = df.drop(columns=["label"])  # already preprocessed
    # Drop string-identifying columns not used in training
    drop_if_exists = ["src_user", "dst_user", "src_comp", "dst_comp"]
    df_features = df_features.drop(columns=[col for col in drop_if_exists if col in df_features.columns])
    row_ids = [f"row_{i}" for i in range(len(df_features))]

    df_np = df_features.astype(np.float32).to_numpy()
    tensor = torch.tensor(df_np, dtype=torch.float32)
    tensor_seq = tensor.unsqueeze(1).repeat(1, 10, 1)  # [B, 10, F]
    last_timestep = tensor_seq[:, -1, :].numpy()       # For ISO

    results = run_all_models(tensor_seq, last_timestep, row_ids)
    summarize(results)
