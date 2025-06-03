import pandas as pd
import torch
from app.services.predictor import GRU_MODEL, LSTM_MODEL, ISO_MODEL, MLP_HEAD, USE_MLP
import json
import numpy as np

FILE_PATH = "data/labeled_data/chunks/chunk_0_labeled.csv"
DEBUG_EXPORT_PATH = "debug/debug_sample_from_chunk.csv"
EXPECTED_FEATURES_PATH = "app/data/expected_features.json"

def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df

def preview_raw_data(df):
    print("🧾 First 5 raw rows:")
    print(df.head())
    print("\n📊 Columns:", df.columns.tolist())

def export_debug_sample(df):
    try:
        debug_sample = df.head(100)
        debug_sample.to_csv(DEBUG_EXPORT_PATH, index=False)
        print(f"💾 Saved first 100 rows to {DEBUG_EXPORT_PATH} for inspection.")
    except Exception as e:
        print(f"❌ Failed to export debug sample: {e}")


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

    print("GRU param count:", sum(p.numel() for p in GRU_MODEL.parameters()))
    print("GRU param sum  :", sum(p.sum().item() for p in GRU_MODEL.parameters()))

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

    # === Optional: Class Balance Diagnostics ===
    if "label" in df.columns:
        print("\n📊 Label Distribution:")
        print(df["label"].value_counts())
        print(f"📉 Anomaly Ratio: {df['label'].eq(-1).mean():.2%}")

    # === Align features to match expected features.json ===
    with open(EXPECTED_FEATURES_PATH) as f:
        expected_features = json.load(f)

    missing = [col for col in expected_features if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_features and col != "label"]

    if missing:
        print(f"❌ Missing columns in chunk: {missing}")
    if extra:
        print(f"⚠️ Extra columns in chunk (ignored): {extra}")

    df_features = df[expected_features].copy()
    export_debug_sample(df_features)

    df_np = df_features.astype(np.float32).to_numpy()

    # === Tensor diagnostics ===
    print("\n🧪 Inference Tensor Stats:")
    print(f"Shape         : {df_np.shape}")
    print(f"Mean          : {df_np.mean():.6f}")
    print(f"Non-zero %    : {(df_np != 0).mean() * 100:.2f}%")

    tensor = torch.tensor(df_np, dtype=torch.float32)
    tensor_seq = tensor.unsqueeze(1).repeat(1, 10, 1)  # [B, 10, F]
    last_timestep = tensor_seq[:, -1, :].numpy()       # For ISO

    row_ids = [f"row_{i}" for i in range(len(df_features))]
    results = run_all_models(tensor_seq, last_timestep, row_ids)
    summarize(results)
