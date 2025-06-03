import pandas as pd
import gzip
import torch
from app.utils.preprocess_utils import preprocess_batch
from app.services.predictor import GRU_MODEL, LSTM_MODEL, ISO_MODEL, MLP_HEAD, USE_MLP
import json

FILE_PATH = "app/merged_100.txt.gz"

DEBUG_EXPORT_PATH = "app/debug_merged_100.csv"

EXPECTED_COLUMNS = [
    "row_id", "time", "src_user", "dst_user",
    "auth_type", "logon_type", "auth_orientation", "success"
]

def load_csv(filepath):
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)
    return df

def export_raw_for_debug(df):
    print(f"💾 Saving raw CSV snapshot to {DEBUG_EXPORT_PATH}")
    df.to_csv(DEBUG_EXPORT_PATH, index=False)

def validate_columns(df):
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        if "row_id" in missing:
            print("🛠️ Generating row_id...")
            df["row_id"] = [f"row_{i}" for i in range(len(df))]
            missing.remove("row_id")
        if missing:
            raise ValueError("🚫 Cannot proceed — required fields missing.")
    print("✅ Column validation passed.")
    return df

def preview_raw_data(df):
    print("🧾 First 5 raw rows:")
    print(df.head())
    print("\n📊 Columns:", df.columns.tolist())

def run_all_models(preprocessed_tensor, raw_matrix, row_ids):
    with torch.no_grad():
        gru_out = GRU_MODEL(preprocessed_tensor).squeeze()
        lstm_out = LSTM_MODEL(preprocessed_tensor).squeeze()

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
    print("🚀 Loading CSV...")
    df = load_csv(FILE_PATH)
    preview_raw_data(df)
    df = validate_columns(df)

    export_raw_for_debug(df)  # 🔥 Save for inspection

    raw_dicts = df.to_dict(orient="records")
    pre_tensor = preprocess_batch(raw_dicts)        # [B, 10, F]
    raw_matrix = pre_tensor[:, -1, :].numpy()       # [B, F] for ISO

    row_ids = df["row_id"].tolist()
    results = run_all_models(pre_tensor, raw_matrix, row_ids)
    summarize(results)
