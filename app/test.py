import gzip
from app.services.predictor import predict_batch
from app.utils.preprocess_utils import preprocess_batch
from app.services.predictor import GRU_MODEL, LSTM_MODEL, ISO_MODEL
import torch
import pandas as pd

FILE_PATH = "app/merged_100.txt.gz"

EXPECTED_COLUMNS = [
    "row_id", "time", "src_user", "dst_user",
    "auth_type", "logon_type", "auth_orientation", "success"
]

def load_csv(filepath):
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)
    return df

def preview_data(df):
    print("🧾 First 5 rows:")
    print(df.head())
    print("\n📊 Columns:", df.columns.tolist())

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in CSV: {missing}")
        if "row_id" in missing:
            print("🛠️ Generating default row_id values...")
            df["row_id"] = [f"row_{i}" for i in range(len(df))]
            missing.remove("row_id")

        if missing:
            raise ValueError("🚫 Cannot continue — required columns are missing.")
    else:
        print("✅ All required columns are present.")


def run_predictions(data):
    batch_like = {"batch_id": "test_batch", "data": data}
    results = predict_batch(type("Obj", (object,), batch_like))
    return results

def summarize_results(results):
    print("\n🔎 Prediction Summary:")
    anomaly_count = sum(1 for r in results if r["anomaly"])
    normal_count = len(results) - anomaly_count
    print(f"Total: {len(results)} | Anomalies: {anomaly_count} | Normal: {normal_count}")

    print("\n🧪 Sample Results:")
    for row in results[:5]:
        print(f"{row['row_id']}: Anomaly={row['anomaly']}, Score={row['score']}")

if __name__ == "__main__":
    print("🚀 Loading and testing file...")
    df = load_csv(FILE_PATH)
    preview_data(df)

    data = df.to_dict(orient="records")
    results = run_predictions(data)
    summarize_results(results)


