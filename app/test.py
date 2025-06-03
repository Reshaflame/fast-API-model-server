import gzip
import json
from app.services.predictor import predict_batch
from app.utils.preprocess_utils import preprocess_batch
from app.services.predictor import GRU_MODEL, LSTM_MODEL, ISO_MODEL
import torch
import pandas as pd

FILE_PATH = "app/merged_100.txt.gz"

def load_data(filepath):
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return lines

def preview_data(data):
    df = pd.DataFrame(data)
    print("🧾 First 5 rows:")
    print(df.head())
    print("\n📊 Columns:", df.columns.tolist())

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
    data = load_data(FILE_PATH)
    preview_data(data)
    results = run_predictions(data)
    summarize_results(results)
