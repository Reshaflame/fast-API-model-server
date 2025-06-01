import torch
import pandas as pd
import json
import os
from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_gru_with_guess, load_lstm_with_guess, load_isolation_forest
from app.models.ensemble_head import EnsembleMLP

# 📦 Load expected feature size
with open("data/expected_features.json") as f:
    input_size = len(json.load(f))

# 🧠 Load models
GRU_MODEL = load_gru_with_guess("models/gru_trained_model.pth", input_size)
LSTM_MODEL = load_lstm_with_guess("models/lstm_rnn_trained_model.pth", input_size)
ISO_MODEL = load_isolation_forest("models/isolation_forest_model.joblib")
MLP_HEAD = EnsembleMLP()

weights_path = "models/mlp_weights.pth"
if os.path.exists(weights_path):
    MLP_HEAD.load_state_dict(torch.load(weights_path))
    print("✅ Loaded saved MLP ensemble weights.")
else:
    print("⚠️ No saved MLP weights found — using default voting weights.")

# ⚖️ Manual ensemble weights (sum to ~1.0)
W_GRU = 0.4167
W_LSTM = 0.4167
W_ISO = 0.1666

def predict_batch(req):
    print(f"📨 Received batch of {len(req.data)} rows.")
    if not req.data:
        return {"error": "No data received."}

    input_tensor = preprocess_batch([row.dict() for row in req.data])  # [B, 10, F]

    # Optional debug: show numeric input stats
    raw_input_matrix = input_tensor[:, -1, :].numpy()
    df_debug = pd.DataFrame(raw_input_matrix)
    print("📊 Sample of preprocessed vectors (last timestep):")
    print(df_debug.head(3))
    print("📈 Stats:")
    print(df_debug.describe())
    print(f"🧮 Preprocessed tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        gru_out = GRU_MODEL(input_tensor).squeeze()
        lstm_out = LSTM_MODEL(input_tensor).squeeze()
        
        # NOTE: Models were trained with flipped labels (-1=anomaly, 1=normal)
        # So we invert the sigmoid output to match "higher = anomaly"

        gru_scores = (1 - gru_out).tolist() if gru_out.ndim > 0 else [1 - gru_out.item()]
        lstm_scores = (1 - lstm_out).tolist() if lstm_out.ndim > 0 else [1 - lstm_out.item()]


    print(f"✅ Flipped GRU scores (anomaly perspective): {gru_scores}")
    print(f"✅ Flipped LSTM scores (anomaly perspective): {lstm_scores}")   

    iso_input = raw_input_matrix  # last timestep
    try:
        iso_scores = ISO_MODEL.decision_function(iso_input)
        if isinstance(iso_scores, float) or not hasattr(iso_scores, "__len__"):
            iso_scores = [iso_scores] * len(req.data)
        print(f"🧪 ISO scores: {iso_scores}")
    except Exception as e:
        print(f"⚠️ Isolation Forest error: {e}")
        iso_scores = [0.0] * len(req.data)

    # 🎛 Option A: manual ensemble logic
    print("📥 Combining scores (manual weights)...")
    for i in range(len(req.data)):
        print(f"Row {i}: GRU={gru_scores[i]}, LSTM={lstm_scores[i]}, ISO={iso_scores[i]}")

    ensemble_preds = [
        W_GRU * gru_scores[i] + W_LSTM * lstm_scores[i] + W_ISO * iso_scores[i]
        for i in range(len(req.data))
    ]

    # 🎛 Option B: Use MLP (uncomment to enable MLP-based prediction)
    # input_scores = torch.tensor(
    #     [[gru_scores[i], lstm_scores[i], iso_scores[i]] for i in range(len(req.data))],
    #     dtype=torch.float32
    # )
    # with torch.no_grad():
    #     raw_preds = MLP_HEAD(input_scores).squeeze()
    #     ensemble_preds = raw_preds.tolist() if raw_preds.ndim > 0 else [raw_preds.item()]

    return [
        {
            "row_id": row.row_id,
            "anomaly": pred > 0.5,
            "score": round(pred, 4)
        }
        for row, pred in zip(req.data, ensemble_preds)
    ]
