import torch
import json
from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_gru_with_guess, load_lstm_with_guess, load_isolation_forest
from app.models.ensemble_head import EnsembleMLP
import os

# 📦 Load input size
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

# Ensemble weights
W_GRU = 0.5
W_LSTM = 0.5
W_ISO = 0.2

def predict_batch(req):
    print(f"📨 Received batch of {len(req.data)} rows.")
    print("🔍 First row example:", req.data[0].dict() if req.data else "EMPTY")

    input_tensor = preprocess_batch([row.dict() for row in req.data])  # [B, 10, F]
    print(f"🧮 Preprocessed tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        gru_out = GRU_MODEL(input_tensor).squeeze()
        lstm_out = LSTM_MODEL(input_tensor).squeeze()

        gru_scores = gru_out.tolist() if isinstance(gru_out, torch.Tensor) and gru_out.ndim > 0 else [gru_out.item()]
        lstm_scores = lstm_out.tolist() if isinstance(lstm_out, torch.Tensor) and lstm_out.ndim > 0 else [lstm_out.item()]
        print(f"✅ Final GRU scores: {gru_scores}")
        print(f"✅ Final LSTM scores: {lstm_scores}")

    
    print(f"🔮 GRU scores: {gru_scores}")
    print(f"🔮 LSTM scores: {lstm_scores}")

    iso_input = input_tensor[:, -1, :].numpy()
    print(f"📊 ISO input shape: {iso_input.shape}")

    try:
        iso_scores = ISO_MODEL.decision_function(iso_input)
        print(f"🧪 ISO scores: {iso_scores}")
        if isinstance(iso_scores, float) or not hasattr(iso_scores, "__len__"):
            iso_scores = [iso_scores] * len(req.data)
    except Exception as e:
        print(f"⚠️ Isolation Forest error: {e}")
        iso_scores = [0.0] * len(req.data)

    # 🧠 Combine scores and run through MLP
    print("📥 Combining scores...")
    input_scores = torch.tensor(
        [[gru_scores[i], lstm_scores[i], iso_scores[i]] for i in range(len(req.data))],
        dtype=torch.float32
    )

    with torch.no_grad():
        raw_preds = MLP_HEAD(input_scores).squeeze()
        ensemble_preds = raw_preds.tolist() if isinstance(raw_preds, torch.Tensor) and raw_preds.ndim > 0 else [raw_preds.item()]


    return [
        {
            "row_id": row.row_id,
            "anomaly": pred > 0.5,
            "score": round(pred, 4)
        }
        for row, pred in zip(req.data, ensemble_preds)
    ]

