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
    input_tensor = preprocess_batch([row.dict() for row in req.data])  # [B, 10, F]

    with torch.no_grad():
        gru_scores = GRU_MODEL(input_tensor).squeeze().tolist()
        lstm_scores = LSTM_MODEL(input_tensor).squeeze().tolist()

    iso_input = input_tensor[:, -1, :].numpy()
    try:
        iso_scores = ISO_MODEL.decision_function(iso_input)
        if isinstance(iso_scores, float) or not hasattr(iso_scores, "__len__"):
            iso_scores = [iso_scores] * len(req.data)
    except Exception as e:
        print(f"⚠️ Isolation Forest error: {e}")
        iso_scores = [0.0] * len(req.data)

    # 🧠 Combine scores and run through MLP
    input_scores = torch.tensor(
        [[gru_scores[i], lstm_scores[i], iso_scores[i]] for i in range(len(req.data))],
        dtype=torch.float32
    )

    with torch.no_grad():
        ensemble_preds = MLP_HEAD(input_scores).squeeze().tolist()

    return [
        {
            "row_id": row.row_id,
            "anomaly": pred > 0.5,
            "score": round(pred, 4)
        }
        for row, pred in zip(req.data, ensemble_preds)
    ]

