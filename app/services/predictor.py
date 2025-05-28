import torch
import json
from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_gru, load_lstm, load_isolation_forest

# 📦 Load input size
with open("data/expected_features.json") as f:
    input_size = len(json.load(f))

# 🧠 Load models
GRU_MODEL = load_gru("models/gru_trained_model.pth", input_size)
LSTM_MODEL = load_lstm("models/lstm_rnn_trained_model.pth", input_size)
ISO_MODEL = load_isolation_forest()  # mock

# Ensemble weights
W_GRU = 0.5
W_LSTM = 0.5
W_ISO = 0.0

def predict_batch(req):
    input_tensor = preprocess_batch([row.dict() for row in req.data])  # [B, 1, F]

    with torch.no_grad():
        gru_scores = GRU_MODEL(input_tensor).squeeze().tolist()
        lstm_scores = LSTM_MODEL(input_tensor).squeeze().tolist()

    # Mock ISO scores (0.0 for now)
    iso_scores = ISO_MODEL.decision_function(None)
    if isinstance(iso_scores, float):
        iso_scores = [iso_scores] * len(req.data)

    predictions = []
    for i, row in enumerate(req.data):
        final_score = (
            W_GRU * gru_scores[i] +
            W_LSTM * lstm_scores[i] +
            W_ISO * iso_scores[i]
        )
        anomaly = final_score > 0.5
        predictions.append({
            "row_id": row.row_id,
            "anomaly": bool(anomaly),
            "score": round(final_score, 4)
        })

    return predictions
