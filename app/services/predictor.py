import torch
import json
from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_gru, load_lstm, load_isolation_forest
from app.models.ensemble_head import EnsembleMLP


# 📦 Load input size
with open("data/expected_features.json") as f:
    input_size = len(json.load(f))

# 🧠 Load models
GRU_MODEL = load_gru("models/gru_trained_model.pth", input_size)
LSTM_MODEL = load_lstm("models/lstm_rnn_trained_model.pth", input_size)
ISO_MODEL = load_isolation_forest()  # mock
MLP_HEAD = EnsembleMLP()


# Ensemble weights
W_GRU = 0.5
W_LSTM = 0.5
W_ISO = 0.0

def predict_batch(req):
    input_tensor = preprocess_batch([row.dict() for row in req.data])  # [B, 10, F]

    with torch.no_grad():
        gru_scores = GRU_MODEL(input_tensor).squeeze().tolist()
        lstm_scores = LSTM_MODEL(input_tensor).squeeze().tolist()

    iso_input = input_tensor[:, -1, :].numpy()
    iso_scores = ISO_MODEL.decision_function(None)
    if isinstance(iso_scores, float):
        iso_scores = [iso_scores] * len(req.data)

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

