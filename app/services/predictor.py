import torch
import pandas as pd
import os
from app.utils.preprocess_utils import preprocess_records, EXPECTED_FEATURES
from app.services.model_loader import load_gru_with_guess, load_lstm_with_guess, load_isolation_forest
from app.models.ensemble_head import EnsembleMLP

# 📦 Feature dimension comes straight from preprocess_utils
input_size = len(EXPECTED_FEATURES)

# 🧠 Load models
GRU_MODEL = load_gru_with_guess("models/gru_trained_model.pth", input_size)
LSTM_MODEL = load_lstm_with_guess("models/lstm_rnn_trained_model.pth", input_size)
ISO_MODEL = load_isolation_forest("models/isolation_forest_model.joblib")
MLP_HEAD = EnsembleMLP()

# ------------------------------------------------------------
# Freeze GRU backbone + residual block; leave .fc trainable
# ------------------------------------------------------------
for name, param in GRU_MODEL.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

# 🎛 Ensemble control
weights_path = "models/mlp_weights.pth"
USE_MLP = os.path.exists(weights_path)

if USE_MLP:
    MLP_HEAD.load_state_dict(torch.load(weights_path))
    print("✅ Loaded saved MLP ensemble weights.")
else:
    print("⚠️ No saved MLP weights found — using manual voting weights.")

# ⚖️ Updated manual ensemble weights (sum to 1.0)
W_GRU = 0.0   # GRU model currently being retrained
W_LSTM = 0.9  # LSTM+RNN carries most of the vote
W_ISO = 0.1   # Isolation‑Forest small contribution

def predict_batch(req):
    """Run ensemble prediction on a FastAPI request body."""
    print(f"📨 Received batch of {len(req.data)} rows.")
    if not req.data:
        return {"error": "No data received."}

    # === Feature engineering ===
    base_feats = preprocess_records([row.dict() for row in req.data])  # (B, F)

    # Make a “fake” 10-step sequence for LSTM & legacy logic
    seq_len      = 10
    input_tensor = base_feats.unsqueeze(1).repeat(1, seq_len, 1)       # (B, 10, F)
    
    # Diagnostics
    raw_input_matrix = input_tensor[:, -1, :].numpy()
    print("🧮 Preprocessed tensor shape:", input_tensor.shape)

    # === Model forward passes ===
    with torch.no_grad():
        gru_out  = GRU_MODEL(input_tensor).squeeze()
        lstm_out = LSTM_MODEL(input_tensor).squeeze()

        # 🚫 NO FLIPPING — models already output 1 = anomaly, 0 = normal
        gru_scores  = gru_out.tolist()  if gru_out.ndim  > 0 else [gru_out.item()]
        lstm_scores = lstm_out.tolist() if lstm_out.ndim > 0 else [lstm_out.item()]

    # Isolation‑Forest (already 0‑1 normalised in wrapper)
    try:
        iso_scores = ISO_MODEL.decision_function(raw_input_matrix)
        if isinstance(iso_scores, float) or not hasattr(iso_scores, "__len__"):
            iso_scores = [iso_scores] * len(req.data)
    except Exception as e:
        print(f"⚠️ Isolation Forest error: {e}")
        iso_scores = [0.0] * len(req.data)

    # === Ensemble ===
    if USE_MLP:
        input_scores = torch.tensor(
            [[gru_scores[i], lstm_scores[i], iso_scores[i]] for i in range(len(req.data))],
            dtype=torch.float32,
        )
        with torch.no_grad():
            raw_preds = MLP_HEAD(input_scores).squeeze()
            ensemble_preds = raw_preds.tolist() if raw_preds.ndim > 0 else [raw_preds.item()]
        print("🤖 MLP ensemble used.")
    else:
        ensemble_preds = [
            W_GRU * gru_scores[i] + W_LSTM * lstm_scores[i] + W_ISO * iso_scores[i]
            for i in range(len(req.data))
        ]
        print("⚖️ Manual weighted ensemble used.")

    # === Build response ===
    return [
        {
            "row_id": row.row_id,
            "anomaly": bool(pred > 0.9),   # cast to plain Python bool
            "score":  round(float(pred), 4) # cast to Python float
        }
        for row, pred in zip(req.data, ensemble_preds)
    ]
