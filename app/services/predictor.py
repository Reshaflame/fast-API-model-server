from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_model
from app.models.ensemble_head import EnsembleMLP
import torch
import json

# 🔁 Load once
with open("data/expected_features.json") as f:
    input_size = len(json.load(f))

MODEL_PATH = "models/gru_trained_model.pth"
MODEL = load_model(MODEL_PATH, input_size)
ISO_MODEL = load_isolation_forest("models/isolation_forest_model.joblib")
MLP_HEAD = EnsembleMLP()

# 🎛 Ensemble control
weights_path = "models/mlp_weights.pth"
USE_MLP = os.path.exists(weights_path)

if USE_MLP:
    MLP_HEAD.load_state_dict(torch.load(weights_path))
    print("✅ Loaded saved MLP ensemble weights.")
else:
    print("⚠️ No saved MLP weights found — using manual voting weights.")

# ⚖️ Updated manual ensemble weights (sum to 1.0)
W_GRU = 0.8   # GRU model currently being retrained
# W_LSTM = 0.9  # LSTM+RNN carries most of the vote
W_ISO = 0.2   # Isolation‑Forest small contribution

def predict_batch(req):
    input_tensor = preprocess_batch([row.dict() for row in req.data])
    with torch.no_grad():
        preds = MODEL(input_tensor)
        binary_preds = (preds > 0.5).squeeze().tolist()

    return [
        {"row_id": row.row_id, "anomaly": bool(binary_preds[i])}
        for i, row in enumerate(req.data)
    ]

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