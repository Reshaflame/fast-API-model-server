from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_model, load_isolation_forest
from app.models.ensemble_head import EnsembleMLP
import torch, json, os

# ---------- load once ----------
with open("data/expected_features.json") as f:
    input_size = len(json.load(f))

GRU_MODEL = load_model("models/gru_trained_model.pth", input_size)
ISO_MODEL = load_isolation_forest("models/isolation_forest_model.joblib")
MLP_HEAD  = EnsembleMLP()

USE_MLP   = os.path.exists("models/mlp_weights.pth")
if USE_MLP:
    MLP_HEAD.load_state_dict(torch.load("models/mlp_weights.pth"))
    print("✅  Loaded saved MLP ensemble weights.")
else:
    print("⚠️  No saved MLP weights found — using manual voting.")

# Manual weights if the MLP isn’t trained yet
W_GRU, W_ISO = 0.8, 0.2          # sum to 1.0
# ---------------------------------

def predict_batch(req):
    """Takes FastAPI RequestBody → returns list[{row_id, anomaly}]"""
    input_tensor = preprocess_batch([row.dict() for row in req.data])

    # ─── Model scores ───────────────────────────────────────────────────────
    with torch.no_grad():
        gru_raw = GRU_MODEL(input_tensor).squeeze()              # tensor[B]
    gru_scores = gru_raw.tolist()                                # python list

    iso_scores = ISO_MODEL.decision_function(
        input_tensor[:, -1, :].numpy()
    )                                                            # list/ndarray

    # ─── Ensemble ──────────────────────────────────────────────────────────
    if USE_MLP:
        mlp_in = torch.tensor(
            [[gru_scores[i], iso_scores[i], 0.0]                 # 3-input MLP
             for i in range(len(req.data))],
            dtype=torch.float32,
        )
        with torch.no_grad():
            final_probs = MLP_HEAD(mlp_in).squeeze().tolist()
        print("🤖  MLP ensemble used.")
    else:
        final_probs = [
            W_GRU * gru_scores[i] + W_ISO * iso_scores[i]
            for i in range(len(req.data))
        ]
        print("⚖️  Manual weighted ensemble used.")

    binary_preds = [p > 0.5 for p in final_probs]

    # ─── Response ──────────────────────────────────────────────────────────
    return [
        {"row_id": row.row_id, "anomaly": bool(binary_preds[i])}
        for i, row in enumerate(req.data)
    ]
