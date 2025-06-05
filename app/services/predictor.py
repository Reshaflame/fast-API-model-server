from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_model, load_isolation_forest
from app.models.ensemble_head import EnsembleMLP, LoRAEnsemble, DeepHead
import torch, json, os
import numpy as np

# ---------- load once ----------
with open("data/expected_features.json") as f:
    input_size = len(json.load(f))

GRU_MODEL = load_model("models/gru_trained_model.pth", input_size)
ISO_MODEL = load_isolation_forest("models/isolation_forest_model.joblib")
base_W = torch.tensor([[0.8, 0.2, 0.0]], dtype=torch.float32)   
base_b = torch.zeros(1)
MLP_HEAD  = LoRAEnsemble(base_W, base_b, rank=1, alpha=1.0)
base_features = 2                # gru + iso
HEAD = DeepHead()

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

    # ─── Model scores ─────────────────────────────────────────────
    with torch.no_grad():
        # always 1-D tensor [B] – even when B == 1
        gru_raw = GRU_MODEL(input_tensor).flatten(start_dim=0)

    gru_scores = gru_raw.tolist()                      # → list of floats

    # Isolation-Forest: decision_function returns scalar when B == 1
    iso_raw = ISO_MODEL.decision_function(input_tensor[:, -1, :].numpy())
    iso_scores = (
        np.atleast_1d(iso_raw).astype(float).tolist()
    )  # → list of floats, length B

    # ─── Ensemble ──────────────────────────────────────────────────────────
    weights_path = "models/mlp_weights.pth"
    if os.path.exists(weights_path):
        input_scores = torch.tensor(
            [[gru_scores[i], iso_scores[i]] for i in range(len(req.data))],
            dtype=torch.float32,
        )
        with torch.no_grad():
            raw_preds = HEAD(input_scores).squeeze()
            ensemble_preds = raw_preds.tolist() if raw_preds.ndim > 0 else [raw_preds.item()]
        print("🤖 MLP ensemble used.")
        print("📈 probs", ensemble_preds[:10])
    else:
        ensemble_preds = [
            W_GRU * gru_scores[i] + W_ISO * iso_scores[i]
            for i in range(len(req.data))
        ]
        print("⚖️ Manual weighted ensemble used.")


    binary_preds = [p > 0.5 for p in ensemble_preds]

    # ─── Response ──────────────────────────────────────────────────────────
    return [
        {"row_id": row.row_id, "anomaly": bool(binary_preds[i])}
        for i, row in enumerate(req.data)
    ]

# ──────────────────────────────────────────────────────────────────────────
# Runtime reset for DeepHead
# ──────────────────────────────────────────────────────────────────────────
def reset_head():
    """
    • Deletes models/mlp_weights.pth  (if present)
    • Re-instantiates HEAD with fresh, random weights
    • Returns True if a file was removed, else False
    """
    import os
    global HEAD

    removed = False
    weight_path = "models/mlp_weights.pth"
    if os.path.exists(weight_path):
        os.remove(weight_path)
        removed = True

    # re-create a brand-new head (same class & device)
    HEAD = DeepHead()
    HEAD.eval()           # default state

    return removed
