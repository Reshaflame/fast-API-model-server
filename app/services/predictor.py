from app.utils.preprocess_utils import preprocess_batch
from app.services.model_loader import load_model
import torch
import json

# 🔁 Load once
with open("data/expected_features.json") as f:
    input_size = len(json.load(f))

MODEL_PATH = "models/gru_trained_model.pth"
MODEL = load_model(MODEL_PATH, input_size)

def predict_batch(req):
    input_tensor = preprocess_batch([row.dict() for row in req.data])
    with torch.no_grad():
        preds = MODEL(input_tensor)
        binary_preds = (preds > 0.5).squeeze().tolist()

    return [
        {"row_id": row.row_id, "anomaly": bool(binary_preds[i])}
        for i, row in enumerate(req.data)
    ]



# Mock Predictor:
# def predict_batch(req):
#     predictions = []
#     toggle = False
#     for row in req.data:
#         predictions.append({
#             "row_id": row.row_id,
#             "anomaly": toggle
#         })
#         toggle = not toggle  # Flip between True/False
#     return predictions
