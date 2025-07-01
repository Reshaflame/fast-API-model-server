from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.services.predictor import predict_batch , GRU_MODEL, LSTM_MODEL, MLP_HEAD
from app.utils.preprocess_utils import preprocess_records
import torch


router = APIRouter()

class DataRow(BaseModel):
    row_id: str
    time: str
    src_user: str
    dst_user: str
    auth_type: str
    logon_type: str
    auth_orientation: str
    success: int

class RequestBody(BaseModel):
    batch_id: str
    data: list[DataRow]

@router.post("/predict")
def predict(req: RequestBody):
    predictions = predict_batch(req)
    return {"batch_id": req.batch_id, "predictions": predictions}

# @router.post("/retrain")
# async def retrain(request: Request):
#     body = await request.json()

#     # Drop the first item if it's the data_size object
#     if isinstance(body, list) and "data_size" in body[0]:
#         body = body[1:]

#     if not body:
#         return {"message": "No data received for retraining."}

#     # Prepare training data: features + labels
#     labels    = [int(r["anomaly"]) for r in body]
#     base_feats = preprocess_records(body)          # (B, F)
#     seq_len    = 10
#     features   = base_feats.unsqueeze(1).repeat(1, seq_len, 1)  # (B,10,F)
#     flat_scores = []

#     with torch.no_grad():
#         for i in range(len(body)):
#             row_tensor = features[i].unsqueeze(0)  # [1, 10, F]
#             g = GRU_MODEL(row_tensor).item()
#             l = LSTM_MODEL(row_tensor).item()
#             iso_score = 0.0  # placeholder or real
#             flat_scores.append([g, l, iso_score])

#     # Fine-tune the MLP
#     X = torch.tensor(flat_scores, dtype=torch.float32)
#     y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

#     loss_fn = torch.nn.BCELoss()
#     optimizer = torch.optim.Adam(MLP_HEAD.parameters(), lr=0.01)

#     MLP_HEAD.train()
#     for epoch in range(10):
#         optimizer.zero_grad()
#         outputs = MLP_HEAD(X)
#         loss = loss_fn(outputs, y)
#         loss.backward()
#         optimizer.step()

#     MLP_HEAD.eval()
#     torch.save(MLP_HEAD.state_dict(), "models/mlp_weights.pth")
#     return {"message": "MLP retrained successfully.", "samples": len(body)}

@router.post("/retrain")
async def retrain_gru(request: Request):
    """
    Fine-tune the *head* (fc layer) of GRU_MODEL on labelled rows.
    Body = list of rows, each row must include `anomaly` (0/1).
    """
    rows = await request.json()
    if not rows:
        return {"error": "no data"}

    def _to_int_bool(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return 1 if v else 0
        # string fallback
        return 1 if str(v).strip().lower() in {"1", "true", "yes"} else 0

    labels = torch.tensor(
        [_to_int_bool(r.get("anomaly", 0)) for r in rows],
        dtype=torch.float32
    )  # shape (B,)
    base   = preprocess_records(rows)               # (B, F)
    seq    = base.unsqueeze(1).repeat(1, 10, 1)      # (B,10,F)

    # Forward once to build computation graph only for .fc params
    GRU_MODEL.train()
    optim = torch.optim.Adam(GRU_MODEL.fc.parameters(), lr=1e-3)
    lossf = torch.nn.BCELoss()

    for _ in range(10 if len(rows) < 100 else 25):
        optim.zero_grad()
        preds = GRU_MODEL(seq).squeeze()         # (B,)
        loss  = lossf(preds, labels)
        loss.backward()
        optim.step()

    epochs = 10 if len(rows) < 100 else 25
    for _ in range(epochs):
        optim.zero_grad()
        preds = GRU_MODEL(seq).squeeze()
        loss  = lossf(preds, labels)
        loss.backward()
        optim.step()

    GRU_MODEL.eval()
    # Persist just in case you restart docker / reload later
    torch.save(GRU_MODEL.state_dict(), "models/gru_trained_model.pth")

    return {
        "msg": "GRU head fine-tuned",
        "samples": len(rows),
        "loss": round(loss.item(), 4)
    }