from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.services.predictor import predict_batch, GRU_MODEL, ISO_MODEL, HEAD
from app.utils.preprocess_utils import preprocess_batch
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

@router.post("/retrain")
async def retrain(request: Request):
    body = await request.json()

    # Drop the first item if it's the data_size object
    if isinstance(body, list) and "data_size" in body[0]:
        body = body[1:]

    if not body:
        return {"message": "No data received for retraining."}

    # Prepare training data: features + labels
    labels = [int(row["anomaly"]) for row in body]
    features = preprocess_batch(body)  # shape: [B, 10, F]
    flat_scores = []

    with torch.no_grad():
        for i in range(len(body)):
            row_t = features[i].unsqueeze(0)      # [1, 1, F] in alpha
            g = GRU_MODEL(row_t).item()
            iso = ISO_MODEL.decision_function(features[i, -1, :].unsqueeze(0).numpy())[0]
            flat_scores.append([g, iso])

    # Fine-tune DeepHead
    X = torch.tensor(flat_scores, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(HEAD.parameters(), lr=0.01, weight_decay=1e-4)

    HEAD.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = HEAD(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"[DeepHead] Epoch {epoch+1:02d}: Loss = {loss.item():.6f}")

    HEAD.eval()
    with torch.no_grad():
        test_preds = HEAD(X).squeeze().tolist()
        print("📈 DeepHead raw probs:", test_preds)
    torch.save(HEAD.state_dict(), "models/mlp_weights.pth")

    # 🔄 Load the freshly saved weights
    HEAD.load_state_dict(torch.load("models/mlp_weights.pth"))
    print("✅ Reloaded DeepHead weights into memory.")

    return {"message": "MLP retrained successfully.", "samples": len(body)}