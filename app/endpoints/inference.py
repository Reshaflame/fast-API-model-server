from fastapi import APIRouter
from pydantic import BaseModel
from app.services.predictor import predict_batch

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
