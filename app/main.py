from fastapi import FastAPI
from app.endpoints.inference import router as inference_router

app = FastAPI()
app.include_router(inference_router, prefix="/api")
