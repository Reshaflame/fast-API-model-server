from fastapi import FastAPI
from app.endpoints.inference import router as inference_router
import os
import redis

app = FastAPI()
app.include_router(inference_router, prefix="/api")

# 🔐 Load Redis URL from secret
redis_url = os.getenv("REDIS_URL")
if not redis_url or not redis_url.startswith("redis://"):
    raise RuntimeError(f"❌ Invalid or missing Redis URL: {redis_url}")
redis_client = redis.Redis.from_url(redis_url)
print(f"🔑 Redis URL loaded successfully.")

# 🔁 Register proxy after app starts
@app.on_event("startup")
async def register_proxy_url():
    proxy_url = os.getenv("RUNPOD_PROXY_URL", "http://127.0.0.1")
    print(f"🌐 Detected RUNPOD_PROXY_URL: {proxy_url}")

    api_predict = f"{proxy_url}/api/predict"
    api_retrain = f"{proxy_url}/api/retrain"

    redis_client.set("TrueDetect:API_ENDPOINT", api_predict)
    redis_client.set("TrueDetect:RETRAIN_ENDPOINT", api_retrain)

    print(f"📡 Registered endpoints: {api_predict}, {api_retrain}")
