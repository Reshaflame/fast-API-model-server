from fastapi import FastAPI
from app.endpoints.inference import router as inference_router
import os
import redis
import requests


app = FastAPI()
app.include_router(inference_router, prefix="/api")

# 🔐 Load Redis URL from environment and validate
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise RuntimeError("❌ Environment variable REDIS_URL is not set.")

redis_client = redis.Redis.from_url(redis_url)
print(f"🔑 Redis URL loaded successfully.")  # Avoid printing sensitive content

# app/main.py (after redis_client is initialized)
REDIS_API_KEY = "TrueDetect:API_ENDPOINT"
REDIS_RETRAIN_KEY = "TrueDetect:RETRAIN_ENDPOINT"

try:
    ip = requests.get("https://api.ipify.org").text.strip()
    api_ip = f"http://{ip}"
except Exception as e:
    print(f"❌ Failed to get public IP: {e}")
    api_ip = "http://127.0.0.1"
api_predict = f"{api_ip}/api/predict"
api_retrain = f"{api_ip}/api/retrain"
redis_client.set(REDIS_API_KEY, api_predict)
redis_client.set(REDIS_RETRAIN_KEY, api_retrain)


print(f"📡 Registered endpoints: {api_predict}, {api_retrain}")
