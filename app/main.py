from fastapi import FastAPI
from app.endpoints.inference import router as inference_router
import os
import redis

app = FastAPI()
app.include_router(inference_router, prefix="/api")

# 🔐 Load Redis URL from environment and validate
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise RuntimeError("❌ Environment variable REDIS_URL is not set.")

redis_client = redis.Redis.from_url(redis_url)
print(f"🔑 Redis URL loaded successfully.")  # Avoid printing sensitive content
