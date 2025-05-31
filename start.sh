#!/bin/bash

# Just grab the proxy URL from hostname and assume it's correct
hostname_val=$(hostname)
export RUNPOD_PROXY_URL="https://${hostname_val}-80.proxy.runpod.net"

echo "🚀 Starting FastAPI server with proxy: $RUNPOD_PROXY_URL"

# Start FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port 80
