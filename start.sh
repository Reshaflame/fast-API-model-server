#!/bin/bash
export RUNPOD_PROXY_URL="https://$(hostname)-80.proxy.runpod.net"
echo "🌐 Using proxy URL: $RUNPOD_PROXY_URL"

# Launch the server
exec uvicorn app.main:app --host 0.0.0.0 --port 80
