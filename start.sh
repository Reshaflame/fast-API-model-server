#!/bin/bash

# Wait for proxy hostname to become reachable
hostname_val=$(hostname)
proxy_url="https://${hostname_val}-80.proxy.runpod.net"

echo "⏳ Waiting for proxy URL to respond: $proxy_url"

# Retry loop (max 10 tries with 3 seconds delay)
for i in {1..10}; do
    if curl -s --head "$proxy_url" | grep "200 OK" > /dev/null; then
        export RUNPOD_PROXY_URL="$proxy_url"
        echo "🌐 Proxy URL is live: $RUNPOD_PROXY_URL"
        break
    else
        echo "🔁 Attempt $i: Proxy not ready, retrying..."
        sleep 3
    fi
done

# Fallback if still not available
if [ -z "$RUNPOD_PROXY_URL" ]; then
    export RUNPOD_PROXY_URL="http://127.0.0.1"
    echo "⚠️ Proxy still not live — falling back to: $RUNPOD_PROXY_URL"
fi

# Start FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port 80
