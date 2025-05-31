#!/bin/bash

# 🕓 Wait a moment to ensure proxy is ready
sleep 30

# 🌐 Export the proxy URL dynamically
export RUNPOD_PROXY_URL="https://$(hostname)-80.proxy.runpod.net"
echo "🌐 Set RUNPOD_PROXY_URL to $RUNPOD_PROXY_URL"

# 🚀 Start the app
exec uvicorn app.main:app --host 0.0.0.0 --port 80
#!/bin/bash
echo "🔍 Waiting for RUNPOD_PROXY_URL to be set..."
for i in {1..30}; do
  if [[ -n "$RUNPOD_PROXY_URL" ]]; then
    echo "✅ RUNPOD_PROXY_URL detected: $RUNPOD_PROXY_URL"
    break
  fi
  sleep 1
done

echo "🚀 Launching server..."
uvicorn app.main:app --host 0.0.0.0 --port 80
