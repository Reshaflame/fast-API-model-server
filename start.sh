#!/bin/bash

# Dynamically export the real proxy URL based on the pod's hostname
export RUNPOD_PROXY_URL="https://$(hostname)-80.proxy.runpod.net"

# Launch the app
uvicorn app.main:app --host 0.0.0.0 --port 80
