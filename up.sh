#!/usr/bin/env bash
# Builds (if needed) and starts the full stack.
#
# Usage:
#   ./up.sh            build + start everything, follow logs
#   ./up.sh --no-build  skip the build step (use existing images)
#   ./up.sh --detach     start without following logs
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found."
    echo "       Copy .env.example to .env and fill in POSTGRES_PASSWORD"
    echo "       and API_KEY first:"
    echo
    echo "         cp .env.example .env"
    echo "         python3 -c \"import secrets; print(secrets.token_urlsafe(32))\"  # for API_KEY"
    echo "         openssl rand -base64 24                                          # for POSTGRES_PASSWORD"
    echo
    exit 1
fi

if [[ ! -f config/cameras.yaml ]]; then
    echo "ERROR: config/cameras.yaml not found."
    echo "       Start from one of the templates:"
    echo
    echo "         cp config/cameras.example.yaml config/cameras.yaml              # real RTSP cameras"
    echo "         cp config/cameras.webcam.example.yaml config/cameras.yaml       # local webcam"
    echo "         cp config/cameras.testvideos.example.yaml config/cameras.yaml   # demo video files"
    echo
    exit 1
fi

if [[ "${1:-}" != "--no-build" ]]; then
    ./build.sh
fi

if [[ ! -f config/model/custom_model.onnx ]]; then
    echo
    echo "WARNING: config/model/custom_model.onnx not found."
    echo "         The inference service will fail to start until you run"
    echo "         the training/ pipeline (see training/README.md) or drop"
    echo "         a compatible ONNX model + labelmap into config/model/."
    echo
fi

docker compose up -d

echo "==> waiting for API health check"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
        echo "==> API is up"
        break
    fi
    sleep 2
done

echo
echo "Frontend:  http://localhost:8080/  (will prompt for the API key from .env on first load)"
echo "API:       http://localhost:8000/api/cameras  (requires: -H \"X-API-Key: \$API_KEY\")"
echo "Logs:      docker compose logs -f"
echo

if [[ "${1:-}" != "--detach" ]]; then
    docker compose logs -f
fi
