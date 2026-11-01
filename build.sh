#!/usr/bin/env bash
# Builds the shared base image, then every service image on top of it.
# Run this (or ./up.sh, which calls it) any time common/, requirements.txt,
# or a service's source changes.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> building nvr-base:latest"
docker build -t nvr-base:latest -f Dockerfile.base .

echo "==> building service images"
docker compose build

echo "==> build complete"
