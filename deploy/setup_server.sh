#!/usr/bin/env bash
# Set up InspectorMCP on a fresh EC2 instance launched from the COOL Graviton4 AMI
# (AWS Marketplace "Cloud Optimized OpenCV For AWS Graviton4", Ubuntu 24.04 arm64).
# Run as the ubuntu user from the repo root:  bash deploy/setup_server.sh
set -euo pipefail

APP_DIR="$(pwd)"

# 1. Python env that uses COOL's OpenCV 5.1 build (not a pip wheel)
python3 -m venv .venv
cat >> .venv/bin/activate <<'EOF'

# COOL (OpenCV 5.1 for Graviton)
export PYTHONPATH="/opt/cool/python_3.12/site-packages/cv2/python-3.12:$PYTHONPATH"
export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib:$LD_LIBRARY_PATH"
EOF
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock
python -c "import cv2; print('OpenCV', cv2.__version__, cv2.__file__)"

# 2. Public ELPV dataset (CC BY-NC-SA 4.0) for demo panels and evaluation
[ -d "$HOME/elpv-dataset" ] || git clone --depth 1 https://github.com/zae-bayern/elpv-dataset.git "$HOME/elpv-dataset"

# 3. Optional: Gemini key for the "Rules + Gemini" brain (kept out of git)
[ -f .env ] || echo "# GEMINI_API_KEY=..." > .env
chmod 600 .env

# 4. Run as a service on :8080 (web console + /mcp)
sed "s#/home/ubuntu/inspector#${APP_DIR}#g" deploy/inspector.service | sudo tee /etc/systemd/system/inspector.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now inspector
sleep 3
curl -fsS localhost:8080/health && echo " - InspectorMCP is up on :8080"
