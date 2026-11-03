#!/usr/bin/env bash
# SPOILOCK AI 서버 EC2 최초 설치 (Ubuntu 24.04, 한 번만 실행)
#   전제: 코드가 /opt/spoilock-ai 에 git clone 되어 있음
#   sudo bash /opt/spoilock-ai/deploy/setup_ec2.sh
set -euo pipefail

APP_DIR=/opt/spoilock-ai
APP_USER=ubuntu

echo "[1/5] 패키지 설치"
apt-get update -y
apt-get install -y python3-venv python3-pip ffmpeg

echo "[2/5] 스왑 4GB"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "[3/5] 가상환경과 라이브러리 (torch 는 CPU 버전)"
cd "$APP_DIR"
sudo -u "$APP_USER" python3 -m venv .venv
sudo -u "$APP_USER" .venv/bin/pip install --upgrade pip
sudo -u "$APP_USER" .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
sudo -u "$APP_USER" .venv/bin/pip install -r deploy/requirements.txt
sha256sum deploy/requirements.txt > .venv/.requirements.sha

echo "[4/5] 임베딩 모델 미리 받기"
sudo -u "$APP_USER" .venv/bin/python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')"

echo "[5/5] 자동 실행 서비스 등록"
cp deploy/spoilock-ai.service /etc/systemd/system/spoilock-ai.service
systemctl daemon-reload
systemctl enable spoilock-ai

if [ ! -f "$APP_DIR/.env" ]; then
  echo "!! $APP_DIR/.env 가 없습니다. 복사한 뒤 'sudo bash deploy/deploy.sh' 를 실행하세요."
  exit 0
fi
bash "$APP_DIR/deploy/deploy.sh" --no-pull
