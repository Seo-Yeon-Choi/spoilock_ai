#!/usr/bin/env bash
# SPOILOCK AI 서버 배포 (GitHub Actions 가 SSM 으로 실행, 직접 실행도 가능)
#   sudo bash /opt/spoilock-ai/deploy/deploy.sh
#   .env · chroma_data · data 는 git 에 없으므로 건드리지 않음
set -euo pipefail

APP_DIR=/opt/spoilock-ai
APP_USER=ubuntu
cd "$APP_DIR"

if [ "${1:-}" != "--no-pull" ]; then
  echo "[deploy] 최신 코드 받기"
  sudo -u "$APP_USER" git fetch --quiet origin main
  sudo -u "$APP_USER" git reset --hard --quiet origin/main
fi
echo "[deploy] 커밋: $(sudo -u "$APP_USER" git log -1 --format='%h %s')"

# requirements 가 바뀐 경우에만 설치 (보통 몇 초)
if ! sha256sum --quiet -c .venv/.requirements.sha 2>/dev/null; then
  echo "[deploy] 라이브러리 변경 → 설치"
  sudo -u "$APP_USER" .venv/bin/pip install --quiet -r deploy/requirements.txt
  sha256sum deploy/requirements.txt > .venv/.requirements.sha
fi

# 서비스 파일이 바뀌었으면 반영
if ! cmp -s deploy/spoilock-ai.service /etc/systemd/system/spoilock-ai.service; then
  cp deploy/spoilock-ai.service /etc/systemd/system/spoilock-ai.service
  systemctl daemon-reload
fi

# Windows 에서 만든 .env 의 CRLF·BOM 제거
sed -i -e 's/\r$//' -e '1s/^\xEF\xBB\xBF//' .env
chown "$APP_USER" .env && chmod 600 .env

echo "[deploy] 재시작"
systemctl restart spoilock-ai

# 모델 로딩까지 최대 3분 대기
for i in $(seq 1 36); do
  sleep 5
  if curl -fsS http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ready":true'; then
    echo "[deploy] 완료: 서버 준비됨 ($((i * 5))초)"
    exit 0
  fi
done
echo "[deploy] 실패: 3분 안에 서버가 준비되지 않음. 최근 로그:"
journalctl -u spoilock-ai -n 40 --no-pager
exit 1
