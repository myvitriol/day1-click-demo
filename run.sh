#!/usr/bin/env bash
# day1-click-demo — 이거 하나만 실행하면 된다.
#
#   bash run.sh                       마이크로 데모 (브라우저 자동 열림)
#   bash run.sh --source file --file sim/sim_dense.flac --loop
#                                     리허설 (마이크 불필요)
#   bash run.sh --selftest            설치 검증만
#
# 첫 실행이면 가상환경·의존성·모델 weights 까지 알아서 준비한다(setup.sh 위임).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VPY="$HERE/.venv/bin/python"
[ -x "$VPY" ] || VPY="$HERE/.venv/Scripts/python.exe"

# .ready 는 setup 이 **검증까지 통과**했을 때만 생긴다. import 만 되는 반쪽 설치를
# 준비 완료로 착각하지 않기 위한 표식이다.
need_setup=0
[ -f "$HERE/.ready" ] || need_setup=1
[ -x "$VPY" ] || need_setup=1
if [ -x "$VPY" ]; then
  "$VPY" -c "import fastapi, uvicorn, torch, DAL" >/dev/null 2>&1 || need_setup=1
fi

if [ "$need_setup" = "1" ]; then
  echo "=================================================="
  echo " First run - preparing the environment (5-15 min)"
  echo "=================================================="
  LOG="$HERE/.setup.log"
  if bash "$HERE/setup.sh" 2>&1 | tee "$LOG"; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$HERE/.ready"
  else
    echo
    echo "Setup failed. Full log: $LOG"
    echo "Fix the reported problem and run 'bash run.sh' again."
    exit 1
  fi
  VPY="$HERE/.venv/bin/python"; [ -x "$VPY" ] || VPY="$HERE/.venv/Scripts/python.exe"
  echo
fi

exec "$VPY" "$HERE/run.py" "$@"
