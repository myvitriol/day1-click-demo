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
# 준비 완료로 착각하지 않기 위한 표식이다. 첫 줄에 그때의 커밋 해시를 적어둔다.
need_setup=0
CUR_REF="$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo nogit)"
[ -f "$HERE/.ready" ] || need_setup=1
[ -x "$VPY" ] || need_setup=1
if [ -x "$VPY" ]; then
  "$VPY" -c "import fastapi, uvicorn, torch, DAL" >/dev/null 2>&1 || need_setup=1
fi

# git pull 로 코드가 바뀌었으면 다시 돌린다 — 새 버전이 요구하는 의존성·검증이
# 있을 수 있다. 이미 갖춰진 것은 setup 이 건너뛰므로 대개 몇 분이면 끝난다.
if [ "$need_setup" = "0" ] && [ "$(head -1 "$HERE/.ready" 2>/dev/null)" != "$CUR_REF" ]; then
  echo "Code changed since the last setup - re-running setup for this version."
  need_setup=1
fi

if [ "$need_setup" = "1" ]; then
  echo "=================================================="
  if [ -f "$HERE/.ready" ]; then
    echo " Updating the environment for the current version"
  else
    echo " First run - preparing the environment (5-15 min)"
  fi
  echo "=================================================="
  LOG="$HERE/.setup.log"
  if bash "$HERE/setup.sh" 2>&1 | tee "$LOG"; then
    { echo "$CUR_REF"; date -u +"%Y-%m-%dT%H:%M:%SZ"; } > "$HERE/.ready"
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
