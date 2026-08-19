#!/usr/bin/env bash
# day1-click-demo — environment preparation. Normally you do NOT run this directly:
# `bash run.sh` calls it automatically on first run.
#
#   DAL_PATH=/path/to/DAL bash setup.sh   # DAL elsewhere (default: ../DAL sibling)
#   SKIP_DOCTOR=1 / SKIP_SELFTEST=1       # skip the verification gates
#   ALLOW_DAL_DRIFT=1                     # tolerate a DAL checkout off the pin
set -euo pipefail

DAL_REF="52af03b4a2586fe633c61a57cc5932b55474533d"   # embed() 포함 (2026-08-19)
DAL_URL="git@github.com:deeplyinc/DAL.git"
MODEL_VER="v4_e20"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
DAL_DIR="${DAL_PATH:-$(dirname "$HERE")/DAL}"
VENV="$HERE/.venv"

B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
step(){ echo; echo "${B}[$1/6] $2${N}"; }
ok(){ echo "      ${G}OK${N}   $1"; }
warn(){ echo "      ${Y}WARN${N} $1"; }
die(){ echo "      ${R}FAIL${N} $1"; exit 1; }

echo "${B}day1-click-demo setup${N}   (DAL: $DAL_DIR @ ${DAL_REF:0:7})"

step 1 "System tools"
command -v git >/dev/null || die "git not found"
git lfs version >/dev/null 2>&1 || die "git-lfs not found (macOS: brew install git-lfs / Ubuntu: apt install git-lfs)"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)' \
  || die "python >= 3.10 required"
ok "git / git-lfs / $("$PY" -V)"
command -v ffmpeg >/dev/null && ok "ffmpeg $(ffmpeg -version | head -1 | awk '{print $3}')" \
  || warn "ffmpeg missing - needed for file playback/rehearsal mode (brew/apt install ffmpeg)"

step 2 "DAL engine (../DAL)"
N_FOLDS=5                       # v4_e20 앙상블 크기 — 이보다 적으면 검증 성능이 아니다
if [ -d "$DAL_DIR/.git" ]; then
  CUR="$(git -C "$DAL_DIR" rev-parse HEAD)"
  if [ "$CUR" = "$DAL_REF" ]; then
    ok "existing checkout at pinned ${DAL_REF:0:7}"
  elif [ "${ALLOW_DAL_DRIFT:-0}" = "1" ]; then
    warn "checkout ${CUR:0:7} != pinned ${DAL_REF:0:7} (ALLOW_DAL_DRIFT=1 - dev mode)"
  else
    die "DAL is at ${CUR:0:7}, pinned is ${DAL_REF:0:7}. Checkout the pin, or ALLOW_DAL_DRIFT=1 for dev."
  fi
  [ -z "$(git -C "$DAL_DIR" status --porcelain)" ] \
    || warn "DAL working tree is dirty - demo will run uncommitted engine code"
else
  echo "      cloning $DAL_URL ..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$DAL_URL" "$DAL_DIR" \
    || die "clone failed (SSH access to private repo?)"
  git -C "$DAL_DIR" checkout --quiet "$DAL_REF"
  ok "cloned at ${DAL_REF:0:7} (LFS smudge skipped - weights pulled below)"
fi
W="$DAL_DIR/DAL/inference/hdtransys/weights/$MODEL_VER"
have_folds() { find "$W" -name 'fold*.pt' -size +1M 2>/dev/null | wc -l | tr -d ' '; }
if [ "$(have_folds)" -eq "$N_FOLDS" ]; then
  ok "weights present ($N_FOLDS folds)"
else
  echo "      pulling weights (~172 MB, LFS) ..."
  git -C "$DAL_DIR" lfs install --local >/dev/null 2>&1 || true
  git -C "$DAL_DIR" lfs pull --include="DAL/inference/hdtransys/weights/$MODEL_VER/*" \
    || die "git lfs pull failed (fallback: DAL/inference/hdtransys/weights_sync.sh pull)"
  [ "$(have_folds)" -eq "$N_FOLDS" ] \
    || die "expected $N_FOLDS fold .pt files, found $(have_folds) - LFS pull incomplete"
  ok "weights pulled ($N_FOLDS folds)"
fi

step 3 "Virtual environment"
[ -d "$VENV" ] && ok "reusing $VENV" || { "$PY" -m venv "$VENV"; ok "created $VENV"; }
VPY="$VENV/bin/python"; [ -x "$VPY" ] || VPY="$VENV/Scripts/python.exe"
"$VPY" -m pip install --quiet --upgrade pip

step 4 "Dependencies (torch is the slow part)"
"$VPY" -m pip install --quiet -r "$DAL_DIR/DAL/inference/hdtransys/requirements.txt" \
  || die "engine requirements failed (CPU-only machine: see note inside that file)"
"$VPY" -m pip install --quiet -r "$HERE/requirements.txt"
"$VPY" -m pip install --quiet -e "$DAL_DIR"
mkdir -p "$HERE/.lock" && "$VPY" -m pip freeze > "$HERE/.lock/requirements.lock.txt"
ok "installed (lock: .lock/requirements.lock.txt)"

step 5 "Diagnosis"
if [ "${SKIP_DOCTOR:-0}" = "1" ]; then
  warn "skipped (SKIP_DOCTOR=1)"
elif ! DAL_PATH="$DAL_DIR" "$VPY" "$HERE/doctor.py"; then
  echo
  die "doctor reported blocking problems - setup is NOT demo-ready (SKIP_DOCTOR=1 to bypass)"
fi

step 6 "Selftest (golden end-to-end)"
if [ "${SKIP_SELFTEST:-0}" = "1" ]; then
  warn "skipped (SKIP_SELFTEST=1)"
else
  DAL_PATH="$DAL_DIR" "$VPY" "$HERE/run.py" --selftest \
    || die "selftest failed - engine/golden mismatch, NOT demo-ready (SKIP_SELFTEST=1 to bypass)"
  ok "golden pair matched - demo-ready"
fi

echo
echo "${B}Environment ready.${N}"
echo "      Rehearsal (no connector needed):   bash run.sh --source file --loop"
