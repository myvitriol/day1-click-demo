"""데모 전역 config — 경로·파라미터는 전부 이 파일 한 곳에서."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# DAL 배치: 형제 폴더(../DAL)가 기본. 환경변수 DAL_PATH 로 우회 가능.
import os
DAL_DIR = Path(os.environ.get("DAL_PATH", REPO.parent / "DAL"))
GOLDEN_FLAC = DAL_DIR / "DAL/inference/hdtransys/fixtures/golden_1cycle.flac"
GOLDEN_EXPECTED = DAL_DIR / "DAL/inference/hdtransys/fixtures/golden_expected.json"
WEIGHTS_DIR = DAL_DIR / "DAL/inference/hdtransys/weights"

# ── 오디오 ──────────────────────────────────────────────────────
SR = 96000                    # 모델 학습 샘플레이트. 다르면 시작을 막는다(리샘플 금지).
WIN_S, HOP_S = 1.0, 0.2       # manifest 검증값과 일치해야 함
WIN_N, HOP_N = int(SR * WIN_S), int(SR * HOP_S)
RING_S = 20                   # ring buffer 길이(초)
FILE_BLOCK = 4800             # file-source 쓰기 블록(50ms). mic 은 blocksize=0(가변).
DEVICE_HINT = "Scarlett"      # 입력 장치 이름 힌트(부분 일치). 없으면 기본 입력.

# ── 파이프라인 ──────────────────────────────────────────────────
QUEUE_MAX = 64                # window 큐. never-drop — 가득이 지속되면 trial 무효(fail-close).
MAX_LAG_S = 5.0               # 추론 지연 상한. 초과 시 그 듣기를 버리고 복구.
SNAP_PRE_S, SNAP_POST_S = 0.25, 0.75   # freeze snapshot 구간(이벤트 t 기준)
AMBIENT_EMBED_PERIOD = 1.0    # IDLE 지도 ambient 점 주기(초)

# ── 엔진 ────────────────────────────────────────────────────────
PAIR_POSTP = False            # False = immediate-only (pair 후처리 off). 데모는 첫 c1 에서
                              # freeze 하므로 pair 기계는 장식이었다(사용자 결정 2026-08-19).
                              # 주의: FA 는 pair 보다 높다(검증치 fa7 vs 3) — 판정 기준(엄격)으로 보완.
                              # selftest 는 golden pair 대조를 위해 항상 pair 모드로 강제한다.
FOLDS = None                  # None=벤치로 자동(5→2→1). 정수면 고정.
BENCH_BUDGET_MS = 120         # p95 가 이걸 넘으면 fold 축소 (hop 200ms 의 60%)
# ── 연속 클릭 카운터 (app/counter.py) ───────────────────────────
#   엔진은 순수 window 분류기로 쓰고 이벤트 집계는 여기서 한다(cycle 개념 없음).
#   실측(golden 2/2, sim_dense 24/24 FA0): 문턱에 예민하지 않다. 현장에서 재조정.
TAU_ON, TAU_OFF, MIN_ON = 0.80, 0.50, 2
LEVEL_TAU_ON = {-1: 0.70, 0: 0.80, 1: 0.90}    # 판정 기준 3단 → tau_on

LEVELS = (-1, 0, 1)           # 판정 기준 3단 = manifest strictness_map
LEVEL_NAMES = {-1: "느슨", 0: "보통", 1: "엄격"}
DEFAULT_LEVEL = 0

# ── 서버 ────────────────────────────────────────────────────────
HOST, PORT = "127.0.0.1", 8765
STREAM_HZ = 10                # 화면 스트림 프레임 주기
ATLAS_PATH = REPO / "atlas.json"

# ── 표시용 STFT(모델 mel 과 무관, 화면 전용) ────────────────────
DISP_NFFT, DISP_HOP, DISP_BINS = 2048, 960, 96   # 100 cols/s, 96 log-spaced bins
DISP_DB_LO, DISP_DB_HI = -12.0, 20.0  # 정상 성분 제거 후 표시 범위(dB).
#   실측(golden, 안정 floor): 조용 p99=+12, 기계가동 p99=+19, >+20 은 0.7% 미만.
