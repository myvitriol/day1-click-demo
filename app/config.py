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
#   주의: 아래 규격은 **대체(임시) 모델**(DAL hdtransys v4_e20)의 것이다.
#   day1 전용 모델로 교체하면 SR/WIN_S/HOP_S·문턱을 그 모델 기준으로 다시 잡는다.
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
AMBIENT_EMBED_PERIOD = 2.0    # 지도 ambient 점 주기(초). 1.0→2.0 (30분 세션 embed 절반)

# ── 엔진 ────────────────────────────────────────────────────────
PAIR_POSTP = False            # False = immediate-only (pair 후처리 off). 데모는 첫 c1 에서
                              # freeze 하므로 pair 기계는 장식이었다(사용자 결정 2026-08-19).
                              # 주의: FA 는 pair 보다 높다(검증치 fa7 vs 3) — 판정 기준(엄격)으로 보완.
                              # selftest 는 golden pair 대조를 위해 항상 pair 모드로 강제한다.
FOLDS = None                  # None=벤치로 자동(5→2→1). 정수면 고정.
BENCH_BUDGET_MS = 120         # p95 가 이걸 넘으면 fold 축소 (hop 200ms 의 60%)
# ── 연속 클릭 카운터 (app/counter.py) ───────────────────────────
#   엔진은 순수 window 분류기로 쓰고 이벤트 집계는 여기서 한다(cycle 개념 없음).
#
#   문턱 상향 근거(2026-08-19). 현장 SWE 가 실제 마이크에서 FA 가 너무 많다고 보고했다.
#   그 환경 녹음이 없어 FA 는 재현할 수 없다 — 우리 96kHz 자료는 golden 하나에서 파생된
#   조용한 실험실 배경뿐이라 구조적으로 FA 가 0이다. 그래서 잴 수 있는 것만 쟀다:
#   **문턱을 올릴 때 진짜 클릭을 언제부터 놓치는가**(sim_dense 48건 전수 스윕).
#     · 클릭 1건 = 정확히 5 window 연속(1s window / 0.2s hop) — 0.80~0.99 어디서나 5
#     · tau_on 0.800→0.995, min_on 2~5 전 조합에서 48/48 유지(손실 0)
#     · 진짜 클릭 window 의 top-class 확률 min 0.959 / median 1.000
#   단 이 48건은 같은 golden 에서 잘라낸 것이라 **독립 표본이 아니다**. "clean 에서
#   0.99 가 무손실"이 "현장에서 0.99 가 안전"을 뜻하지 않는다(현장 soft click 은 더
#   약할 수 있다). 그래서 saturation(0.99) 까지 가지 않고 0.93 에 둔다 — 종전 0.80 대비
#   유의미하게 올리면서 clean 최저값 0.959 에 여유를 남긴다.
#
#   MIN_ON 은 올리지 않는다(2 유지). window 1s / hop 0.2s = 80% 겹침이라 연속 window 는
#   독립 증거가 아니고, 짧은 잡음도 그냥 5 window 에 들어온다 — duration gate 가 아니다.
#   올리면 이벤트 시각만 뒤로 밀려(2→5 는 +0.6s) snapshot 이 실제 transient 를 놓친다.
#   "한 클래스가 최소 얼마"라는 2차 조건도 넣지 않는다: tau_on 0.93 이면 max(c1,c2) >=
#   0.465 가 이미 자동이고, 그보다 높은 floor 는 확률이 쪼개진 soft click 을 버린다
#   (0.50/0.43 같은 것) — 판정량을 1-p_others 로 고른 이유가 바로 그 보호다.
TAU_ON, TAU_OFF, MIN_ON = 0.93, 0.50, 2
#   TAU_OFF 는 낮게 둔다. 올리면 잡음이 더 쉽게 release 되어 같은 잡음이 반복 카운트된다.
LEVEL_TAU_ON = {-1: 0.90, 0: 0.93, 1: 0.97}    # 판정 기준 3단 → tau_on

LEVELS = (-1, 0, 1)           # 판정 기준 3단 = manifest strictness_map
LEVEL_NAMES = {-1: "느슨", 0: "보통", 1: "엄격"}
DEFAULT_LEVEL = 0

# ── 서버 ────────────────────────────────────────────────────────
HOST, PORT = "127.0.0.1", 8765
STREAM_HZ = 10                # 화면 스트림(파형·status) 주기
ATLAS_PATH = REPO / "atlas.json"

# ── 표시용 STFT(모델 mel 과 무관, 화면 전용) ────────────────────
DISP_NFFT, DISP_HOP, DISP_BINS = 2048, 960, 96   # 96 log-spaced bins
#   DISP_HOP 이 파형·스펙트로그램 공통 시간 격자다: 960/96k = 10ms → 둘 다 100 프레임/s.
#   웹이 1프레임=1px 로 그리므로 두 화면의 흐름 속도가 정확히 같다.
SNAP_HOP = 240                # 정지 화면(1초 확대) 전용 hop = 2.5ms → 392 컬럼.
#   1회성 비용(batched 아니어도 ~9ms)이라 실시간보다 촘촘하게 떠도 된다. NFFT 는 절대
#   바꾸지 않는다 — spec_floor(행별 dB)를 실시간과 공유하는데 FFT magnitude 가 window
#   길이에 따라 이동해 정지 화면 밝기만 틀어진다(Codex P1).
DISP_DB_LO, DISP_DB_HI = -12.0, 20.0  # 정상 성분 제거 후 표시 범위(dB).
#   실측(golden, 안정 floor): 조용 p99=+12, 기계가동 p99=+19, >+20 은 0.7% 미만.
