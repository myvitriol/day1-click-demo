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
#   문턱 정하기 — 우리 자료로는 정할 수 없다는 것이 실측 결론이다(2026-08-19).
#
#   이 모델은 clean 자료에서 클릭과 배경을 **극단적으로** 갈라놓는다:
#     · 배경 window 의 p_click 중앙값 0.0000 (p75 도 0.0000)
#     · 진짜 클릭 window 는 1.0000 부근, 클릭 1건 = 5 window 연속(1s win / 0.2s hop)
#   그래서 sim_dense(진짜 48건) 전수 스윕에서 **tau_on 0.30 ~ 0.995, min_on 1~5 어느
#   조합에서도 결과가 똑같이 48/48** 이었다 — 놓침도 FA 도 0. 즉 우리가 가진 96kHz
#   자료(golden 하나에서 파생)로는 문턱의 좋고 나쁨을 구분할 수 없다.
#
#   그러므로 값은 **현장 피드백으로만** 움직인다. 지금까지의 경과:
#     0.80 기본 → FA 가 많다는 보고 → 0.93 → 그래도 많다고 읽고 0.97 로 올림
#     → 실제로는 **안 잡히는** 쪽이 문제였다(내가 방향을 반대로 읽었다) → 아래로 내림
#   현장 클릭이 golden 보다 훨씬 약하다는 뜻이므로, 느슨을 0.35 까지 내려 여유를 크게
#   준다. clean 에서 손실이 없음은 위 스윕으로 확인됐다.
#
#   MIN_ON 은 1 로 내린다. window 1s / hop 0.2s = 80% 겹침이라 연속 window 는 독립
#   증거가 아니어서 FA 를 거의 못 막는 반면(짧은 잡음도 그냥 5 window 에 들어온다),
#   현장 클릭이 약해 문턱 위 window 가 1개뿐인 경우를 놓치게 만든다.
#   대가: 이벤트 시각이 0.2s 앞당겨진다(golden 실측 7.7 → 7.5s, 실제 클릭은 8.1s).
#   즉 정지 화면 구간 [t-0.25, t+0.75] 안에서 클릭이 더 오른쪽(t+0.6)에 온다 — 아직
#   구간 안이지만 여유가 0.15s 뿐이다. 화면에서 클릭이 오른쪽에 치우쳐 보이면
#   SNAP_PRE_S/SNAP_POST_S 를 옮겨야 한다(그때 cycle.py 의 CAM shift 도 함께).
#   "한 클래스가 최소 얼마"라는 2차 조건은 넣지 않는다 — 확률이 c1/c2 로 쪼개진 soft
#   click 을 버리게 되고, 판정량을 1-p_others 로 고른 이유가 바로 그 보호다.
#
#   FA 가 다시 문제가 되면 화면 dial 로 엄격(0.80)까지 올릴 수 있다. 그걸로도 안 되면
#   문턱 문제가 아니다 — 검출 시점의 점수를 로그로 남겨서 봐야 한다.
TAU_ON, TAU_OFF, MIN_ON = 0.55, 0.20, 1
#   TAU_OFF 는 tau_on 보다 넉넉히 낮게. 붙여 두면 그 사이를 진동하는 신호가 release →
#   재commit 을 반복해 한 소리가 여러 번 세진다.
LEVEL_TAU_ON = {-1: 0.35, 0: 0.55, 1: 0.80}    # 판정 기준 3단 → tau_on

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
