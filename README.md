# day1-click-demo

커넥터 체결음(click) 실시간 검출 **현장 영업 데모**. 이름의 `day1` = 데이터 수집 계약 전
**첫날 현장에서 바로 가치를 보여준다**는 뜻이다.

노트북 한 대에서 전부 돈다 — Python 이 마이크(96 kHz)와 모델을 독점하고, 브라우저는
화면만 그린다. 인증서·VPN·사내 서버가 필요 없다.

> ### ⚠️ 지금 모델은 **대체(임시) 모델**이다
> 현재 붙어 있는 것은 다른 공정용으로 학습된 **DAL `hdtransys` v4_e20** 이다 — 배선·화면·
> 집계 로직을 먼저 완성하려고 **자리를 채워 둔 것**이고, 이 데모의 최종 모델이 아니다.
> **day1 데모 전용 모델이 따로 만들어질 예정**이며, 준비되면 이 자리를 교체한다.
>
> 그래서 지금 보이는 성능 수치·판정 문턱·클래스 이름(`click_1`/`click_2`)은 **대체 모델의
> 성질**이다. 교체 시 바뀔 것과 안 바뀔 것:
>
> | 바뀜 | 안 바뀜 |
> |---|---|
> | weights·클래스 정의·확률 분포 | 화면·조작(play/stop·감지 ON/OFF·리셋) |
> | 판정 문턱 기본값(`TAU_ON` 등) | 연속 카운터 방식(hysteresis 상승 edge) |
> | 입력 규격 가능성(96 kHz·ch0) | 캡처·집계·표시 파이프라인 구조 |
>
> 교체 지점은 `app/engine.py` 한 곳이다(엔진을 순수 window 분류기로만 쓰기 때문).

---

## 시작하기 (처음 쓰는 노트북)

**1. 두 repo 를 같은 폴더에 나란히 clone**

```bash
cd ~/work                                          # 아무 작업 폴더
git clone git@github.com:deeplyinc/DAL.git
git clone git@github.com:myvitriol/day1-click-demo.git
```

형제 배치가 전제다 — `day1-click-demo` 가 `../DAL` 을 찾는다:

```
~/work/
├── DAL/                # 대체 모델 엔진 (private repo, pinned commit)
└── day1-click-demo/    # 이 repo
```

**2. 실행 — 이 한 줄이 전부다**

```bash
cd day1-click-demo
bash run.sh
```

첫 실행이면 `run.sh` 가 알아서 준비한다(5~15분): 가상환경 생성 → 의존성 설치 →
DAL pin 확인 + 모델 weights(LFS, ~172MB) 받기 → 환경 진단 → golden 검증 → **앱 실행 +
브라우저 자동 열림**. 두 번째 실행부터는 준비를 건너뛰고 바로 뜬다.

**3. 브라우저에서 마이크 선택**

화면이 열리면 좌상단 드롭다운이 분홍으로 깜빡인다:

```
[▼ 마이크를 선택하세요]        배지: 입력 대기 — 위에서 마이크를 선택하세요
```

Scarlett 등 외장 인터페이스를 고르면 96 kHz 로 열고 **바로 감지가 시작된다**
(이름에 `★` 가 붙은 항목이 권장 장치). 클릭이 잡히면 화면이 자동으로 멈추고,
`재생` 을 누르면 이어서 듣는다.

> 마이크가 96 kHz 로 안 열려도 서버는 죽지 않는다 — 화면이 열린 상태로 대기하고
> 다른 장치를 고를 수 있다. 96 kHz 가 아니면 **시작을 거부**한다(리샘플로 때우지 않는다 —
> 모델이 96 kHz 로 학습됐고, 48 kHz 를 올려도 학습 대역은 돌아오지 않는다).

### macOS 준비물 (1회)

```bash
brew install git-lfs ffmpeg
```

마이크 권한은 **Chrome 이 아니라 `run.sh` 를 실행한 터미널**에 필요하다
(시스템 설정 → 개인정보 보호 → 마이크에서 터미널/iTerm 허용).

---

## 그 밖의 실행 방법

```bash
bash run.sh --source file --file sim/sim_dense.flac --loop   # 리허설 (마이크·커넥터 불필요)
bash run.sh --selftest                                       # 검증만 (headless)
bash run.sh --postp pair                                     # 예전 pair 후처리로 되돌리기
.venv/bin/python doctor.py                                   # 환경·오디오·엔진 진단
.venv/bin/python tools/make_sim_audio.py --dense             # 리허설용 오디오 생성 (repo 미포함)
```

`setup.sh` 는 `run.sh` 가 자동으로 부르는 준비 단계다 — 직접 실행할 필요 없다.

---

## 구조

```
run.py            # 진입점  (--postp off|pair, --source mic|file, --file, --loop, --selftest)
app/config.py     # 모든 경로·파라미터 (한 파일)
app/audio.py      # 캡처(96k 강제)·RingBuffer·window slicer (절대 sample index)
app/engine.py     # DAL 래퍼 — 3단(-1/0/+1) 변형을 weight 공유로 보유
app/cycle.py      # 연속 inference + play/stop, freeze snapshot, fail-close
app/atlas.py      # 소리의 지도 좌표계 (atlas.json 고정 or 세션 폴백)
web/index.html    # 화면 (표시 전용 WebSocket 클라이언트)
web/assets/       # DEEPLY CI 원본 (무수정 — 여백은 CSS 로만 처리)
PLAN.md · mockups/  # 기획서 · 화면 설계 이력 (v01~v1.0)
tools/build_atlas.py  # 참조 wav → atlas.json (PCA-2 고정)
tools/make_sim_audio.py # 긴 시뮬 오디오 생성 (--dense = 클릭 자주)
app/xai.py        # GradCAM (fold0, target backbone.out_c.0)
app/counter.py    # 연속 클릭 카운터 (hysteresis 상승 edge) ← 실제 판정기
doctor.py         # 설치·현장 진단
```

## 동작 모델 (2026-08-18 단순화)

시험 선언(체결/방해음) 없음 — **모델은 재생 중이면 항상 inference 모드**다.

- **play/stop 토글 하나**가 스트리밍을 제어한다. 정지는 서버가 그림 전송을 실제로 멈춘다.
- **검출 → 자동 정지.** 잡힌 1초를 해부해 보여주고, 재생을 눌러야만 다시 흐른다.
- 재개/판정 기준 변경 시 classifier 를 finalize(자동 reset) — DAL 의 cycle 경계 계약 준수.
- 상태 메시지는 전부 `rev`(전환 revision)를 실어 보내고 클라이언트는 낡은 rev 를 버린다
  — in-flight 역전으로 버튼이 "안 먹는" 상태를 만들 수 없다.

## 화면 기능 (2026-08-19 현재)

| | |
|---|---|
| play/stop | 스트리밍 토글. 정지 시 서버가 그림 전송을 멈춘다 |
| 감지 ON/OFF | 추론 자체를 쉰다(흐름만 표시). 잡담 구간용 |
| ↺ 리셋 | 오늘 잡은 클릭 · 지도 ★ · 마지막 검출 패널 함께 초기화 |
| 판정 기준 3단 | `--postp off`(기본)에서는 c1 threshold 0.7 / 0.8 / 0.9 |
| 마지막 검출 | 확신(모델) · SNR · 판정 기준 · 오늘 n번째 |
| freeze 화면 | 파형에 SNR 화살표, 스펙트로그램에 GradCAM 스포트라이트 |
| 소리의 지도 | quantile 매핑 + anchor-warp(무리 조임/벌림). 검출 누적 시 강도 상승 |
| 입력 장치 | mic 모드면 헤더가 드롭다운(★=이름 힌트 일치). 전환 시 스트림 경계 리셋 |

## 이벤트 집계 = 연속 카운터 (2026-08-19 확정)

**엔진은 순수 1초 window 분류기**(`postp=False, immediate=False`)로 쓰고, 이벤트 집계는
`app/counter.py` 가 한다. DAL 은 무수정.

- DAL 의 두 후처리는 **PLC cycle 전제**다(pair 는 c1→c2 짝, immediate 는 cycle당 1회 +
  finalize 로만 재무장). 이 데모는 "어떤 클릭이든 하나씩 계속 센다"가 요구라 둘 다 맞지 않는다.
- 방식: hysteresis(`tau_on`/`tau_off`) + `min_on` 연속 → **상승 edge 만 1건으로 카운트**.
  시간 refractory 없음 — release 가 신호 기반이라 더 정확하다.
- c1/c2 구분은 표시에만 쓰고 카운트는 클래스 무관. **c2 단독도 세진다**(pair 모드에선 안 됐다).
- 판정 기준 3단 = `tau_on` 0.70 / 0.80 / 0.90.
- **정지는 정지다** — 추론·카운트 모두 멈춘다. 재생하면 카운터 상태만 비우고(총계 유지) 이어 듣는다.

실측(**대체 모델 기준** — 전용 모델로 바꾸면 다시 재야 한다):
golden(클릭 2개) → **2건 정확**, sim_dense 120초(24개) → **24건 정확·FA 0**,
최소 간격 2.0s 분리. `tau_on` .7~.9 × `tau_off` .3~.5 × `min_on` 1~2 여섯 조합 전부 동일 개수.
한계: window 가 1초라 **1초 이내 연타는 원리적으로 분리 못 한다**.

## 후처리 모드 (중요 — 대체 모델 기준)

기본은 **`--postp off` = immediate-only**. 데모는 첫 c1 에서 화면을 멈추므로 pair 기계가
장식이었다(2026-08-19 사용자 결정). 대가와 대응:

- pair 대비 **FA 가 높다**(검증치 fa 7 vs 3). 현장에서 오검출이 잦으면 **판정 기준 `엄격`**,
  그래도 불안하면 `--postp pair` 로 즉시 되돌린다.
- `strictness` 3단은 이 모드에서 **c1 threshold(0.7/0.8/0.9)만** 움직인다 — 기존 pair
  strictness 성능표는 더 이상 근거가 아니다.
- `run.py --selftest` 는 golden pair 대조를 위해 pair 를 강제하고, **별도로 off 모드
  스모크**(첫 c1 슬롯)를 함께 검사한다.

## 운영 규칙 (설계 근거)

- **96 kHz 가 아니면 시작하지 않는다.** 리샘플로 조용히 때우지 않는다 (학습 대역 재현 불가).
- **fail-close.** ring overrun·추론 지연 초과·캡처 overflow → 그 듣기를 버리고 즉시 복구.
- 오늘 잡은 클릭 = **화면에 실제로 보여준 검출 수**(조기 c1 알림 포함, pair 확정 아님).
