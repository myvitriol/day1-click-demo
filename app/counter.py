"""연속 클릭 카운터 — cycle 개념 없음.

DAL 의 두 후처리(pair-selection / conf-immediate)는 **PLC cycle 전제**다:
전자는 c1→c2 짝을 기다리고, 후자는 cycle 당 1회만 발행해 finalize 로만 재무장한다.
이 데모는 "어떤 클릭이든 하나씩 계속 센다"가 요구라 둘 다 맞지 않는다.

그래서 엔진은 순수 1초 window 분류기로 쓰고(postp/immediate 모두 off), 여기서
DAL postprocessing/hysteresis.py 의 논리(tau_on/tau_off/min_on)를 그대로 따르되
**상승 edge(others → commit)만 이벤트로 발행**한다.

판정량은 **클릭 총확률 `1 - p_others`** 다 (c1/c2 중 하나가 아니라 합).
사용자 제안이고, 인공 케이스로 구조적 우월성을 확인했다:
  ① soft click 이 c1/c2 로 쪼개질 때(0.45/0.45, 총 0.90) 개별 최대 방식은 **놓친다**
  ② 한 클릭 안에서 확신이 c1→c2 로 옮겨가면 개별 방식은 release→재commit 으로 **2건 이중 카운트**
  ③ 진짜로 떨어진 클릭 2개 분리, others 우세 구간 FA 0 은 두 방식 동일
실제 golden/sim_dense 에서는 클릭이 결정적이라(쪼개짐 0건) 두 방식 결과가 같았다 — 즉
이 선택은 현재 성능을 바꾸지 않고 **현장 soft click 에서의 실패 모드만 제거**한다.

시간 기반 refractory 를 두지 않는 이유(실측 근거):
  같은 클릭의 나머지 window 는 commit 상태를 유지하므로 중복 카운트가 안 되고,
  다음 클릭은 확률이 tau_off 아래로 내려갔다 올라올 때 세진다 — 신호 기반이라
  시간 기반보다 정확하다. golden(클릭 2개)→2건, sim_dense 120s(24개)→24건·FA 0,
  최소 간격 2.0s 분리 확인 (tau_on .7~.9 × tau_off .3~.5 × min_on 1~2 전부 동일).
  단 window 자체가 1초라 1초 이내 연타는 원리적으로 분리 못 한다.
"""
from . import config as C
from .modelspec import SPEC

OTHERS = SPEC.others_index             # 모델 교체 계약은 app/modelspec.py 한 곳


class ClickCounter:
    def __init__(self, tau_on=None, tau_off=None, min_on=None):
        self.tau_on = C.TAU_ON if tau_on is None else float(tau_on)
        self.tau_off = C.TAU_OFF if tau_off is None else float(tau_off)
        self.min_on = C.MIN_ON if min_on is None else int(min_on)
        self.reset()

    def reset(self):
        """스트림 경계(장치 전환·복구)에서만. 옛 신호는 무관하니 완전히 재무장한다.
        카운트 총계는 건드리지 않는다."""
        self.on_click = False                  # 지금 '클릭 중' 상태인가 (클래스 무관)
        self._on = 0

    def resync(self):
        """재개(정지→재생) 경계 — latch(on_click)는 **보존**한다.

        방금 센 클릭의 tail 이 아직 1s window 안에 tau_on 위로 남아 있다. 여기서
        재무장하면 같은 소리가 상승 edge 로 다시 잡힌다 — 실측으로 물리 클릭 1개가
        2건이 됐다(0.6s 간격 = min_on 2 hop + 정지·재개 왕복). release 는 원래
        설계대로 신호가 tau_off 아래로 내려갈 때만 일어난다.

        대가: 정지 중에 시작해 재개 후에도 계속 tau_on 위인 클릭 1건은 놓친다.
        클릭은 수백 ms 소리라 그 창이 좁고, 반대쪽 실패(모든 클릭이 2배로 세짐)가
        데모에서 훨씬 치명적이다.
        """
        self._on = 0                           # window 연속성이 끊겼다 — 연속 high 만 초기화

    def set_level(self, level: int):
        """판정 기준 3단 → tau_on 만 이동 (엄격할수록 높은 문턱)."""
        self.tau_on = C.LEVEL_TAU_ON.get(int(level), C.TAU_ON)
        self._on = 0                           # 새 문턱으로 다시 세게 한다

    def push(self, probs):
        """window 확률 → 이번에 새로 발생한 이벤트 dict 또는 None.

        판정: p_click = 1 - p_others. 이 하나의 양으로 on/off 를 다 본다.
        반환 {"cls": 1|2, "score": p_click, "top": p[top]} — cls 는 표시·CAM 용이고
        카운트는 클래스와 무관하게 1건이다.
        """
        p = list(probs)
        p_click = 1.0 - float(p[OTHERS])

        if self.on_click and p_click < self.tau_off:      # release → 자동 재무장
            self.on_click = False
            self._on = 0

        ev = None
        if p_click >= self.tau_on:
            self._on += 1
            if self._on >= self.min_on and not self.on_click:   # ← 상승 edge = 클릭 1건
                cls = 1 + max(range(len(p) - 1), key=lambda i: p[i + 1])
                ev = {"cls": cls, "score": round(p_click, 4),
                      "top": round(float(p[cls]), 4)}
                self.on_click = True
        else:
            self._on = 0
        return ev

    def state(self):
        return {"on_click": self.on_click, "tau_on": round(self.tau_on, 3),
                "tau_off": round(self.tau_off, 3), "min_on": self.min_on}
