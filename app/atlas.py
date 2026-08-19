"""소리의 지도 좌표계.

atlas.json(참조 embedding 의 PCA 투영)이 있으면 그 고정 좌표계로 transform 만 한다.
없으면 세션 중 모은 embedding 으로 온라인 PCA — 참조 무리는 없지만 점은 의미 있게 움직인다.
(참조 데이터 확보 전까지의 정직한 폴백. 화면에는 '참조 지도 없음'이 표시된다.)
"""
import base64
import json

import numpy as np

from . import config as C


def _project(e, mean, comps, span):
    xy = (e - mean) @ comps.T / span            # 대략 [-1,1]
    return float(np.clip(xy[0] * .5 + .5, .03, .97)), float(np.clip(xy[1] * .5 + .5, .03, .97))


BLEND_SIGMA = 0.18          # 실시간 점의 군집 배정 폭. 두 중심 거리(약 0.47)보다 충분히
#   작아야 경계에서만 섞인다. 원본 군집 반경을 쓰면 흩어진 쪽이 상대를 끌어당겨 배치가
#   뭉개진다(실측: 목표 0.37/0.75 가 0.44~0.65 / 0.47~0.77 로 겹쳤다).


def _one(x, y, g):
    """군집 하나의 변환: 축별 정규화 → 반경 압축 → 목표 원으로.

    ① 축별 p90 으로 나눈다 — 세로로 길게 늘어진 무리가 **둥근 덩어리**가 된다.
    ② 반경을 tanh 로 눌러 **모든 점이 목표 원 안에** 들어오게 한다. 그냥 p90 으로
       나누면 상위 10% 가 원 밖으로 튀어나가 무리가 흩어져 보였다(실측: 목표 0.205
       반경이 0.247~0.915 로 퍼졌다). 압축은 중심부는 거의 그대로 두고 먼 점만
       끌어당기므로 가운데가 촘촘해지는 효과도 같이 난다.
    """
    sc, dc = g["sc"], g["dc"]
    sr = g["sr"]
    srx, sry = (sr, sr) if isinstance(sr, (int, float)) else (sr[0], sr[1])
    u = (x - sc[0]) / max(srx, 1e-6)
    v = (y - sc[1]) / max(sry, 1e-6)
    r = float(np.hypot(u, v))
    if r > 1e-9:
        k = float(np.tanh(r * 0.85)) / r        # r=1 → 0.69, r→∞ → 1.0 (원 안 보장)
        u *= k; v *= k
    return dc[0] + u * g["dr"], dc[1] + v * g["dr"]


def apply_layout(x, y, layout, cls=None):
    """표시 배치 재매핑 (placeholder 지도 전용, tools/relayout_atlas.py 가 만든다).

    참조점과 **실시간 embedding 이 같은 이 함수를 통과**해야 실시간 점이 blob 위에
    찍힌다. layout 이 없으면(진짜 atlas) 아무것도 하지 않는다.

    cls 를 주면(참조점처럼 라벨을 아는 경우) 그 군집 변환만 그대로 쓴다. 라벨이 없는
    실시간 점은 거리 기반으로 섞는다 — 경계에서 튀지 않게.
    """
    if not layout:
        return x, y
    if cls is not None:
        g = next((q for q in layout if q["cls"] == cls), None)
        if g is not None:
            ax, ay = _one(x, y, g)
            return float(np.clip(ax, .02, .98)), float(np.clip(ay, .02, .98))
    sw = ax = ay = 0.0
    for g in layout:
        sc = g["sc"]
        d2 = (x - sc[0]) ** 2 + (y - sc[1]) ** 2
        w = float(np.exp(-d2 / (2.0 * BLEND_SIGMA ** 2))) + 1e-9
        gx, gy = _one(x, y, g)
        ax += w * gx; ay += w * gy; sw += w
    return float(np.clip(ax / sw, .02, .98)), float(np.clip(ay / sw, .02, .98))


class Atlas:
    def __init__(self):
        self.fixed = False
        self.mean = self.comps = None
        self.span = 1.0
        self.ref_points = []                    # [{x,y,cls}]
        self.layout = []                        # 표시 배치 재매핑 파라미터
        self.placeholder = False                # 임시(golden 기반) 지도인가
        self._session = []                      # 온라인 폴백용 embedding 모음
        if C.ATLAS_PATH.exists():
            d = json.loads(C.ATLAS_PATH.read_text())
            self.mean = np.array(d["mean"], np.float32)
            self.comps = np.array(d["components"], np.float32)   # [2, D]
            self.span = float(d.get("span", 1.0))
            self.ref_points = d.get("points", [])
            self.layout = d.get("layout") or []      # 표시 배치 재매핑(placeholder 단계)
            self.placeholder = bool(d.get("placeholder"))
            self.fixed = True
            print(f"[atlas] loaded {C.ATLAS_PATH.name}: {len(self.ref_points)} ref points")
        else:
            print("[atlas] no atlas.json - falling back to session-online PCA")

    def transform(self, e: np.ndarray):
        if self.fixed:
            # 모델 교체로 embedding 차원이 바뀌면 기존 atlas 는 무효다.
            # 예외로 cycle 스레드를 죽이지 않고 지도만 끈다(Codex P1).
            if e.shape[0] != self.mean.shape[0]:
                print(f"[atlas] dim mismatch: atlas={self.mean.shape[0]} model={e.shape[0]} "
                      "- disabling the fixed map (rebuild with tools/build_atlas.py)")
                self.fixed = False
                self.ref_points = []
                self.mean = self.comps = None
                return 0.5, 0.5
            return apply_layout(*_project(e, self.mean, self.comps, self.span), self.layout)
        # 폴백: 세션 온라인 PCA (20개 모일 때까지는 중앙 부근 고정)
        self._session.append(e.astype(np.float32))
        if len(self._session) < 20:
            return 0.5, 0.5
        if len(self._session) % 20 == 0 or self.mean is None:
            X = np.stack(self._session[-500:])
            self.mean = X.mean(0)
            _, _, vt = np.linalg.svd(X - self.mean, full_matrices=False)
            self.comps = vt[:2]
            self.span = float(np.abs((X - self.mean) @ self.comps.T).max() + 1e-9)
        return _project(e, self.mean, self.comps, self.span)

    def hello_payload(self):
        return {"fixed": self.fixed, "points": self.ref_points,
                "layout": self.layout,          # 있으면 웹이 quantile 매핑을 끈다
                "placeholder": self.placeholder}


def build_atlas(samples, out_path=None, placeholder=False):
    """samples = [(embedding[D], cls_str)] → atlas.json 저장. tools/build_atlas.py 가 사용."""
    X = np.stack([e for e, _ in samples]).astype(np.float32)
    mean = X.mean(0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = vt[:2]
    span = float(np.abs((X - mean) @ comps.T).max() + 1e-9)
    pts = []
    for e, cls in samples:
        x, y = _project(e, mean, comps, span)
        pts.append({"x": round(x, 4), "y": round(y, 4), "cls": cls})
    d = {"mean": mean.tolist(), "components": comps.tolist(), "span": span, "points": pts,
         "placeholder": bool(placeholder)}
    out = out_path or C.ATLAS_PATH
    out.write_text(json.dumps(d))
    print(f"[atlas] wrote {out} ({len(pts)} points)")
    return d
