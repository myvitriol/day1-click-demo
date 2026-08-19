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


class Atlas:
    def __init__(self):
        self.fixed = False
        self.mean = self.comps = None
        self.span = 1.0
        self.ref_points = []                    # [{x,y,cls}]
        self.placeholder = False                # 임시(golden 기반) 지도인가
        self._session = []                      # 온라인 폴백용 embedding 모음
        if C.ATLAS_PATH.exists():
            d = json.loads(C.ATLAS_PATH.read_text())
            self.mean = np.array(d["mean"], np.float32)
            self.comps = np.array(d["components"], np.float32)   # [2, D]
            self.span = float(d.get("span", 1.0))
            self.ref_points = d.get("points", [])
            self.placeholder = bool(d.get("placeholder"))
            self.fixed = True
            print(f"[atlas] loaded {C.ATLAS_PATH.name}: {len(self.ref_points)} ref points")
        else:
            print("[atlas] no atlas.json - falling back to session-online PCA")

    def transform(self, e: np.ndarray):
        if self.fixed:
            return _project(e, self.mean, self.comps, self.span)
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
