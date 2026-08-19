"""소리의 지도 배치를 보기 좋게 다시 깐다 (표시 전용 · placeholder 단계).

왜: 지금 atlas 는 golden 하나에서 뽑은 임시 지도라 PCA 투영이 그대로는 잘 안 읽힌다 —
"그 밖의 소리"가 대각선으로 길게 흩뿌려지고, 체결음은 두 덩어리로 쪼개져 한쪽이 화면
구석에 작게 떠 있었다. 데모에서 보여줄 그림은 단순하다: **그 밖의 소리는 넓고 촘촘한
한 덩어리, 기존 체결음은 그 옆에 조금 떨어진 작은 덩어리.**

어떻게: 군집별 등방 변환(중심 이동 + 스케일)을 만들어 atlas.json 에 `layout` 으로 남긴다.
참조점 좌표와 **실시간 embedding 좌표가 같은 변환을 통과**해야 실시간 점이 blob 위에
찍히므로, app/atlas.py 의 transform() 도 이 layout 을 읽어 쓴다(공유가 핵심이다).

진짜 지도는 데모 세트로 직접 녹음해 tools/build_atlas.py 로 다시 만든다. 그때 이
재배치는 필요 없어진다(layout 키를 지우면 원래 투영으로 돌아간다).

usage: python tools/relayout_atlas.py [--out atlas.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import config as C

# 목표 배치 — 화면 [0,1]^2 기준. 사용자 지시: 그 밖의 소리는 dense 하지만 크게,
# 기존 체결음은 옆에 조금 떨어져 작게.
DST = {
    "background": {"c": (0.37, 0.50), "r": 0.205},
    "click":      {"c": (0.75, 0.57), "r": 0.078},
}
GAP_NOTE = "두 중심 거리 0.39, 반경 합 0.283 → 살짝 떨어진 두 덩어리"


def cluster_stat(pts):
    """중심과 **축별** 반경(p90 — outlier 하나가 스케일을 망치지 않게).

    축별로 재야 세로로 긴 무리를 둥글게 펼 수 있다.
    """
    a = np.array([[p["x"], p["y"]] for p in pts], np.float64)
    c = a.mean(0)
    rx = max(float(np.percentile(np.abs(a[:, 0] - c[0]), 90)), 1e-6)
    ry = max(float(np.percentile(np.abs(a[:, 1] - c[1]), 90)), 1e-6)
    return c, [rx, ry]


def main():
    ap = argparse.ArgumentParser("relayout_atlas")
    ap.add_argument("--out", default=None, help="기본: atlas.json 제자리 갱신")
    a = ap.parse_args()

    path = C.ATLAS_PATH
    d = json.loads(path.read_text())
    pts = d["points"]

    groups = {}
    for p in pts:
        groups.setdefault("background" if p["cls"] == "background" else "click", []).append(p)
    missing = [k for k in DST if k not in groups]
    if missing:
        print(f"[relayout] 없는 군집 {missing} - 그대로 둔다")
        return

    layout = []
    for k, g in groups.items():
        c, r = cluster_stat(g)
        dst = DST[k]
        layout.append({"cls": k,
                       "sc": [round(float(c[0]), 5), round(float(c[1]), 5)],
                       "sr": [round(r[0], 5), round(r[1], 5)],
                       "dc": list(dst["c"]), "dr": dst["r"]})
        print(f"  {k:11s} n={len(g):3d}  원본 중심 ({c[0]:.3f},{c[1]:.3f}) "
              f"r=({r[0]:.3f},{r[1]:.3f})  →  ({dst['c'][0]:.2f},{dst['c'][1]:.2f}) r={dst['r']:.3f}")

    # 참조점 좌표도 같은 변환으로 갱신한다. 원본은 sx/sy 로 남겨 되돌릴 수 있게.
    from app.atlas import apply_layout
    for p in pts:
        if "sx" not in p:
            p["sx"], p["sy"] = p["x"], p["y"]
        cls = "background" if p["cls"] == "background" else "click"
        x, y = apply_layout(p["sx"], p["sy"], layout, cls)
        p["x"], p["y"] = round(x, 4), round(y, 4)

    d["layout"] = layout
    d["layout_note"] = GAP_NOTE
    out = Path(a.out) if a.out else path
    out.write_text(json.dumps(d))
    xs = [p["x"] for p in pts if p["cls"] == "background"]
    ys = [p["y"] for p in pts if p["cls"] == "background"]
    cxs = [p["x"] for p in pts if p["cls"] != "background"]
    cys = [p["y"] for p in pts if p["cls"] != "background"]
    print(f"\n  그 밖의 소리 x {min(xs):.3f}~{max(xs):.3f}  y {min(ys):.3f}~{max(ys):.3f}")
    print(f"  기존 체결음  x {min(cxs):.3f}~{max(cxs):.3f}  y {min(cys):.3f}~{max(cys):.3f}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
