"""Build atlas.json (reference map) from labeled wav files.

usage:
  python tools/build_atlas.py --dir click=/path/clicks --dir background=/path/bg
  python tools/build_atlas.py --golden        # dev: derive from DAL golden fixture

Each wav is sliced into 1s windows (hop 0.5s); every window becomes one reference
point of that class. The projection (PCA-2) is then FROZEN into atlas.json - live
points at demo time only use transform().
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (run from anywhere)

import numpy as np


def windows_from(path, sr, win, hop):
    import torchaudio
    wav, fsr = torchaudio.load(str(path))
    if fsr != sr:
        print(f"  skip {path.name}: sr {fsr} != {sr}")
        return
    x = wav[0].numpy()
    for s in range(0, len(x) - win + 1, hop):
        yield x[s:s + win]


def main():
    from app import config as C
    from app.atlas import build_atlas
    from app.engine import Engine

    ap = argparse.ArgumentParser("build_atlas")
    ap.add_argument("--dir", action="append", default=[],
                    help="cls=/path/to/wavs (repeatable)")
    ap.add_argument("--golden", action="store_true",
                    help="derive a dev atlas from the DAL golden fixture")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    eng = Engine(folds=1)          # embedding 은 fold0 만 쓰므로 1 fold 로 충분
    samples = []

    if a.golden:
        # 개발용 임시 지도 — golden 1 cycle(96k ch0, 실도메인)에서 뽑는다.
        # 진짜 참조 지도는 데모세트로 직접 녹음해 만들 것(atlas.placeholder=True 로 표시).
        import json
        exp = json.loads(C.GOLDEN_EXPECTED.read_text())["cycles"]["ok"]["pair"]
        hop = C.SR // 20                                  # 0.05s — 더 촘촘하게
        bg_keep = 0
        for i, w in enumerate(windows_from(C.GOLDEN_FLAC, C.SR, C.WIN_N, hop)):
            t = i * 0.05 + 0.5                            # window 중심 시각
            d = min(abs(t - tc) for tc in exp)
            if d < 0.40:
                cls = "click"
            elif d > 1.0:
                bg_keep += 1
                cls = "background"
            else:
                continue                                   # 애매한 경계는 버린다
            samples.append((eng.embed(w), cls))
        print(f"golden: {len(samples)} windows")

    for spec in a.dir:
        cls, _, d = spec.partition("=")
        files = sorted(Path(d).glob("*.wav")) + sorted(Path(d).glob("*.flac"))
        n0 = len(samples)
        for f in files:
            for w in windows_from(f, C.SR, C.WIN_N, C.WIN_N // 2):
                samples.append((eng.embed(w), cls))
        print(f"{cls}: {len(files)} files -> {len(samples) - n0} windows")

    if len(samples) < 10:
        print("not enough samples (need >= 10)")
        return 1
    build_atlas(samples, Path(a.out) if a.out else None,
                placeholder=bool(a.golden))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
