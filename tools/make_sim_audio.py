"""Build a longer simulation source from the golden cycle.

golden(14.1s, 클릭 2개)을 그대로 반복하면 15초마다 검출·무음 경계가 나온다.
여기서는 golden 의 조용한 구간(1..4s)을 랜덤 길이로 이어붙여 사이를 채운
~3분짜리 파일을 만든다 — 무음 0, 클릭 간격 20~30s, 전부 실녹음 소리.

usage: python tools/make_sim_audio.py [--minutes 3] [--out sim/sim_long.flac]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchaudio

from app import config as C

XF = int(0.02 * C.SR)          # 이음새 crossfade 20ms


def xfade_concat(parts):
    out = parts[0]
    ramp = np.linspace(0, 1, XF, dtype=np.float32)
    for p in parts[1:]:
        out[-XF:] = out[-XF:] * (1 - ramp) + p[:XF] * ramp
        out = np.concatenate([out, p[XF:]])
    return out


def main():
    ap = argparse.ArgumentParser("make_sim_audio")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--out", default="sim/sim_long.flac")
    ap.add_argument("--dense", action="store_true",
                    help="frequent clicks: splice c1/c2 snippets every 5-10s instead of full cycles")
    a = ap.parse_args()

    wav, sr = torchaudio.load(str(C.GOLDEN_FLAC))
    assert sr == C.SR
    x = wav[0].numpy()
    quiet = x[int(1.0 * sr):int(4.0 * sr)]          # 조용(기계 대기) 3s 풀

    rng = np.random.default_rng(20260819)

    def quiet_span(sec):
        n = int(sec * sr)
        parts = []
        while sum(len(p) for p in parts) < n + XF:
            s = rng.integers(0, len(quiet) - sr)
            parts.append(quiet[s:s + rng.integers(sr, 2 * sr)].copy())
        return xfade_concat(parts)[:n]

    if a.dense:
        # c1+c2 통짜 스니펫(7.3~11.2s) — 엔진 계약이 pair(c1→c2)라 c2 단독은 안 잡힌다(실측).
        # 각 스니펫당 c1 에서 검출 1회가 확실히 나오고, 간격을 좁혀 검출 빈도를 올린다.
        snip = x[int(7.30 * sr):int(11.20 * sr)].copy()
        parts = [quiet_span(rng.uniform(2, 5))]
        total = len(parts[0]); target = int(a.minutes * 60 * sr); n_cycles = 0
        while total < target:
            sn = snip * float(10 ** (rng.uniform(-2, 2) / 20))   # ±2dB 변주
            parts.append(np.clip(sn, -1, 1))
            n_cycles += 1
            parts.append(quiet_span(rng.uniform(4, 8)))          # 스니펫 간격 4~8s
            total = sum(len(p) for p in parts)
    else:
        parts = [quiet_span(rng.uniform(4, 8))]
        total = len(parts[0]); target = int(a.minutes * 60 * sr); n_cycles = 0
        while total < target:
            parts.append(x.copy())                   # 사이클(클릭 2개 포함)
            n_cycles += 1
            parts.append(quiet_span(rng.uniform(8, 18)))     # 사이클 사이 8~18s
            total = sum(len(p) for p in parts)
    y = xfade_concat(parts)

    out = Path(__file__).resolve().parent.parent / a.out
    out.parent.mkdir(exist_ok=True)
    t = torch.from_numpy(y).unsqueeze(0)
    try:
        torchaudio.save(str(out), t, sr)
    except Exception:
        out = out.with_suffix(".wav")
        torchaudio.save(str(out), t, sr)
    print(f"wrote {out}  {len(y)/sr:.1f}s  clicks/cycles={n_cycles}  peak={np.abs(y).max():.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
