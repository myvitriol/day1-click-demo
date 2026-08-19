"""Environment / audio / engine diagnosis for the demo laptop.

usage: python doctor.py [--dal PATH] [--skip-engine] [--skip-audio]
Exit code 0 = usable, 1 = blocking problem found.
"""
import argparse
import json
import sys
import time
from pathlib import Path

OK, WARN, FAIL = "\033[32mOK  \033[0m", "\033[33mWARN\033[0m", "\033[31mFAIL\033[0m"
problems = []


def report(tag, msg, blocking=False):
    print(f"  [{tag}] {msg}")
    if tag is FAIL and blocking:
        problems.append(msg)


def main():
    ap = argparse.ArgumentParser("doctor")
    ap.add_argument("--dal", default=None)
    ap.add_argument("--skip-engine", action="store_true")
    ap.add_argument("--skip-audio", action="store_true")
    a = ap.parse_args()
    if a.dal:
        import os
        os.environ["DAL_PATH"] = a.dal
    from app import config as C

    print("== day1-click-demo doctor ==")

    # 1. python / packages
    v = sys.version_info
    report(OK if v >= (3, 10) else FAIL, f"python {v.major}.{v.minor}.{v.micro}", blocking=v < (3, 10))
    for mod in ("numpy", "torch", "torchaudio", "fastapi", "uvicorn"):
        try:
            m = __import__(mod)
            report(OK, f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            report(FAIL, f"{mod} not installed", blocking=True)

    # 2. DAL + weights
    try:
        import DAL  # noqa
        report(OK, f"DAL importable ({Path(DAL.__file__).parent})")
    except ImportError:
        report(FAIL, f"DAL not importable (expected at {C.DAL_DIR})", blocking=True)
        return finish()
    mani_p = C.WEIGHTS_DIR / "manifest.json"
    if not mani_p.exists():
        report(FAIL, f"manifest missing: {mani_p}", blocking=True)
        return finish()
    mani = json.loads(mani_p.read_text())
    ver = mani["default_version"]
    pts = sorted((C.WEIGHTS_DIR / ver).glob("*.pt"))
    big = [p for p in pts if p.stat().st_size > 1e6]
    if big:
        report(OK, f"weights {ver}: {len(big)} folds")
    else:
        report(FAIL, f"weights {ver}: missing or LFS pointers - run: git lfs pull "
                     f"--include='DAL/inference/hdtransys/weights/{ver}/*'", blocking=True)
    report(OK if C.GOLDEN_FLAC.exists() else WARN, f"golden fixture {'present' if C.GOLDEN_FLAC.exists() else 'missing'}")

    # 3. audio devices
    if not a.skip_audio:
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            ins = [(i, d) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
            report(OK if ins else FAIL, f"{len(ins)} input device(s)", blocking=not ins)
            hit = [(i, d) for i, d in ins if C.DEVICE_HINT.lower() in d["name"].lower()]
            if hit:
                i, d = hit[0]
                report(OK, f"external interface found: '{d['name']}' ({d['max_input_channels']}ch)")
                try:
                    extra = sd.CoreAudioSettings(change_device_parameters=True) \
                        if sys.platform == "darwin" else None
                    with sd.InputStream(device=i, samplerate=C.SR, channels=1,
                                        dtype="float32", blocksize=0, extra_settings=extra) as st:
                        got = int(st.samplerate)
                        time.sleep(1.0)
                    report(OK if got == C.SR else FAIL,
                           f"opened at {got} Hz (need {C.SR})", blocking=got != C.SR)
                except Exception as e:
                    report(FAIL, f"cannot open at {C.SR} Hz: {e}", blocking=True)
            else:
                report(WARN, f"no device matching '{C.DEVICE_HINT}' - default input will be used. "
                             "On macOS check mic permission (System Settings > Privacy > Microphone).")
        except ImportError:
            report(WARN, "sounddevice not installed - mic capture unavailable (file source still works)")
    else:
        report(WARN, "audio checks skipped")

    # 4. engine bench
    if not a.skip_engine:
        try:
            from app.engine import Engine
            t0 = time.time()
            eng = Engine()
            report(OK, f"engine loaded in {time.time() - t0:.1f}s - folds={eng.folds} "
                       f"p95={eng.p95_ms:.0f}ms (budget {C.BENCH_BUDGET_MS}ms)")
            if eng.embed_ok:
                report(OK, "embed() available - sound map enabled")
            else:
                report(WARN, "DAL checkout lacks embed() - demo runs but the sound map "
                             "is disabled. Commit the DAL embed patch and update the pin.")
        except Exception as e:
            report(FAIL, f"engine load failed: {e}", blocking=True)
    else:
        report(WARN, "engine checks skipped")

    return finish()


def finish():
    print()
    if problems:
        print(f"VERDICT: NOT READY - {len(problems)} blocking problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERDICT: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
