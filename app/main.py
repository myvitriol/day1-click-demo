"""엔트리 로직. selftest(headless) 와 serve(uvicorn+WS) 두 모드.

fastapi/uvicorn import 는 serve 안에서만 한다 — selftest 는 torch+DAL 만으로 돈다.
pause/resume 은 WS 핸들러가 ctrl 의 bool 을 직접 뒤집는다(즉답). 중복 명령은
무시하지 않고 현재 상태를 재전송해 어긋난 클라이언트를 치유한다.
"""
import argparse
import json
import os
import queue

from . import config as C
from .atlas import Atlas
from .audio import FileSource, RingBuffer, WindowSlicer
from .cycle import CycleController
from .engine import Engine


def build_args():
    ap = argparse.ArgumentParser("day1-click-demo")
    ap.add_argument("--source", choices=["mic", "file"], default="mic",
                    help="audio source (default: mic)")
    ap.add_argument("--file", default=str(C.GOLDEN_FLAC),
                    help="file path for --source file (default: DAL golden fixture)")
    ap.add_argument("--fast", action="store_true", help="file source without realtime pacing")
    ap.add_argument("--loop", action="store_true", help="loop the file source")
    ap.add_argument("--port", type=int, default=C.PORT)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--strictness", type=int, default=C.DEFAULT_LEVEL, choices=list(C.LEVELS))
    ap.add_argument("--folds", type=int, default=None, help="ensemble folds (default: auto)")
    ap.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto)")
    ap.add_argument("--postp", choices=["off", "pair"],
                    default=("pair" if C.PAIR_POSTP else "off"),
                    help="pair post-processing (default from config; selftest always uses pair)")
    ap.add_argument("--selftest", action="store_true",
                    help="headless: feed golden file, compare final pair, exit 0/1")
    return ap


def make_pipeline(args, emit):
    ring = RingBuffer(C.RING_S * C.SR)
    eng = Engine(folds=args.folds, device=args.device,
                 pair_postp=(args.postp == "pair"))
    eng.set_level(args.strictness)
    atlas = Atlas()
    win_q = queue.Queue(maxsize=C.QUEUE_MAX)

    realtime = not (args.source == "file" and args.fast)
    ctrl = CycleController(ring, win_q, eng, atlas, emit, realtime=realtime)
    if args.source == "file":
        src = FileSource(ring, args.file, fast=args.fast, loop=args.loop, on_error=ctrl.fail)
    else:
        from .audio import MicSource, NoSource
        try:                                   # 기본 장치가 96kHz 로 안 열려도 서버는 뜬다
            src = MicSource(ring, on_error=ctrl.fail)
        except Exception as e:
            print(f"[audio] mic not started: {e}")
            print("[audio] -> open the page and pick an input from the dropdown")
            src = NoSource(str(e))
    slicer = WindowSlicer(ring, win_q, src.eof, on_error=ctrl.fail)
    ctrl.src, ctrl.slicer = src, slicer        # status() 가 파이프라인 건강을 보고
    return ring, eng, atlas, ctrl, src, slicer


# ── selftest ────────────────────────────────────────────────────
def selftest(args):
    args.source, args.fast, args.loop = "file", True, False
    args.postp = "pair"                        # golden pair 대조는 pair postp 전제
    events = []
    ring, eng, atlas, ctrl, src, slicer = make_pipeline(args, events.append)
    ctrl.auto_pause = False                    # 전 구간 추론 → EOF finalize (golden parity)
    ctrl.slicer_done = slicer.done
    src.start(); slicer.start(); ctrl.start()
    ctrl.join(timeout=180)

    exp = json.loads(C.GOLDEN_EXPECTED.read_text())["cycles"]["ok"]
    fin = [e for e in events if e["type"] == "eof_final"]
    if not fin:
        print("SELFTEST FAIL: no eof_final event")
        return 1
    pair = fin[-1]["pair"]
    got = [round(p["t"], 3) if p else None for p in pair]
    want = exp["pair"]
    ok = (got[0] is not None and got[1] is not None
          and abs(got[0] - want[0]) < 0.011 and abs(got[1] - want[1]) < 0.011
          and fin[-1]["n_windows"] == exp["n_frames"])
    print(f"SELFTEST windows={fin[-1]['n_windows']} (expect {exp['n_frames']})  "
          f"pair={got}  want={want}")

    # 기본 배포 모드(immediate-only) 스모크 — pair 강제 selftest 만으론 off 경로가 무검증 (Codex R7)
    # 기본 배포 모드(연속 카운터) 스모크 — golden 은 클릭 정확히 2개다
    off_ok = False
    try:
        import torch
        import torchaudio
        from .counter import ClickCounter
        eng2 = Engine(folds=1, pair_postp=False)
        wav2, _ = torchaudio.load(str(C.GOLDEN_FLAC))
        x2 = wav2[0].numpy()
        cnt = ClickCounter()
        n, ts = 0, []
        for k in range((len(x2) - C.WIN_N) // C.HOP_N + 1):
            s0 = k * C.HOP_N
            o = eng2.clf.predict(torch.from_numpy(x2[s0:s0 + C.WIN_N]))
            if cnt.push(o["probs"]) is not None:
                n += 1
                ts.append(round(k * 0.2 + 0.5, 1))
        off_ok = (n == 2)
        print(f"COUNTER smoke: {n} clicks at {ts} (expect 2)  {'OK' if off_ok else 'FAIL'}")
    except Exception as e:
        print(f"COUNTER smoke FAIL: {e}")
    ok = ok and off_ok
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ── serve ───────────────────────────────────────────────────────
def _build_id():
    """코드 버전 식별자. 브라우저가 재연결할 때 이걸 비교해 **스스로 새로고침**한다.

    git pull 로 백엔드는 새 코드가 되지만, 이미 열려 있는 탭은 index.html 을 다시
    요청하지 않아 옛 화면이 그대로 남는다. WS 는 자동 재연결되므로 화면과 백엔드가
    어긋난 채로 계속 도는 상황이 생긴다. 커밋 해시만 쓰면 개발 중(uncommitted)에는
    안 바뀌므로 index.html 의 mtime/size 도 섞는다.
    """
    import subprocess
    h = ""
    try:
        h = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        pass
    try:
        st = (C.REPO / "web/index.html").stat()
        return f"{h or 'nogit'}-{int(st.st_mtime)}-{st.st_size}"
    except Exception:
        return h or "unknown"


BUILD_ID = _build_id()


def serve(args):
    import asyncio  # noqa: F401  (dev_lock/to_thread 에서 사용)
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    loop_holder = {}
    ui_q = None                                # lifespan 에서 생성
    clients = set()

    def emit(msg):                             # cycle 스레드 → asyncio
        lp = loop_holder.get("loop")
        if lp is not None:
            lp.call_soon_threadsafe(ui_q.put_nowait, msg)

    ring, eng, atlas, ctrl, src, slicer = make_pipeline(args, emit)
    from .audio import list_inputs
    nonlocal_src = {"src": src}                # 장치 전환 시 최신 소스 추적
    dev_lock = asyncio.Lock()                  # 전환 직렬화 (PortAudio 호출은 스레드로)

    app = FastAPI()
    app.mount("/assets", StaticFiles(directory=C.REPO / "web/assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(C.REPO / "web/index.html")

    async def broadcaster():
        while True:
            try:
                msg = await ui_q.get()
                try:
                    txt = json.dumps(msg, default=float)
                except Exception as e:
                    print(f"[ws] dropped unserializable message type={msg.get('type')}: {e}")
                    continue
                for ws in list(clients):
                    try:
                        await ws.send_text(txt)
                    except Exception:
                        clients.discard(ws)
            except Exception as e:             # 이 task 는 절대 죽으면 안 된다
                print(f"[ws] broadcaster error: {e}")

    async def streamer():
        """10Hz 화면 스트림. 파형과 스펙트로그램을 **같은 시간 격자**로 만든다.

        예전에는 tick 마다 "그 사이 들어온 만큼"을 20 포인트로 눌러 담고(포인트당 5ms,
        지터에 따라 가변), 스펙트로그램은 격 tick 에서 그 tick 구간만 STFT 했다(컬럼당
        25ms, 구간 누락). 웹은 둘 다 1단위=1px 로 그리므로 같은 폭에서 파형이 5배 빠르게
        흘렀다(실측 196.9 pt/s vs 39.4 col/s). 이제 둘 다 절대 sample 격자(DISP_HOP=10ms)
        에서 같은 프레임을 떠 1:1 로 정렬한다 — 각 100/s, 1px = 10ms.

        정지 중에도 status 는 10Hz — 상태가 늦게 도는 일이 없게.
        """
        seq = 0
        tick = 0
        nxt = None                             # 다음 프레임의 시작 sample (절대 인덱스)
        G, NF = C.DISP_HOP, C.DISP_NFFT
        MAXF = C.SR // G                       # 한 tick 에 만들 프레임 상한 = 1초분
        from .cycle import pack_spec, spec_rows
        while True:
            try:
                await asyncio.sleep(1 / C.STREAM_HZ)
                tick += 1
                seq += 1
                w = ring.write_idx
                if ctrl.paused:
                    nxt = None                 # 재개는 live edge 부터
                    await ui_q.put({"type": "stream", "seq": seq, "env": None,
                                    "spec": None, "status": ctrl.status()})
                    continue
                if nxt is None:
                    nxt = max(0, w - NF)
                nf = (w - nxt - NF) // G + 1   # NFFT 를 채울 수 있는 프레임 수
                if nf <= 0:                    # 입력 없음(미선택·죽음) — 상태만 보낸다
                    if tick % C.STREAM_HZ == 0:
                        await ui_q.put({"type": "stream", "seq": seq, "env": None,
                                        "spec": None, "status": ctrl.status()})
                    continue
                if nf > MAXF:                  # 밀렸다 — 따라잡고 그만큼 화면은 건너뛴다
                    nxt += (nf - MAXF) * G
                    nf = MAXF
                seg = ring.read_at(nxt, (nf - 1) * G + NF)
                if seg is None:                # ring 이 이미 지나갔다 — live edge 로 재동기
                    nxt = None
                    continue
                nxt += nf * G
                env = [[round(float(seg[i * G:(i + 1) * G].min()), 4),
                        round(float(seg[i * G:(i + 1) * G].max()), 4)] for i in range(nf)]
                spec = None                    # 프레임 i 의 시작 sample 이 env 와 동일하다
                if ctrl.spec_floor is not None:
                    spec = pack_spec(spec_rows(seg) - ctrl.spec_floor[:, None])
                await ui_q.put({"type": "stream", "seq": seq, "env": env, "spec": spec,
                                "status": ctrl.status()})
            except Exception as e:
                print(f"[ws] streamer error: {e}")

    def hello_payload():
        cur = nonlocal_src["src"]
        return {"type": "hello", "sr": C.SR, "build": BUILD_ID, "device": cur.info(),
                "snap": {"pre_s": C.SNAP_PRE_S, "post_s": C.SNAP_POST_S,
                         "pre_frac": C.SNAP_PRE_S / (C.SNAP_PRE_S + C.SNAP_POST_S)},
                "inputs": (list_inputs() if args.source == "mic" else []),
                "source_kind": cur.info().get("kind"),
                "postp": args.postp,
                "model": {"version": eng.version, "folds": eng.folds,
                          "p95_ms": round(eng.p95_ms, 1)},
                "levels": {str(k): C.LEVEL_NAMES[k] for k in C.LEVELS},
                "level": eng.level, "hop": C.HOP_S,
                "paused": ctrl.paused, "detect_on": ctrl.detect_on,
                "counter": ctrl.counter.state(),
                "last_freeze": ctrl.last_freeze,
                "click_total": ctrl.click_total, "count_rev": ctrl.count_rev,
                "atlas": atlas.hello_payload()}

    @app.websocket("/ws")
    async def ws_ep(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        await ws.send_text(json.dumps(hello_payload(), default=float))
        try:
            while True:
                m = json.loads(await ws.receive_text())
                cmd = m.get("cmd")
                if cmd == "pause":
                    ctrl.pause_now()
                    # 항상 즉시 ACK(방송) — 첫 pause 든 중복이든 현재 상태가 진실
                    await ui_q.put({"type": "paused", "rev": ctrl.rev})
                    if ctrl.last_freeze is not None:
                        await ws.send_text(json.dumps(ctrl.last_freeze, default=float))
                elif cmd == "resume":
                    ctrl.resume_now()
                    await ui_q.put({"type": "resume", "rev": ctrl.rev})
                elif cmd == "detect":
                    ctrl.detect_now(bool(m.get("on", True)))
                    await ui_q.put({"type": "detect", "on": ctrl.detect_on})
                elif cmd == "reset_count":
                    ctrl.reset_count_now()
                    await ui_q.put({"type": "count", "click_total": ctrl.click_total,
                                    "count_rev": ctrl.count_rev})
                elif cmd == "dial":
                    ctrl.req_dial(int(m.get("level", 0)))
                elif cmd == "device" and args.source == "mic":
                    idx = int(m.get("index", -1))

                    def _swap(old, want):
                        # 생성 실패 → old 유지. old 정지 후 start 실패 → old 로 복귀 시도.
                        from .audio import MicSource, NoSource
                        new = MicSource(ring, on_error=ctrl.fail, device=want)  # 미시작
                        old_idx = old.info().get("index", -1)
                        old.stop()
                        ctrl.req_source_switch()   # ★ 새 stream 이 ring 에 쓰기 시작하기 전에 경계를 박는다
                        try:
                            new.start()
                            return new
                        except Exception:
                            try:
                                new.stop()
                            except Exception:
                                pass
                            if old_idx is not None and old_idx >= 0:
                                back = MicSource(ring, on_error=ctrl.fail, device=old_idx)
                                ctrl.req_source_switch()      # 복귀도 경계가 필요하다
                                back.start()
                                ctrl.src = back
                                nonlocal_src["src"] = back
                                raise RuntimeError("new device failed; previous input restored")
                            from .audio import NoSource as _NS   # 복귀할 장치도 없다 → 미선택 상태
                            ns = _NS("previous input could not be restored")
                            ctrl.src = ns
                            nonlocal_src["src"] = ns
                            raise RuntimeError("new device failed; no input active")

                    async with dev_lock:
                        old = nonlocal_src["src"]
                        try:
                            new = await asyncio.to_thread(_swap, old, idx)
                            nonlocal_src["src"] = new
                            ctrl.src = new
                            await ui_q.put({"type": "device", "device": new.info(),
                                            "inputs": list_inputs()})
                        except Exception as e:
                            await ui_q.put({"type": "error",
                                            "msg": f"input switch failed: {e}"})
                            cur = nonlocal_src["src"]
                            await ui_q.put({"type": "device", "device": cur.info(),
                                            "inputs": list_inputs()})
        except WebSocketDisconnect:
            clients.discard(ws)

    @app.on_event("startup")
    async def _startup():
        nonlocal ui_q
        loop_holder["loop"] = asyncio.get_running_loop()
        ui_q = asyncio.Queue(maxsize=1000)
        src.start(); slicer.start(); ctrl.start()
        asyncio.create_task(broadcaster())
        asyncio.create_task(streamer())
        if not args.no_browser:
            import webbrowser
            webbrowser.open(f"http://{C.HOST}:{args.port}/")

    print(f"[serve] http://{C.HOST}:{args.port}/  (source={args.source})")
    # 돌고 있는 데모를 뒤에서 정확히 찾을 수 있게 PID 를 남긴다. pgrep -f 패턴으로
    # 찾는 방식은 위험하다 — 그 문자열을 명령줄에 가진 **자기 셸까지 잡는다**(실제로
    # 그렇게 셸이 죽는 것을 봤다). 파일에는 PID 와 포트를 적어둔다.
    lock = C.REPO / ".run.lock"
    try:
        lock.write_text(f"{os.getpid()}\n{args.port}\n")
    except OSError:
        pass
    try:
        uvicorn.run(app, host=C.HOST, port=args.port, log_level="warning")
    finally:
        try:
            if lock.exists() and lock.read_text().split("\n")[0].strip() == str(os.getpid()):
                lock.unlink()
        except OSError:
            pass


def main(argv=None):
    args = build_args().parse_args(argv)
    if args.selftest:
        raise SystemExit(selftest(args))
    serve(args)
