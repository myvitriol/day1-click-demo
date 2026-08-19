"""연속 inference + play/stop (시험 선언 없음).

모델은 재생 중이면 항상 inference 모드다 — 체결/방해음 같은 라벨 구분이 없다.
- 재생: 매 window predict. 검출(pair 슬롯 첫 등장) → snapshot 완성 시 freeze + 자동 정지.
- 정지: 추론·화면 모두 멈춘다(오디오와 ring 은 계속 흐른다).
- 재개: 다음 window 직전 classifier finalize(자동 reset) 후 새로 듣기 시작.
  검출이 첫 슬롯에서 화면을 멈추므로 한 번의 '듣기'에 여러 pair 가 섞이는 일은 없다
  (DAL 의 cycle-경계 계약은 resume/dial 시 finalize 로 지킨다).

즉답성·순서 원칙 (Codex R3):
- paused 의 진실은 이 객체의 bool 하나. WS 핸들러가 pause_now()/resume_now() 로 직접 뒤집는다.
- rev   = 정지/재생 전환마다 +1. 모든 상태 메시지에 실어 보내고, 클라이언트는 낡은 rev 를 버린다.
- epoch = '듣기' 세대. resume/dial 마다 +1. snapshot 은 자기 epoch 가 지나갔으면 스스로 폐기된다
  (pause 중 만들던 정지 화면이 resume 직후에 도착해 화면을 되얼리는 레이스 차단).
"""
import base64
import io
import queue
import threading
import time
import wave as wavmod

import numpy as np

from . import config as C


from .modelspec import SPEC               # 표시용 라벨. 카운트는 클래스와 무관하게 1건


def wav_b64(seg: np.ndarray) -> str:
    pcm = (np.clip(seg, -1, 1) * 32767).astype("<i2")
    bio = io.BytesIO()
    with wavmod.open(bio, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(C.SR)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(bio.getvalue()).decode()


def click_onset(w: np.ndarray) -> int:
    """window 안에서 클릭이 시작된 sample 위치. 짧은 창 RMS 의 증가폭이 가장 큰 지점.

    window 중심을 이벤트 시각으로 쓰면 클릭이 window 어디에 있든 같은 값이 나와서
    화면이 실제 소리와 어긋난다 — min_on=1 에서는 클릭이 window 뒤쪽에 처음 들어온
    순간에 발생하므로 특히 크게 벌어진다. golden 실측(정답 8.10 / 10.30s):
      window 중심 오차 0.600s · 절대 peak 0.351s · RMS peak 0.418s · **이 방법 0.174s**
    "배경의 N배" 같은 절대 문턱은 못 쓴다 — soft click 은 배경보다 9~15 dB 밖에
    높지 않아(golden 실측) 문턱을 넘는 구간이 잡히지 않는다.
    """
    n = 480                                    # 5ms
    r = np.sqrt(np.convolve(w * w, np.ones(n, np.float32) / n, mode="same") + 1e-12)
    return int(np.argmax(np.diff(r)))


def spec_rows(seg: np.ndarray, hop: int = None) -> np.ndarray:
    """표시 전용 log-magnitude STFT → [DISP_BINS, cols] dB (저→고).

    hop 만 바꿔 쓸 수 있다(정지 화면은 더 촘촘하게). NFFT 는 고정 — spec_floor 를
    실시간과 공유하므로 window 길이가 바뀌면 밝기 기준이 어긋난다.
    """
    n, h = C.DISP_NFFT, (C.DISP_HOP if hop is None else int(hop))
    cols = max(1, (len(seg) - n) // h + 1)
    win = np.hanning(n).astype(np.float32)
    fr = np.stack([np.abs(np.fft.rfft(seg[i * h:i * h + n] * win)) for i in range(cols)])
    freqs = np.fft.rfftfreq(n, 1 / C.SR)
    edges = np.geomspace(40, C.SR / 2, C.DISP_BINS + 1)
    img = np.zeros((C.DISP_BINS, cols), np.float32)
    centers = np.sqrt(edges[:-1] * edges[1:])
    for b in range(C.DISP_BINS):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if m.any():
            img[b] = fr[:, m].mean(1)
        else:                                  # 저역: bin 폭 < FFT 해상도 → 최근접 FFT bin 으로
            img[b] = fr[:, int(np.argmin(np.abs(freqs - centers[b])))]
    return 20 * np.log10(img + 1e-6)


def pack_spec(rows: np.ndarray) -> dict:
    """dB 격자 → uint8 b64. 고정 범위라 프레임마다 밝기가 출렁이지 않는다."""
    u8 = np.clip((rows - C.DISP_DB_LO) / (C.DISP_DB_HI - C.DISP_DB_LO), 0, 1)
    u8 = (u8 * 255).astype(np.uint8)
    return {"b64": base64.b64encode(u8.tobytes()).decode(),
            "rows": int(rows.shape[0]), "cols": int(rows.shape[1])}


def disp_spec(seg: np.ndarray, floor: np.ndarray = None, hop: int = None) -> dict:
    """표시용 스펙트로그램.

    설비 hum 같은 **정상 성분을 주파수 줄마다 빼서** 보여준다(표시 전용 — 모델 입력과 무관).
    안 빼면 저역 harmonic 이 항상 최대치로 깔려 클릭의 세로줄을 덮는다.
    floor 가 주어지면 그것으로, 없으면 이 구간의 행별 median 으로 뺀다(열이 충분할 때만).
    """
    rows = spec_rows(seg, hop)
    if floor is not None:
        rows = rows - floor[:, None]
    else:                                  # 기준 없음 → 이 구간의 하위 30% 를 기준으로(무음 경계 방어)
        rows = rows - np.percentile(rows, 30, axis=1, keepdims=True)
    return pack_spec(rows)


class CycleController(threading.Thread):
    def __init__(self, ring, win_queue, engine, atlas, emit,
                 slicer_done=None, realtime=True):
        super().__init__(daemon=True, name="cycle")
        self.ring, self.q, self.eng, self.atlas = ring, win_queue, engine, atlas
        self.emit = emit
        self.slicer_done = slicer_done         # selftest EOF barrier
        self.realtime = realtime
        self.stop_flag = threading.Event()

        self.paused = False                    # 진실. WS 핸들러가 직접 뒤집는다.
        self.detect_on = True                  # 감지 모드. False = 흐름만 보고 추론 안 함
        self.rev = 0                           # 정지/재생 전환 revision
        self.listen_gen = 0                    # '듣는 중' 세대 — pause/resume/장치전환마다 +1.
        #   predict 는 수십 ms 걸린다. 그 사이 상태가 바뀌면 낡은 결과를 세면 안 된다(Codex P0).
        self.switch_sample = 0                 # 이 sample 이전에 시작한 window 는 옛 입력 것
        self.epoch = 0                         # 듣기 세대 (resume/dial 마다 +1)
        self.auto_pause = True                 # selftest 에서 끔
        self._need_resync = False              # 재개 → latch 보존(완전 재무장은 _reset_listen
        #   을 장치전환·복구 경로에서 cycle 스레드가 직접 부른다)
        self._reqs = queue.Queue()             # 무거운 요청만: pause_snap / dial
        self._froze = False
        self._first_start = None
        self._pending_snap = None              # (target_sample, info{..., epoch})
        self.last_freeze = None
        self._ew0 = self._seg0 = 0             # CAM 정렬용 (snapshot 마다 갱신)
        self.click_total = 0
        self.count_rev = 0                     # 카운터 세대 (리셋마다 +1)
        self.eof_result = None
        self._n_pred = 0
        self._err = None
        self._last_win = time.time()
        self._rms_hist = []
        self._onsets = []
        self.session_t0 = time.time()
        self.src = None
        self.slicer = None
        from .counter import ClickCounter
        self.counter = ClickCounter()          # 연속 이벤트 집계 (cycle 개념 없음)
        self.counter.set_level(self.eng.level)
        self.spec_floor = None                 # 방의 정상 스펙트럼(행별 dB). 표시 기준을 공유한다
        self._floor_n = 0

    # ── WS(이벤트 루프)에서 직접 부른다 — bool/int 쓰기만 ───────
    def pause_now(self) -> bool:
        """즉시 정지. True=상태가 바뀜 / False=이미 정지(호출측이 현재 상태를 재전송)."""
        if self.paused:
            return False
        self.paused = True
        self.rev += 1
        self.listen_gen += 1                   # 진행 중 predict 결과 무효화
        self._pending_snap = None              # 예약된 auto freeze 취소 — 아래 manual 이
        #   화면을 잡는다(카운트는 이미 올라갔으니 손실 없다). 안 지우면 manual freeze
        #   뒤에 같은 epoch 의 auto freeze 가 한 번 더 온다.
        self._reqs.put(("pause_snap", (self.rev, self.epoch)))
        return True

    def resume_now(self) -> bool:
        if not self.paused:
            return False
        self.rev += 1
        self.epoch += 1                        # 미완 snapshot 은 epoch 로 자멸
        self.listen_gen += 1
        self.last_freeze = None
        self._pending_snap = None
        self._need_resync = True               # 카운터 latch 는 보존, 연속 high 만 초기화
        self.paused = False                    # ← 반드시 마지막. 먼저 풀면 cycle 스레드가
        #   위 필드들이 갱신되기 전의 반쪽 상태로 window 를 처리한다.
        return True

    def detect_now(self, on: bool) -> bool:
        """감지 모드 전환 — WS 핸들러가 직접 부른다. True=상태가 바뀜."""
        on = bool(on)
        if on == self.detect_on:
            return False
        self.detect_on = on
        self.epoch += 1                        # in-flight predict/snapshot 이 있어도 epoch 로 자멸
        self.listen_gen += 1
        if on:
            self.counter.reset()               # 쉰 동안의 상태는 안 이어감(총계는 유지)
        self._pending_snap = None
        self._froze = False
        return True

    def reset_count_now(self):
        self.click_total = 0
        self.count_rev += 1                    # 낡은 freeze 가 카운터·★ 를 되살리지 못하게

    def req_dial(self, level):
        self._reqs.put(("dial", level))

    def req_source_switch(self):
        """입력 장치 전환 후 — 옛/새 소리가 섞인 구간을 버리고 새 귀·새 기준으로.

        경계를 sample 로 못 박는다: 이 지점 이전에 **시작한** window 는 옛 입력이 섞여 있어
        추론에서 제외한다(Codex P0). 세대도 올려 진행 중 predict 결과를 무효화한다.
        """
        self.switch_sample = self.ring.write_idx + C.WIN_N   # 경계 이후 완전히 새 입력인 window 부터
        self.listen_gen += 1
        self._reqs.put(("source_switch", None))

    def fail(self, msg):
        self._err = msg

    # ── 메인 루프 ───────────────────────────────────────────────
    def run(self):
        while not self.stop_flag.is_set():
            self._handle_reqs()
            try:
                k, start, w = self.q.get(timeout=0.05)
            except queue.Empty:
                if self.slicer_done is not None and self.slicer_done.is_set():
                    self._finish_eof()
                    break
                self._check_snapshot()
                continue

            self._last_win = time.time()

            if self._err:
                msg, self._err = self._err, None
                self._recover(msg)
                continue
            lag_s = (self.ring.write_idx - (start + C.WIN_N)) / C.SR
            if self.realtime and not self.paused and lag_s > C.MAX_LAG_S:
                self._recover(f"pipeline lag {lag_s:.1f}s")
                continue

            # 정지(paused)는 정지다 — 추론·카운트·지표 전부 멈춘다(지표까지 멈춰야 floor·
            # RMS·density 가 정지 구간에 오염되지 않는다 — Codex P1).
            if self.paused:
                self._check_snapshot()         # 정지 직전 예약된 화면만 완성
                continue
            if start < self.switch_sample:     # 옛 입력이 섞인 window — 버린다
                continue

            self._metrics(k, w)

            if self.detect_on:                 # 감지 꺼짐 = 추론도 쉼 (CPU 절약)
                if self._need_resync:          # 재개 — latch 는 이어간다
                    self._resync_listen()
                gen = self.listen_gen          # predict 전 세대 캡처
                out = self.eng.predict(w)
                self._n_pred += 1
                if gen != self.listen_gen or self.paused:
                    continue                   # 추론 중 정지·전환됨 → 이 결과는 버린다
                ev = self.counter.push(out["probs"])
                if ev is not None:
                    self.click_total += 1
                    if self.auto_pause and self._pending_snap is None:
                        # 표시 기준 시각은 window 중심이 아니라 **소리가 시작된 지점**이다
                        t_abs = start + click_onset(w)
                        self._pending_snap = (t_abs + int(C.SNAP_POST_S * C.SR),
                                              {"ev": {"label": SPEC.label(ev["cls"]),
                                                      "t": 0.0, "score": round(ev["score"], 4)},
                                               "cls": ev["cls"],     # CAM 대상 — 표시 이름과 분리
                                               "t_abs": t_abs,
                                               "win_start": start,   # 모델이 실제로 본 window
                                               "pair": [None, None],
                                               "epoch": self.epoch})
                    else:                      # 자동정지 안 하는 모드 → 카운터만 알린다
                        self.emit({"type": "count", "click_total": self.click_total,
                                   "count_rev": self.count_rev})
            self._ambient(w)
            self._check_snapshot()

    # ── 요청 처리 (cycle 스레드) ────────────────────────────────
    def _handle_reqs(self):
        while True:
            try:
                cmd, arg = self._reqs.get_nowait()
            except queue.Empty:
                return
            try:
                if cmd == "pause_snap":
                    self._manual_snapshot(*arg)
                elif cmd == "dial":
                    self._switch_level(int(arg))
                elif cmd == "source_switch":
                    try:
                        while True:
                            self.q.get_nowait()
                    except queue.Empty:
                        pass
                    self.epoch += 1
                    self._reset_listen()
                    self.spec_floor = None      # 새 장치의 정상 스펙트럼으로 다시 학습
                    self._floor_n = 0
                    self._rms_hist = []
            except Exception as e:
                self.emit({"type": "error", "msg": f"request {cmd} failed: {e}"})

    def _switch_level(self, lvl):
        if lvl not in self.eng.variants:
            self.emit({"type": "error", "msg": f"unknown level {lvl}"})
            return
        self.eng.set_level(lvl)                # 엔진 변형(shift detector 등) 정합 유지
        self.counter.set_level(lvl)            # 실제 판정 문턱은 카운터의 tau_on
        self.epoch += 1                        # 이전 기준의 미완 snapshot 무효화
        self._pending_snap = None
        self.emit({"type": "dial", "level": lvl,
                   "counter": self.counter.state()})

    def _reset_listen(self):
        """스트림 경계(장치 전환·복구)에서만. 카운터 상태만 비우고 총계는 유지한다."""
        try:
            self.eng.reset()
        except Exception:
            pass
        self.counter.reset()
        self._need_resync = False
        self._froze = False
        self._first_start = None
        self._pending_snap = None

    def _resync_listen(self):
        """재개 경계 — 엔진 상태만 비우고 카운터 latch 는 보존한다(counter.resync 주석 참조)."""
        try:
            self.eng.reset()
        except Exception:
            pass
        self.counter.resync()
        self._need_resync = False
        self._froze = False
        self._first_start = None

    def _recover(self, msg):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self._reset_listen()
        self.emit({"type": "error", "msg": f"오디오 경고 — {msg}"})

    def _finish_eof(self):
        """selftest 전용: EOF 에서 finalize 결과를 방송."""
        if self._n_pred > 0 and self.eof_result is None:
            try:
                self.eof_result = self.eng.finalize()
            except Exception as e:
                self.emit({"type": "error", "msg": f"eof finalize failed: {e}"})
                return
            self.emit({"type": "eof_final", "pair": self.eof_result["pair"],
                       "n_windows": self._n_pred})

    # ── snapshot ────────────────────────────────────────────────
    def _snap_common(self, t_abs, win_start=None):
        s0 = max(0, t_abs - int(C.SNAP_PRE_S * C.SR))
        seg = self.ring.read_at(s0, int((C.SNAP_PRE_S + C.SNAP_POST_S) * C.SR))
        # CAM 은 **모델이 실제로 판정한 window** 를 설명해야 한다. 그래서 트리거 window
        # 의 시작을 그대로 쓴다(없으면 t_abs 중심으로 폴백).
        e0 = max(0, win_start if win_start is not None else t_abs - C.WIN_N // 2)
        ew = self.ring.read_at(e0, C.WIN_N)
        self._ew0 = e0                         # CAM 정렬용 (seg 기준 offset 계산)
        self._seg0 = s0
        if seg is None or ew is None:
            return None
        if self.eng.embed_ok:
            x, y = self.atlas.transform(self.eng.embed(ew))
            mp = {"x": x, "y": y}
        else:                                  # DAL pin 에 embed() 없음 — 지도 없이 진행
            mp = None
        floor = self._floor_db()
        peak = float(20 * np.log10(np.abs(seg).max() + 1e-9) + 100)
        return seg, ew, mp, round(max(0.0, peak - floor), 1)

    def _check_snapshot(self):
        pend = self._pending_snap              # 한 번만 읽는다 — WS 스레드가 취소할 수 있다
        if not pend:
            return
        target, info = pend
        if self.ring.write_idx < target:
            return
        self._pending_snap = None
        if info["epoch"] != self.epoch:        # resume/dial 이 끼어듦 — 이 검출은 지난 세대
            return
        try:
            got = self._snap_common(info["t_abs"], info.get("win_start"))
        except Exception:
            got = None
        if got is None:
            self._froze = False                # 재검출 허용 (snapshot 만 실패)
            return
        seg, ew, mp, snr = got
        spec = disp_spec(seg, self.spec_floor, C.SNAP_HOP)
        # CAM 은 표시 구간(seg)이 아니라 **검출 트리거 window(ew)** 를 설명한다(Codex R5).
        # 두 구간의 시작이 다르므로 sample 차이를 그대로 열 수로 환산해 옮긴다 — 예전에는
        # 0.25*cols 로 못박아 뒀는데, 이벤트 시각을 onset 으로 바꾼 뒤로는 그 값이 맞지 않는다.
        cam = None
        span = None                            # CAM 이 기여했다고 본 시간 범위(seg 상대)
        cls = info.get("cls")                  # 표시 이름(display_names)에서 역파싱하지 않는다
        if cls is not None:
            u8 = self.eng.cam(ew, cls, spec["cols"])
            if u8 is not None:
                cols = u8.shape[1]
                off = int(round((self._ew0 - self._seg0) / len(seg) * cols))
                grid = np.zeros_like(u8)
                if off >= 0:
                    if off < cols:
                        grid[:, off:] = u8[:, :cols - off]
                else:
                    if -off < cols:
                        grid[:, :cols + off] = u8[:, -off:]
                # 기여 구간(열 범위)을 함께 보낸다. 파형의 붉은 표시·SNR 화살표가
                # 이걸 쓰므로 세 표시가 같은 구간을 가리킨다.
                cmax = grid.max(axis=0)
                thr = max(int(cmax.max()), 1) * 0.55
                idx = np.flatnonzero(cmax >= thr)
                if len(idx):
                    span = [round(float(idx[0]) / cols, 4),
                            round(float(idx[-1] + 1) / cols, 4)]
                    # SNR 은 seg 전체 peak 가 아니라 **그 구간의** peak 로 잰다. 전체로
                    # 재면 클릭이 아닌 다른 큰 소리를 가리켜 화살표가 엉뚱한 곳에 섰다.
                    a0 = max(0, int(span[0] * len(seg)))
                    a1 = min(len(seg), max(a0 + 1, int(span[1] * len(seg))))
                    pk = float(20 * np.log10(np.abs(seg[a0:a1]).max() + 1e-9) + 100)
                    snr = round(max(0.0, pk - self._floor_db()), 1)
                cam = {"b64": base64.b64encode(grid.tobytes()).decode(),
                       "rows": int(grid.shape[0]), "cols": int(grid.shape[1])}
        if info["epoch"] != self.epoch:        # 무거운 일 동안 resume/dial 개입 — 폐기(Codex R5)
            return
        if not self.paused:                    # 자동 정지 (연출)
            self.paused = True
            self.rev += 1
        msg = {"type": "freeze", "manual": False, "rev": self.rev,
               "ev": info["ev"], "pair": info["pair"],
               "wav_b64": wav_b64(seg), "spec": spec, "cam": cam, "span": span,
               "map": mp, "snr_db": snr,
               "click_total": self.click_total, "count_rev": self.count_rev}
        self.last_freeze = msg
        self.emit(msg)

    def _manual_snapshot(self, rev_at, epoch_at):
        """정지 버튼 — 방금 1초를 붙잡는다. 그 사이 상태가 바뀌었으면 폐기."""
        if not self.paused or self.rev != rev_at or self.epoch != epoch_at:
            return
        if self.last_freeze is not None:       # 자동 검출 freeze 가 이미 화면을 잡음
            return
        w = self.ring.write_idx
        t_abs = max(0, w - int(C.SNAP_POST_S * C.SR))
        try:
            got = self._snap_common(t_abs)
        except Exception:
            return
        if got is None:
            return
        if not self.paused or self.rev != rev_at:   # 무거운 일 후 재확인
            return
        seg, _ew, mp, snr = got
        msg = {"type": "freeze", "manual": True, "rev": self.rev,
               "ev": None, "pair": [None, None],
               "wav_b64": wav_b64(seg), "spec": disp_spec(seg, self.spec_floor, C.SNAP_HOP),
               "map": mp, "snr_db": snr,
               "click_total": self.click_total, "count_rev": self.count_rev}
        self.last_freeze = msg
        self.emit(msg)

    # ── ambient / 지표 ──────────────────────────────────────────
    def _ambient(self, w):
        if not self.eng.embed_ok:
            return
        now = time.time()
        if now - getattr(self, "_last_amb", 0) < C.AMBIENT_EMBED_PERIOD:
            return
        self._last_amb = now
        try:
            x, y = self.atlas.transform(self.eng.embed(w))
        except Exception as e:                 # 지도만 포기하고 검출은 계속
            print(f"[cycle] WARN ambient embed failed - disabling map: {e}")
            self.eng.embed_ok = False
            return
        self.emit({"type": "ambient", "x": x, "y": y})

    def _update_floor(self, w, db):
        """1초에 한 번, 방의 정상 스펙트럼을 갱신(표시 전용).

        구간마다 median 을 다시 잡으면 무음 경계에서 기준이 튀어 화면이 통째로 포화된다
        (실측: 표시범위 밖 58%). 그래서 스트림 전체 기준을 하나 들고 공유한다.
        """
        if db < 25:                            # 무음(파일 loop 경계 등)은 기준을 오염시킨다
            return
        self._floor_n += 1
        if self._floor_n % 5:                  # hop 0.2s → 1Hz
            return
        rows = spec_rows(w)
        col_db = rows.mean(axis=0)             # 무음과 걸친 window: 무음 '열'만 골라 버린다
        ok = col_db > col_db.max() - 30        #   (안 걸러내면 p30 이 무음 열로 폭락 → 전체 포화)
        if ok.sum() < 10:
            return
        med = np.percentile(rows[:, ok], 30, axis=1)
        self.spec_floor = med if self.spec_floor is None else 0.9 * self.spec_floor + 0.1 * med

    def _metrics(self, k, w):
        db = float(20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-9) + 100)
        self._update_floor(w, db)
        self._rms_hist.append((k, db))
        self._rms_hist = self._rms_hist[-75:]
        if len(self._rms_hist) >= 2 and db - self._rms_hist[-2][1] > 6:
            self._onsets.append(time.time())
        self._onsets = [t for t in self._onsets if time.time() - t < 60]

    def _floor_db(self):
        if not self._rms_hist:
            return 0.0
        return float(np.percentile([d for _, d in self._rms_hist], 20))

    def status(self):
        latest = self._rms_hist[-1][1] if self._rms_hist else 0.0
        floor = self._floor_db()
        sl = self.slicer
        return {"paused": self.paused, "rev": self.rev, "detect_on": self.detect_on,
                "count_rev": self.count_rev,
                "state": "PAUSED" if self.paused else "LIVE",
                "session_s": int(time.time() - self.session_t0),
                "click_total": self.click_total, "floor_db": round(floor, 1),
                "level_db": round(latest, 1), "snr_db": round(max(0.0, latest - floor), 1),
                "density": len(self._onsets), "level": self.eng.level,
                "qdepth": self.q.qsize(),
                "starved_s": round(time.time() - self._last_win, 1),
                "src_alive": bool(self.src and getattr(self.src, "is_alive", lambda: True)()),
                "slicer_alive": bool(sl and sl.is_alive()),
                "counter": self.counter.state(), "listen_gen": self.listen_gen,
                "resyncs": getattr(sl, "resyncs", 0),
                "backpressure": getattr(sl, "backpressure", 0)}

    def stop(self):
        self.stop_flag.set()
