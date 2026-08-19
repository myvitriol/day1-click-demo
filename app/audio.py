"""오디오 소스(mic/file) → RingBuffer → 절대 sample index 기준 window slicer.

시각의 진실은 전부 '스트림 시작 이후 누적 sample 수'다 — wallclock 을 쓰지 않는다.
window k = [k*HOP_N, k*HOP_N + WIN_N). 완전히 써진 뒤에만 낸다.
"""
import queue
import threading
import time

import numpy as np

from . import config as C


class RingOverrun(RuntimeError):
    """읽기 전에 데이터가 덮어써짐 — 파이프라인이 실시간을 못 따라간 것."""


class RingBuffer:
    def __init__(self, capacity: int):
        self.buf = np.zeros(capacity, np.float32)
        self.cap = capacity
        self.write_idx = 0                     # 절대 누적 sample 수
        self._lock = threading.Lock()

    def write(self, block: np.ndarray):
        block = np.asarray(block, np.float32).reshape(-1)
        with self._lock:
            n = len(block)
            i = self.write_idx % self.cap
            e = min(n, self.cap - i)
            self.buf[i:i + e] = block[:e]
            if n > e:
                self.buf[: n - e] = block[e:]
            self.write_idx += n

    def read_at(self, start: int, n: int):
        """[start, start+n) 절대 구간. 미도래→None, 덮어써짐→RingOverrun."""
        out = np.empty(n, np.float32)          # 할당은 lock 밖 — 캡처 콜백 경합 최소화
        with self._lock:
            w = self.write_idx
            if start + n > w:
                return None
            if start < w - self.cap:
                raise RingOverrun(f"ring overrun: start={start} < {w - self.cap}")
            i = start % self.cap
            e = min(n, self.cap - i)
            out[:e] = self.buf[i:i + e]
            if n > e:
                out[e:] = self.buf[: n - e]
        return out


class FileSource(threading.Thread):
    """flac/wav → ring. 실시간 페이싱(기본) 또는 fast(무페이싱). loop 시 사이에 1s 무음."""

    def __init__(self, ring: RingBuffer, path, fast=False, loop=False, on_error=None):
        super().__init__(daemon=True, name="file-source")
        import torchaudio                       # DAL 의존성에 이미 포함
        wav, sr = torchaudio.load(str(path))    # [C, L]
        if sr != C.SR:
            raise RuntimeError(f"file sample rate {sr} != {C.SR} (resampling is forbidden)")
        self.data = wav[0].numpy().astype(np.float32)   # ch0
        self.ring, self.fast, self.loop = ring, fast, loop
        self.eof = threading.Event()
        self.stop_flag = threading.Event()
        self.on_error = on_error
        self.name_str = f"file:{path}"

    def run(self):
        try:
            while not self.stop_flag.is_set():
                for i in range(0, len(self.data), C.FILE_BLOCK):
                    if self.stop_flag.is_set():
                        return
                    self.ring.write(self.data[i:i + C.FILE_BLOCK])
                    if not self.fast:
                        time.sleep(C.FILE_BLOCK / C.SR)
                if not self.loop:
                    break
                for _ in range(C.SR // C.FILE_BLOCK):        # cycle 사이 1s 무음(페이싱 유지)
                    if self.stop_flag.is_set():
                        return
                    self.ring.write(np.zeros(C.FILE_BLOCK, np.float32))
                    if not self.fast:
                        time.sleep(C.FILE_BLOCK / C.SR)
        except Exception as e:                                # crash-proof: 소스 죽음 → 상위 통지
            if self.on_error:
                self.on_error(f"file source failed: {e}")
        finally:
            self.eof.set()

    def stop(self):
        self.stop_flag.set()

    def info(self):
        return {"kind": "file", "name": self.name_str, "sr": C.SR, "channels": 1}


def list_inputs():
    """입력 장치 목록 [{index,name,channels,default,hint}] — sounddevice 없으면 []."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    out = []
    try:
        default_in = sd.default.device[0]
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] <= 0:
                continue
            out.append({"index": i, "name": str(d["name"]),
                        "channels": int(d["max_input_channels"]),
                        "default": i == default_in,
                        "hint": C.DEVICE_HINT.lower() in d["name"].lower()})
    except Exception:
        return []
    return out


class NoSource:
    """입력 미선택 상태. 서버는 뜨고 화면은 열리며, 사용자가 드롭다운에서 고를 때까지 조용하다.

    데모 당일 흐름: run.sh → 브라우저 열림 → 마이크 선택 → 시작.
    마이크가 96kHz 로 안 열려도 서버가 죽지 않아야 이 흐름이 성립한다.
    """

    def __init__(self, reason=""):
        self.eof = threading.Event()
        self.reason = reason

    def start(self):
        pass

    def stop(self):
        pass

    def is_alive(self):
        return False

    def info(self):
        return {"kind": "none", "name": "입력을 선택하세요", "sr": C.SR,
                "channels": 0, "index": -1, "reason": self.reason}


class MicSource:
    """sounddevice 캡처. 96k 로 실제 열렸는지 확인하고 아니면 시작 거부."""

    def __init__(self, ring: RingBuffer, on_error=None, device=None):
        import sounddevice as sd                # lazy — 파일 소스만 쓰는 서버엔 없어도 됨
        self.sd = sd
        self.ring = ring
        self.on_error = on_error
        self.eof = threading.Event()            # mic 은 EOF 없음(인터페이스 통일용)
        dev = self._pick_device() if device is None else int(device)
        self.dev_info = sd.query_devices(dev, "input")
        extra = None
        import sys
        if sys.platform == "darwin":            # CoreAudio: 장치 rate 를 96k 로 실제 변경 요구
            extra = sd.CoreAudioSettings(change_device_parameters=True)
        # 항상 mono(1ch) — 어차피 ch0 만 쓰고, doctor 의 1ch 시험과도 일치한다 (Codex R7)
        self.stream = sd.InputStream(
            device=dev, samplerate=C.SR, channels=1, dtype="float32",
            blocksize=0, callback=self._cb, extra_settings=extra,   # blocksize=0: CoreAudio 가변
        )
        if int(self.stream.samplerate) != C.SR:
            try:
                self.stream.close()             # 실패한 스트림을 물고 있지 않는다
            except Exception:
                pass
            raise RuntimeError(
                f"device opened at {self.stream.samplerate} Hz, need {C.SR} Hz - refusing to start")

    def _pick_device(self):
        devs = self.sd.query_devices()
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and C.DEVICE_HINT.lower() in d["name"].lower():
                return i
        print(f"WARN: no input device matching '{C.DEVICE_HINT}' - using system default. "
              "Check the interface cable / mic permission before the demo.")
        return self.sd.default.device[0]        # 화면 헤더에 실제 장치명이 표시된다

    def _cb(self, indata, frames, t, status):
        # 콜백은 복사만 한다. 무거운 일 금지.
        if status and status.input_overflow:
            if self.on_error:
                self.on_error("input overflow (capture)")
        self.ring.write(np.ascontiguousarray(indata[:, 0]))

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass

    def is_alive(self):
        try:
            return bool(self.stream.active)
        except Exception:
            return False

    def info(self):
        return {"kind": "mic", "name": str(self.dev_info["name"]),
                "sr": int(self.stream.samplerate), "channels": int(self.stream.channels),
                "index": int(self.stream.device) if not isinstance(self.stream.device, tuple)
                         else int(self.stream.device[0])}


class WindowSlicer(threading.Thread):
    """ring → (k, start, window[WIN_N]) 를 큐에. 완성된 window 만.

    이 스레드는 **죽지 않는다**. 뒤처져서 ring 이 덮어써졌으면(overrun) 예외로 끝내는 대신
    live edge 로 재동기하고 계속 간다 — 라이브 데모에서 슬라이서가 조용히 죽으면 화면은
    흐르는데 검출만 영구 정지하는(진단 불가) 상태가 된다.
    """

    def __init__(self, ring: RingBuffer, out_queue, source_eof: threading.Event, on_error=None):
        super().__init__(daemon=True, name="slicer")
        self.ring, self.q, self.src_eof = ring, out_queue, source_eof
        self.stop_flag = threading.Event()
        self.done = threading.Event()           # 진짜 EOF 까지 전부 냈음(selftest barrier)
        self.on_error = on_error
        self.k = 0
        self.last_start = 0
        self.resyncs = 0                        # overrun 재동기 횟수 (status 로 노출)
        self.backpressure = 0                   # 큐 가득으로 대기한 횟수

    def _resync(self, why):
        """live edge 직전으로 k 를 점프. 건너뛴 구간은 포기한다(그 trial 은 무효 처리됨)."""
        live = self.ring.write_idx
        self.k = max(self.k, (live - C.WIN_N) // C.HOP_N + 1)
        self.resyncs += 1
        if self.on_error:
            self.on_error(f"{why} - resynced to live edge (#{self.resyncs})")

    def run(self):
        try:
            while not self.stop_flag.is_set():
                start = self.k * C.HOP_N
                try:
                    w = self.ring.read_at(start, C.WIN_N)
                except RingOverrun as e:
                    self._resync(str(e))
                    continue
                except Exception as e:          # 예상 밖 오류에도 멈추지 않는다
                    self._resync(f"slicer read failed: {e}")
                    time.sleep(0.05)
                    continue
                if w is None:
                    if self.src_eof.is_set() and self.ring.write_idx < start + C.WIN_N:
                        break                    # 소스 끝 — 더는 완성될 window 가 없다
                    time.sleep(0.005)
                    continue
                while not self.stop_flag.is_set():
                    try:
                        self.q.put((self.k, start, w), timeout=0.5)
                        break
                    except queue.Full:
                        self.backpressure += 1   # 소비자가 느림 — 대기하되 영구 블록은 안 한다
                self.last_start = start
                self.k += 1
        finally:
            self.done.set()

    def stop(self):
        self.stop_flag.set()
