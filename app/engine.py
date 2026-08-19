"""모델 래퍼 — **현재 엔진은 대체(임시) 모델이다.**

붙어 있는 것은 다른 공정용으로 학습된 DAL hdtransys v4_e20 이고, day1 데모 전용 모델이
따로 준비되면 이 파일이 교체 지점이다. 엔진을 순수 1초 window 분류기로만 쓰기 때문
(이벤트 집계는 app/counter.py) 교체 시 바꿀 것은 아래 셋뿐이다:
  ① import·load 하는 classifier  ② predict 가 돌려주는 probs 의 클래스 순서(others=0 전제)
  ③ config 의 SR/WIN_S/HOP_S 와 판정 문턱 기본값
화면·조작·연속 카운터 방식은 그대로 간다.

3단(-1/0/+1) 변형을 weight 공유로 미리 만들어 둔다.

predict/embed/finalize 는 전부 engine 스레드(=CycleController)에서만 부른다(직렬화).
level 변형은 base 의 models 리스트를 그대로 공유하므로 메모리·로딩 추가비용이 없다.
merge 로직은 DAL infer.py load(strictness=) 와 동일 규칙(tau 계열만 이동, D/min_gap 불변).
"""
import json
import time

import numpy as np
import torch

from . import config as C


def level_params(mani: dict, version: str, hop: float, level: int):
    """manifest → (postp dict, immediate dict). infer.py 의 strictness 병합을 그대로 재현."""
    ver = mani["versions"][version]
    postp = dict(ver["postp"][f"{hop:g}"])
    imm = dict(ver["immediate"])
    if level != 0:
        e = ver["strictness_map"][str(int(level))]
        postp.update({k: v for k, v in e.items() if k in ("tau_low", "rescue_p", "rescue_z")})
        imm.update({"tau_hi_c1": e["tau_hi_c1"], "tau_hi_c2": e["tau_hi_c2"]})
    return postp, imm


class Engine:
    def __init__(self, folds=None, device=None, pair_postp=None):
        from DAL.inference.hdtransys import HDTransysClassifier
        self._cls = HDTransysClassifier
        self.pair_postp = C.PAIR_POSTP if pair_postp is None else bool(pair_postp)

        self.mani = json.loads((C.WEIGHTS_DIR / "manifest.json").read_text())
        self.version = self.mani["default_version"]
        self.hop = float(self.mani.get("default_hop", C.HOP_S))

        # postp off 면 immediate 까지 끈다 → 엔진은 순수 1초 window 분류기.
        # 이벤트 집계는 app/counter.py(연속 hysteresis)가 한다.
        pk = None if self.pair_postp else False
        im = None if self.pair_postp else False
        base = HDTransysClassifier.load(weights_dir=C.WEIGHTS_DIR, device=device, folds=folds,
                                        postp=pk, immediate=im, save=False)
        if folds is None and self._bench(base) > C.BENCH_BUDGET_MS:
            for n in (2, 1):
                base = HDTransysClassifier.load(weights_dir=C.WEIGHTS_DIR, device=device,
                                                folds=n, postp=pk, immediate=im, save=False)
                if self._bench(base) <= C.BENCH_BUDGET_MS or n == 1:
                    break
        print(f"[engine] postp={'pair (cycle)' if self.pair_postp else 'off - window classifier'}")
        self.folds = len(base.models)
        self.p95_ms = self._bench(base, runs=10)
        print(f"[engine] version={self.version} folds={self.folds} p95={self.p95_ms:.0f}ms "
              f"(budget {C.BENCH_BUDGET_MS}ms)")

        # shift baseline 은 모든 변형에 동일 적용 (load 와 같은 규칙)
        bpath = C.WEIGHTS_DIR / f"baseline_{self.version}.json"
        baseline = json.loads(bpath.read_text()) if bpath.exists() else None
        algo = self.mani["versions"][self.version].get("postp_algo")
        self.algo = algo or "pair_v1"

        self.variants = {}
        for lvl in C.LEVELS:
            if lvl == 0:
                self.variants[0] = base
                continue
            postp, imm = level_params(self.mani, self.version, self.hop, lvl)
            self.variants[lvl] = HDTransysClassifier(
                models=base.models, class_names=base.class_names, channel=base.channel,
                device=base.device, window_sec=base.window_sec, hop_sec=base.hop_sec,
                postp=(postp if self.pair_postp else None), postp_algo=algo,
                immediate=(imm if self.pair_postp else None),
                shift_baseline=baseline, save=False,
            )
        self.level = C.DEFAULT_LEVEL

        # DAL pin 에 embed()(2026-08-17 add-only 추가분)가 없으면 지도 기능만 끄고 돈다.
        self.embed_ok = hasattr(self.variants[0], "embed") and hasattr(
            self.variants[0].models[0], "forward_with_embedding")
        if not self.embed_ok:
            print("[engine] WARN DAL checkout lacks embed() - sound map disabled. "
                  "Commit the DAL embed patch and update the pin in setup.sh.")

        # GradCAM (fold0 = 대표 fold). 실패해도 데모는 계속 — CAM 표시만 꺼진다.
        try:
            from .xai import ClickCAM, TARGET_SUFFIX
            self._cam = ClickCAM(base.models[0])
            self._cam.explain(np.zeros(C.WIN_N, np.float32), 1, 8)   # 워밍업(CUDA 첫 backward ~1.3s)
            print(f"[engine] gradcam ready (fold0, target {TARGET_SUFFIX})")
        except Exception as e:
            self._cam = None
            print(f"[engine] WARN gradcam unavailable: {e}")

    # ── 라이브 경로 (engine 스레드 전용) ────────────────────────
    @property
    def clf(self):
        return self.variants[self.level]

    def set_level(self, level: int):
        assert level in C.LEVELS, f"unknown level {level}"
        self.level = int(level)

    def reset(self):
        self.clf.reset()

    def predict(self, window: np.ndarray) -> dict:
        return self.clf.predict(torch.from_numpy(np.ascontiguousarray(window)))

    def finalize(self) -> dict:
        return self.clf.finalize()

    def embed(self, window: np.ndarray) -> np.ndarray:
        return self.variants[0].embed(torch.from_numpy(np.ascontiguousarray(window)))

    def cam(self, window: np.ndarray, cls: int, disp_cols: int):
        """검출 클래스 기여영역(근사) uint8 [DISP_BINS, cols]. 실패 시 None + 자기 비활성."""
        if self._cam is None:
            return None
        try:
            return self._cam.explain(window, cls, disp_cols)
        except Exception as e:
            print(f"[engine] WARN gradcam failed - disabling: {e}")
            try:
                self._cam.gc.reset()           # hook leak 방지 (Codex R5)
            except Exception:
                pass
            self._cam = None
            return None

    # ── 유틸 ────────────────────────────────────────────────────
    def _bench(self, clf, runs=6) -> float:
        wav = torch.zeros(1, C.WIN_N)
        for _ in range(2):
            clf.predict(wav)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            clf.predict(wav)
            ts.append((time.perf_counter() - t0) * 1e3)
        clf.reset()
        return float(np.percentile(ts, 95))

    def all_level_params(self):
        return {lvl: level_params(self.mani, self.version, self.hop, lvl) for lvl in C.LEVELS}
