"""모델 교체 계약 — **전용 모델로 갈아탈 때 손댈 곳은 여기 하나다.**

현재 엔진(DAL hdtransys v4_e20)은 **대체(임시) 모델**이다. 교체 시 코드 곳곳에 흩어진
가정을 찾아다니지 않도록, 모델에 의존하는 사실을 이 파일에 모은다.
startup 에서 spec.validate() 가 실제 모델과 대조해 **어긋나면 즉시 죽는다**(조용히
틀린 화면을 보여주는 것보다 낫다).
"""
from dataclasses import dataclass, field

from . import config as C


@dataclass
class ModelSpec:
    name: str = "hdtransys v4_e20 (placeholder)"
    sr: int = C.SR                     # 입력 샘플레이트
    win_s: float = C.WIN_S             # window 길이(초)
    hop_s: float = C.HOP_S             # 호출 간격(초)
    others_index: int = 0              # '아무것도 아님' 클래스 위치 — counter 의 p_click 기준
    class_names: tuple = ("others", "1", "2")        # ckpt 실측값 (표시는 display_names)
    display_names: tuple = ("others", "클릭 1", "클릭 2")
    #   클릭으로 취급할 클래스(= others 를 뺀 전부). 이름만 표시에 쓰고 카운트는 클래스 무관.
    cam_target_suffix: str = "backbone.out_c.0"     # GradCAM 대상 층 이름 꼬리
    cam_n_mels: int = 128                            # CAM 축 매핑용 mel 밴드 수
    embed_dim: int = 960                             # 지도 좌표용 embedding 차원
    placeholder: bool = True                         # 전용 모델로 바뀌면 False

    @property
    def click_indices(self):
        return tuple(i for i in range(len(self.class_names)) if i != self.others_index)

    def label(self, idx: int) -> str:
        """화면 표시용 이름."""
        if 0 <= idx < len(self.display_names):
            return self.display_names[idx]
        return f"class_{idx}"

    def validate(self, clf) -> list:
        """실제 모델과 대조. 치명 불일치는 예외, 경미한 것은 경고 목록으로 반환."""
        warn = []
        names = list(getattr(clf, "class_names", []) or [])
        if names:
            if names[self.others_index] != self.class_names[self.others_index]:
                raise RuntimeError(
                    f"ModelSpec mismatch: others_index={self.others_index} 는 "
                    f"'{self.class_names[self.others_index]}' 여야 하는데 모델은 '{names[self.others_index]}' 다. "
                    "counter 의 p_click = 1 - p_others 가 깨진다.")
            if tuple(names) != tuple(self.class_names):
                warn.append(f"class_names 다름: spec={self.class_names} model={tuple(names)}")
        for attr, want, what in (("window_sec", self.win_s, "window"),
                                 ("hop_sec", self.hop_s, "hop")):
            got = getattr(clf, attr, None)
            if got is not None and abs(float(got) - want) > 1e-6:
                raise RuntimeError(f"ModelSpec mismatch: {what} spec={want}s model={got}s")
        cfg = getattr(clf.models[0], "config", None) or {}
        got_sr = cfg.get("sample_rate")
        if got_sr and int(got_sr) != self.sr:
            raise RuntimeError(
                f"ModelSpec mismatch: sample_rate spec={self.sr} model={got_sr} "
                "(리샘플로 때우지 않는다 — config.SR 을 모델에 맞춰라)")
        return warn


SPEC = ModelSpec()
