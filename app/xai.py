"""freeze 화면용 GradCAM — "클릭 판정에 기여한 영역(근사)".

DAL models/xai/GradCAM 을 수정 없이 그대로 쓰되, wav 가 아니라 mel 입력 어댑터
(_SpecHead)에 물린다. fold0(대표 fold) 하나만 설명한다 — 5-fold 앙상블 판정
전체의 인과가 아니므로 화면 문구도 "근사 · 대표 fold" 로 명시한다.
DyMN 의 dynamic conv 는 전역 문맥이 섞이므로 위치의 인과가 아니라 기여의 근사다.
"""
import numpy as np
import torch

from . import config as C

TARGET_SUFFIX = "backbone.out_c.0"      # activation [960, 4, 13].
# 실측(golden c1/c2, 2026-08-18): layers.11(8×26)은 시간축이 경계·엉뚱한 곳을 짚었고
# out_c 계열이 일관되게 나았다. out_c.0(pre-BN)과 out_c(post-act)는 결과 동일 실측.
# class logit vs (click−others) contrast 도 동일 → 단순한 class logit 사용.
# 주의: 근사 진실로 쓴 seg 내 RMS-argmax 는 진폭 기준이라 soft click 에선 빗나갈 수 있다
# — 최종 검증은 체결 타이밍을 아는 실기 리허설에서 한다.
MIN_FOLD0_PROB = 0.4                    # 대표 fold 확신이 이보다 낮으면 CAM 표시 안 함(과장 방지)


class _SpecHead(torch.nn.Module):
    """mel [B,1,F,T] → logits. DAL GradCAM 이 요구하는 forward(spec) 형태."""

    def __init__(self, infer):
        super().__init__()
        self.infer = infer

    def forward(self, mel):
        return self.infer.head(self.infer._extract_from_mel(mel))


def _mel_centers_hz(n_mels, f_max, f_min=0.0):
    """HTK mel 스케일의 각 밴드 중심 주파수."""
    m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    inv = lambda x: 700.0 * (10.0 ** (x / 2595.0) - 1.0)
    pts = np.linspace(m(f_min), m(f_max), n_mels + 2)
    return inv(pts[1:-1])


class ClickCAM:
    def __init__(self, infer):
        from DAL.models.xai.gradcam import GradCAM
        self.infer = infer
        target = None
        last_conv = None
        for n, mod in infer.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                last_conv = mod
            if n.endswith(TARGET_SUFFIX):
                target = mod
        if target is None:                  # 백업: 마지막 Conv2d
            target = last_conv
        if target is None:
            raise RuntimeError("no conv layer found for GradCAM")
        self.gc = GradCAM(_SpecHead(infer), target)

        # mel 행(모델 축) → 표시 스펙트로그램 행(geomspace) 매핑.
        # 단순 row-resize 금지(Codex R4) — mel 중심주파수 기준으로 재배치한다.
        n_mels = 128
        centers = _mel_centers_hz(n_mels, C.SR / 2)
        edges = np.geomspace(40, C.SR / 2, C.DISP_BINS + 1)
        rows = np.searchsorted(edges, centers) - 1
        rows[centers < edges[0]] = -1       # 40Hz 미만은 표시 밖
        self.row_of_mel = np.clip(rows, -1, C.DISP_BINS - 1)

    @torch.enable_grad()
    def explain(self, window: np.ndarray, cls: int, disp_cols: int) -> np.ndarray:
        """1s window + 클래스(1=c1, 2=c2) → 표시 격자 uint8 [DISP_BINS, disp_cols].

        시간축은 양쪽 다 같은 1s 구간이라 선형 리샘플, 주파수축은 mel 중심 기준.
        비어 있는 표시 행은 0 으로 둔다(이웃 보간으로 과장하지 않는다).
        """
        dev = next(self.infer.head.parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(window)).float().view(1, 1, -1).to(dev)
        try:
            mel = self.infer.preprocessor(x)                   # [1,1,128,T]
            with torch.no_grad():                              # 대표 fold 확신 게이트
                p = torch.softmax(self.gc.model(mel), -1)[0, int(cls)].item()
            if p < MIN_FOLD0_PROB:
                return None
            cam = self.gc.generate(mel, target_class=int(cls))  # [1,128,T] (0..1)
            cam = cam[0].detach().cpu().numpy()
        finally:                                               # fold0 grad ~35MB 즉시 반환
            self.gc.model.zero_grad(set_to_none=True)

        t_idx = np.linspace(0, cam.shape[1] - 1, disp_cols).round().astype(int)
        cam_t = cam[:, t_idx]                                  # [128, cols]
        out = np.zeros((C.DISP_BINS, disp_cols), np.float32)
        cnt = np.zeros(C.DISP_BINS, np.float32)
        for mel_i, row in enumerate(self.row_of_mel):
            if row < 0:
                continue
            out[row] += cam_t[mel_i]
            cnt[row] += 1
        cnt[cnt == 0] = 1
        out /= cnt[:, None]
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
