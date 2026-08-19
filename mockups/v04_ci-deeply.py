"""Render a static mockup PNG of the 1-Day Demo screen (no browser needed).

v04: three panels + DEEPLY CI. Waveform + spectrogram (shared time window) on the left,
sound map on the right. Every number that used to live in a side card is folded
into one of the three views, so the page still reads if the extras are removed.

버저닝: 아래 VERSION/SLUG/NOTE 세 줄만 바꾸고 실행하면
mockups/ 에 PNG + 그 버전을 만든 스크립트 사본 + INDEX.md 한 줄이 함께 남는다.
같은 VERSION 으로 다시 실행하면 그 버전을 덮어쓴다.
"""
import math
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CMAP = matplotlib.colormaps["viridis"]     # 스펙트로그램 컬러맵 (matplotlib 기본)

# ── 이 버전의 정체 ──────────────────────────────────────────────
VERSION = "v04"
SLUG = "ci-deeply"
NOTE = ("v03 에 DEEPLY CI 적용 — 로고 삽입, Deeply Pink(#FF3D59)=검출/이 이벤트, "
        "Deeply Black(#333132)=기존 체결음·본문. 스펙트로그램은 viridis 유지")

BASE = Path(__file__).resolve().parent
VDIR = BASE / "mockups"
VDIR.mkdir(exist_ok=True)
OUT = VDIR / f"{VERSION}_{SLUG}.png"

S = 2
W, H = 1280 * S, 720 * S

KR = "/home/myeonghoon/.local/share/fonts/NanumGothic-Regular.ttf"
KRB = "/home/myeonghoon/.local/share/fonts/NanumGothic-Bold.ttf"
MN = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MNB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
f = lambda p, s: ImageFont.truetype(p, int(s * S))

# DEEPLY CI (Nuclino > Design > CI)
PINK = "#ff3d59"          # Deeply Pink — 검출 · 이 이벤트
BLK = "#333132"           # Deeply Black — 본문 · 기존 체결음

BG, CARD, LINE = "#f3f2f3", "#ffffff", "#e7e5e6"
TX, TX2, TX3 = BLK, "#706e70", "#a8a6a8"
HOT, OK = PINK, "#22c55e"
GRAY_DOT = (201, 198, 200, 115)      # 그 밖의 소리
BLK_DOT = (51, 49, 50, 130)          # 기존 체결음
PINK_SOFT, BLK_SOFT = "#fff0f2", "#f2f1f2"

ONSET, ANLZ = 0.25, 0.56

img = Image.new("RGB", (W, H), BG)
rng = np.random.default_rng(20260817)

px = lambda *a: [int(v * S) for v in a]


def txt(dr, x, y, s, font, fill=TX, anchor="la"):
    dr.text((int(x * S), int(y * S)), s, font=font, fill=fill, anchor=anchor)


def tw(dr, s, font):
    return dr.textlength(s, font=font) / S


# 좌우 두 컬럼을 위·아래가 공유 (세로 정렬)
CARDS = {
    "wave": (18, 70, 764, 384),
    "spec": (18, 396, 764, 708),
    "map":  (776, 70, 1262, 708),
}
sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
shd = ImageDraw.Draw(sh)
for x0, y0, x1, y1 in CARDS.values():
    shd.rounded_rectangle(px(x0, y0 + 3, x1, y1 + 5), radius=int(14 * S), fill=(50, 48, 49, 32))
img = Image.alpha_composite(img.convert("RGBA"), sh.filter(ImageFilter.GaussianBlur(7 * S))).convert("RGB")
d = ImageDraw.Draw(img, "RGBA")
for x0, y0, x1, y1 in CARDS.values():
    d.rounded_rectangle(px(x0, y0, x1, y1), radius=int(14 * S), fill=CARD)


def caption(x, y, s):
    """처음 보는 사람용 한 줄 설명 — 작고 회색이 아니라 읽히는 크기로."""
    txt(d, x, y, s, f(KR, 12), "#565456")


# ═══ 헤더 (얇게) — DEEPLY CI ════════════════════════════════════
d.rectangle(px(0, 0, 1280, 58), fill=CARD)
d.rectangle(px(0, 0, 1280, 4), fill=PINK)            # 브랜드 스트립
d.line(px(0, 58, 1280, 58), fill=LINE, width=S)

logo = Image.open(BASE / "assets" / "deeply_horizontal.png").convert("RGBA")
logo = logo.crop(logo.getbbox())
LH = int(23 * S)
LW = int(logo.width * LH / logo.height)
img.paste(logo.resize((LW, LH), Image.LANCZOS), (int(26 * S), int(20 * S)), logo.resize((LW, LH), Image.LANCZOS))
d = ImageDraw.Draw(img, "RGBA")
lx0 = 26 + LW / S

d.line(px(lx0 + 22, 16, lx0 + 22, 46), fill=LINE, width=S)
d.ellipse(px(lx0 + 40, 27, lx0 + 49, 36), fill=OK)
txt(d, lx0 + 60, 18, "Scarlett 2i2 USB", f(KRB, 13), TX)
txt(d, lx0 + 60, 35, "외장 마이크 연결됨 · 48 kHz", f(KR, 9.5), TX2)

d.rounded_rectangle(px(560, 16, 756, 46), radius=int(15 * S), fill=PINK_SOFT)
d.rectangle(px(578, 24, 582, 38), fill=PINK)
d.rectangle(px(586, 24, 590, 38), fill=PINK)
txt(d, 600, 23, "정지 — 방금 잡은 소리 보는 중", f(KRB, 11), "#d3213f")

txt(d, 1262, 16, "오늘 잡은 클릭", f(KR, 10), TX3, anchor="ra")
txt(d, 1262, 28, "47", f(MNB, 23), PINK, anchor="ra")
txt(d, 1053, 24, "세션", f(KR, 10), TX3)
txt(d, 1140, 23, "00:12:34", f(MN, 12), TX2, anchor="ra")


# ═══ 1. 실시간 파형 — SNR·배경·밀도를 이 안에 흡수 ══════════════
x0, y0, x1, y1 = CARDS["wave"]
txt(d, x0 + 22, y0 + 16, "실시간 파형", f(KRB, 14), TX)
txt(d, x0 + 22 + 84, y0 + 20, "소리의 크기", f(KR, 10.5), TX3)
txt(d, x1 - 22, y0 + 20, "최근 1분 12회 감지", f(KR, 10.5), TX3, anchor="ra")

wx0, wx1 = x0 + 84, x1 - 22
wcy = y0 + 48 + (y1 - 92 - y0 - 48) / 2
wamp = (y1 - 92 - y0 - 48) / 2
n = int((wx1 - wx0) * S)

env = .050 + rng.random(n) * .040
sig = rng.normal(0, 1, n) * env
c = int(ONSET * n)
L = int(n * .30)
k = np.arange(L)
sig[c:c + L] += np.exp(-k / (L * .10)) * np.sin(k * .16) * .92
sig = np.clip(sig, -1, 1)

# 배경 소음 띠 = 눈으로 보는 SNR 기준선
bh = .135 * wamp
d.rectangle(px(wx0, wcy - bh, wx1, wcy + bh), fill=(120, 118, 119, 34))
for yy in (wcy - bh, wcy + bh):
    for xx in range(int(wx0), int(wx1), 9):
        d.line(px(xx, yy, xx + 5, yy), fill=(90, 88, 89, 125), width=S)
txt(d, x0 + 76, wcy - bh - 7, "배경 소음", f(KRB, 10), TX2, anchor="rd")
txt(d, x0 + 76, wcy - bh + 5, "54 dB", f(MN, 9.5), TX3, anchor="ra")

ax = wx0 + (wx1 - wx0) * ONSET
bx = wx0 + (wx1 - wx0) * ANLZ
for i in range(n):
    v = abs(sig[i]) * wamp
    xx = wx0 * S + i
    d.line([xx, wcy * S - v * S, xx, wcy * S + v * S],
           fill=PINK if ax * S <= xx <= bx * S else "#9c999b", width=1)

# "배경보다 이만큼 크다" = SNR 을 숫자 대신 높이로
arx = bx + 26
pk = wcy - .92 * wamp
d.line(px(arx, pk, arx, wcy - bh), fill="#d3213f", width=int(1.6 * S))
for yy, dy in ((pk, 6), (wcy - bh, -6)):
    d.polygon([px(arx - 4, yy + dy), px(arx + 4, yy + dy), px(arx, yy)], fill="#d3213f")
d.line(px(ax, pk, arx + 8, pk), fill=(255, 61, 89, 95), width=S)
txt(d, arx + 12, (pk + wcy - bh) / 2 - 16, "배경보다", f(KR, 10.5), "#d3213f")
txt(d, arx + 12, (pk + wcy - bh) / 2 - 2, "18 dB", f(MNB, 17), "#d3213f")
txt(d, arx + 12, (pk + wcy - bh) / 2 + 18, "더 큼", f(KR, 10.5), "#d3213f")

caption(x0 + 22, y1 - 76,
        "회색 띠가 이 방의 평소 소음. 분홍으로 크게 솟은 것이 방금 체결한 소리다.")
caption(x0 + 22, y1 - 54,
        "띠 위로 높이 솟을수록 잡기 쉽다 — 이 소리는 배경보다 18 dB 크다.")
d.line(px(x0 + 22, y1 - 28, x1 - 22, y1 - 28), fill=BLK_SOFT, width=S)
txt(d, x0 + 22, y1 - 22, "가로축 = 시간 1.0초  ·  아래 스펙트로그램과 같은 구간",
    f(KR, 9.5), TX3)


# ═══ 2. 스펙트로그램 — 같은 구간, 모델이 본 곳 ══════════════════
x0, y0, x1, y1 = CARDS["spec"]
txt(d, x0 + 22, y0 + 16, "스펙트로그램", f(KRB, 14), TX)
txt(d, x0 + 22 + 92, y0 + 20, "소리의 높낮이", f(KR, 10.5), TX3)
txt(d, x1 - 22, y0 + 20, "위 파형과 같은 1.0초", f(KR, 10.5), TX3, anchor="ra")

sx0, sy0 = x0 + 84, y0 + 46
sx1, sy1 = x1 - 22, y1 - 92
gw, gh = int((sx1 - sx0) * S), int((sy1 - sy0) * S)

nb, nt = 110, 300
prof = np.linspace(.06, .70, nb)[:, None] ** 1.35
base = prof * (.88 + .12 * np.sin(np.linspace(0, 6, nt)))[None, :] + rng.random((nb, nt)) * .10
for row, amp in [(int(nb * .90), .26), (int(nb * .78), .16)]:
    base[row:row + 2] += amp
cc = int(ONSET * nt)
for j in range(int(nt * .32)):
    if cc + j < nt:
        base[:, cc + j] = np.clip(base[:, cc + j] + .85 * math.exp(-j / (nt * .035)), 0, 1)
base = np.clip(base, 0, 1)
lut = (CMAP(base)[..., :3] * 255).astype(np.uint8)
spec = Image.fromarray(lut).resize((gw, gh), Image.BILINEAR)
mask = Image.new("L", (gw, gh), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, gw - 1, gh - 1], radius=int(6 * S), fill=255)
img.paste(spec, (int(sx0 * S), int(sy0 * S)), mask)
d = ImageDraw.Draw(img, "RGBA")

txt(d, x0 + 76, sy0 + 2, "높은 음", f(KR, 9.5), TX3, anchor="ra")
txt(d, x0 + 76, sy1 - 12, "낮은 음", f(KR, 9.5), TX3, anchor="ra")

ax2 = sx0 + (sx1 - sx0) * ONSET
ay2 = sy0 + (sy1 - sy0) * .44
# viridis 위에서는 주황이 죽는다 — attribution 은 흰 점선으로
for rw, rh, al in ((58, 84, 26), (38, 56, 34), (20, 30, 44)):
    d.ellipse(px(ax2 - rw * .3, ay2 - rh / 2, ax2 + rw * .9, ay2 + rh / 2), fill=(255, 255, 255, al))
ecx, ecy = ax2 + 58 * .3, ay2
erx, ery = 58 * .6, 42
for a in range(0, 360, 9):
    a0, a1 = math.radians(a), math.radians(a + 5)
    d.line(px(ecx + math.cos(a0) * erx, ecy + math.sin(a0) * ery,
              ecx + math.cos(a1) * erx, ecy + math.sin(a1) * ery),
           fill=(255, 255, 255, 225), width=int(1.6 * S))
lb = "모델이 주목한 부분"
lw = tw(d, lb, f(KRB, 10)) + 16
d.rounded_rectangle(px(ax2 + 58, ay2 - 68, ax2 + 58 + lw, ay2 - 46), radius=int(6 * S),
                    fill=(255, 255, 255, 235))
txt(d, ax2 + 66, ay2 - 63, lb, f(KRB, 10), "#1f2937")
d.line(px(ax2 + 34, ay2 - 40, ax2 + 58, ay2 - 50), fill=(255, 255, 255, 200), width=int(1.4 * S))

caption(x0 + 22, y1 - 76,
        "같은 소리를 색으로 편 그림. 노랗게 밝을수록 그 높이의 소리가 세다.")
caption(x0 + 22, y1 - 54,
        "체결음은 낮은 음부터 높은 음까지 한꺼번에 퍼져서 세로줄 하나로 보인다.")
d.line(px(x0 + 22, y1 - 28, x1 - 22, y1 - 28), fill=BLK_SOFT, width=S)
txt(d, x0 + 22, y1 - 22, "흰 점선 = 모델이 주목한 부분 (목업 — 모델 미연결)", f(KR, 9.5), TX3)

# viridis 미니 컬러바 — 색이 무슨 뜻인지 처음 보는 사람도 알게
cb1, cbw = x1 - 22, 96
txt(d, cb1, y1 - 22, "강함", f(KR, 9), TX3, anchor="ra")
cb0 = cb1 - 26 - cbw
for i2 in range(int(cbw * S)):
    r2, g2, b2 = [int(v * 255) for v in CMAP(i2 / (cbw * S))[:3]]
    d.rectangle([int(cb0 * S) + i2, int((y1 - 21) * S), int(cb0 * S) + i2 + 1, int((y1 - 13) * S)],
                fill=(r2, g2, b2))
txt(d, cb0 - 6, y1 - 22, "약함", f(KR, 9), TX3, anchor="ra")


# ═══ 3. 소리의 지도 — 판정과 성적표를 이 안에 흡수 ══════════════
x0, y0, x1, y1 = CARDS["map"]
txt(d, x0 + 22, y0 + 16, "소리의 지도", f(KRB, 14), TX)
txt(d, x0 + 22 + 84, y0 + 20, "소리끼리 얼마나 닮았나", f(KR, 10.5), TX3)

gx0, gy0 = x0 + 22, y0 + 50
gx1, gy1 = x1 - 22, y1 - 168
gw2, gh2 = gx1 - gx0, gy1 - gy0
AR = gw2 / gh2
mp = lambda u, v: ((gx0 + u * gw2) * S, (gy0 + v * gh2) * S)

CL = [(.30, .38, .135, BLK_DOT, 85, True),
      (.52, .78, .095, GRAY_DOT, 45, False),
      (.74, .46, .245, GRAY_DOT, 165, False)]
for cu, cv, r, col, cnt, near in CL:
    if near:
        cx, cy = mp(cu, cv)
        for m, al in ((1.0, 80), (1.75, 32)):
            rad = r * gw2 * m * S
            for a in range(0, 360, 10):
                a0, a1 = math.radians(a), math.radians(a + 5.5)
                d.line([cx + math.cos(a0) * rad, cy + math.sin(a0) * rad,
                        cx + math.cos(a1) * rad, cy + math.sin(a1) * rad],
                       fill=(51, 49, 50, al), width=int(1.3 * S))
    for _ in range(cnt):
        a, dd = rng.uniform(0, 6.28), math.sqrt(rng.random()) * r
        cx, cy = mp(cu + math.cos(a) * dd, cv + math.sin(a) * dd * AR)
        d.ellipse([cx - 2.7 * S, cy - 2.7 * S, cx + 2.7 * S, cy + 2.7 * S], fill=col)

txt(d, *[v / S for v in mp(.30, .16)], "기존 체결음", f(KRB, 11), BLK, anchor="mm")
txt(d, *[v / S for v in mp(.80, .17)], "그 밖의 소리", f(KRB, 11), TX3, anchor="mm")

sx, sy = mp(.335, .445)
for rr, al in ((26, 60), (17, 105)):
    d.ellipse([sx - rr * S, sy - rr * S, sx + rr * S, sy + rr * S],
              outline=(255, 61, 89, al), width=int(2 * S))
star = []
for i in range(10):
    a = -math.pi / 2 + i * math.pi / 5
    rr = (4.2 if i % 2 else 10.5) * S
    star.append((sx + math.cos(a) * rr, sy + math.sin(a) * rr))
d.polygon(star, fill=HOT)

cxx, cyy = sx / S + 30, sy / S - 42
d.line(px(sx / S + 13, sy / S - 10, cxx - 4, cyy + 14), fill=(255, 61, 89, 150), width=int(1.4 * S))
l1, l2 = "★ 방금 그 소리", "기존 체결음 무리 안에 들어왔다"
bw2 = max(tw(d, l1, f(KRB, 11)), tw(d, l2, f(KR, 10.5))) + 20
d.rounded_rectangle(px(cxx, cyy, cxx + bw2, cyy + 44), radius=int(8 * S), fill=PINK_SOFT)
txt(d, cxx + 10, cyy + 6, l1, f(KRB, 11), "#d3213f")
txt(d, cxx + 10, cyy + 24, l2, f(KR, 10.5), "#b8663f")

caption(x0 + 22, y1 - 152,
        "닮은 소리끼리 가까이 모아 놓은 지도. 회색은 이 방의 온갖 소음이다.")

# 성적표 — 별도 박스 대신 지도 아래 스트립
d.line(px(x0 + 22, y1 - 122, x1 - 22, y1 - 122), fill=BLK_SOFT, width=S)
txt(d, x0 + 22, y1 - 110, "이번 자리에서 해본 시험", f(KRB, 11.5), TX)
for i, (t1, big, t2, col, bgc) in enumerate([
        ("체결해 봤을 때", "8번 중 7번", "잡았다", PINK, PINK_SOFT),
        ("헷갈릴 소리를 냈을 때", "4번 중 0번", "잘못 잡았다", BLK, BLK_SOFT)]):
    bxx = x0 + 22 + i * 226
    d.rounded_rectangle(px(bxx, y1 - 90, bxx + 214, y1 - 24), radius=int(10 * S), fill=bgc)
    txt(d, bxx + 16, y1 - 80, t1, f(KR, 10), TX2)
    txt(d, bxx + 16, y1 - 64, big, f(KRB, 17), col)
    txt(d, bxx + 16, y1 - 42, t2, f(KR, 10), TX3)

img.resize((W // S, H // S), Image.LANCZOS).save(OUT)
shutil.copy(OUT, BASE / "mockup.png")
shutil.copy(__file__, VDIR / f"{VERSION}_{SLUG}.py")

idx = VDIR / "INDEX.md"
head = ["# 1-Day Demo 목업 버전 기록", "",
        "각 버전은 PNG + 그 PNG 를 만든 스크립트가 짝으로 남아 있다.", ""]
old = idx.read_text(encoding="utf-8").splitlines() if idx.exists() else head
rows = [l for l in old if l.startswith("- **") and not l.startswith(f"- **{VERSION}**")]
rows.append(f"- **{VERSION}** [`{VERSION}_{SLUG}.png`]({VERSION}_{SLUG}.png) — {NOTE}")
idx.write_text("\n".join(head + sorted(rows)) + "\n", encoding="utf-8")

print(f"saved : {OUT}  ({W // S}x{H // S})")
print(f"script: {VDIR / f'{VERSION}_{SLUG}.py'}")
print(f"index : {idx}")
