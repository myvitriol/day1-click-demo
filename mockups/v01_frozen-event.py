"""Render a static mockup PNG of the 1-Day Demo screen (no browser needed).

State shown: FROZEN — a click was detected and the three main views are held
on that single event until the presenter resumes.

버저닝: 아래 VERSION/SLUG/NOTE 세 줄만 바꾸고 실행하면
mockups/ 에 PNG + 그 버전을 만든 스크립트 사본 + INDEX.md 한 줄이 함께 남는다.
같은 VERSION 으로 다시 실행하면 그 버전을 덮어쓴다.
"""
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── 이 버전의 정체 ──────────────────────────────────────────────
VERSION = "v01"
SLUG = "frozen-event"
NOTE = ("정지(freeze) 상태 · 파형 60% + 지도 40% 나란히 · 스펙트로그램 축소 · "
        "검출 로그를 버리고 '이 이벤트 판정 + 시험 성적표'로 교체")

BASE = Path(__file__).resolve().parent
VDIR = BASE / "mockups"
VDIR.mkdir(exist_ok=True)
OUT = VDIR / f"{VERSION}_{SLUG}.png"

S = 2                       # supersample scale
W, H = 1280 * S, 720 * S

KR = "/home/myeonghoon/.local/share/fonts/NanumGothic-Regular.ttf"
KRB = "/home/myeonghoon/.local/share/fonts/NanumGothic-Bold.ttf"
MN = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MNB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
f = lambda p, s: ImageFont.truetype(p, int(s * S))

BG, CARD, LINE = "#eef2f7", "#ffffff", "#e3e9f0"
TX, TX2, TX3 = "#0f172a", "#64748b", "#9aa8bb"
ACC, HOT, OK = "#00bfa6", "#ff7a3d", "#22c55e"
SOFT = "#f7f9fc"

ONSET = 0.25          # 정지 화면의 시간창 안에서 클릭이 놓인 위치 (pre-roll 250ms)
ANLZ = 0.56           # 분석 구간 끝

img = Image.new("RGB", (W, H), BG)
rng = np.random.default_rng(20260817)


def px(*a):
    return [int(v * S) for v in a]


def txt(dr, x, y, s, font, fill=TX, anchor="la"):
    dr.text((int(x * S), int(y * S)), s, font=font, fill=fill, anchor=anchor)


def tw(dr, s, font):
    return dr.textlength(s, font=font) / S


# ── 세로 정렬: 좌우 두 컬럼(18–764 / 776–1262)을 위·아래 모두 공유 ──
CARDS = {
    "wave":  (18, 80, 764, 368),
    "map":   (776, 80, 1262, 368),
    "stat":  (18, 380, 385, 702),
    "spec":  (397, 380, 764, 702),
    "judge": (776, 380, 1262, 702),
}
sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
shd = ImageDraw.Draw(sh)
for x0, y0, x1, y1 in CARDS.values():
    shd.rounded_rectangle(px(x0, y0 + 3, x1, y1 + 5), radius=int(14 * S), fill=(30, 55, 95, 34))
img = Image.alpha_composite(img.convert("RGBA"), sh.filter(ImageFilter.GaussianBlur(7 * S))).convert("RGB")
d = ImageDraw.Draw(img, "RGBA")
for x0, y0, x1, y1 in CARDS.values():
    d.rounded_rectangle(px(x0, y0, x1, y1), radius=int(14 * S), fill=CARD)


def head(key, title, right=None, rcol=TX3):
    x0, y0, x1, y1 = CARDS[key]
    txt(d, x0 + 22, y0 + 16, title, f(KRB, 13), TX)
    if right:
        txt(d, x1 - 22, y0 + 19, right, f(KR, 10), rcol, anchor="ra")
    return x0, y0, x1, y1


# ═══ 헤더 ═══════════════════════════════════════════════════════
d.rectangle(px(0, 0, 1280, 62), fill=CARD)
d.line(px(0, 62, 1280, 62), fill=LINE, width=S)

d.ellipse(px(26, 27, 36, 37), fill=OK)
d.ellipse(px(21, 22, 41, 42), outline=(34, 197, 94, 70), width=int(1.5 * S))
txt(d, 52, 16, "Scarlett 2i2 USB", f(KRB, 14), TX)
txt(d, 52, 36, "외장 입력 자동 선택됨  ·  48 kHz · 1ch  ·  AGC/NS/EC off", f(KR, 10), TX2)

# 상태 배지 — 정지 중임을 헤더가 계속 알려준다
bw = 232
d.rounded_rectangle(px(470, 15, 470 + bw, 47), radius=int(16 * S), fill="#fff3ea")
d.rectangle(px(492, 24, 496, 38), fill=HOT)
d.rectangle(px(500, 24, 504, 38), fill=HOT)
txt(d, 514, 20, "이벤트 검토 중", f(KRB, 12), "#d9612a")
txt(d, 514, 34, "14:02:31 · 검출 #47", f(KR, 9.5), "#c47a53")

txt(d, 1000, 20, "세션", f(KR, 10), TX3)
txt(d, 1000, 33, "00:12:34", f(MNB, 15), TX2)
txt(d, 1262, 18, "오늘 잡은 클릭", f(KR, 10), TX3, anchor="ra")
txt(d, 1262, 30, "47", f(MNB, 25), HOT, anchor="ra")
d.line(px(975, 16, 975, 46), fill=LINE, width=S)
d.line(px(1130, 16, 1130, 46), fill=LINE, width=S)


# ═══ 1. 실시간 파형 — 정지, 이벤트 구간 확대 ════════════════════
x0, y0, x1, y1 = head("wave", "실시간 파형", "정지 · 이벤트 #47 구간 1.0초", HOT)
wx0, wx1 = x0 + 22, x1 - 22
wcy, wamp = (y0 + 56 + y1 - 34) / 2, (y1 - 34 - y0 - 56) / 2
n = int((wx1 - wx0) * S)

env = .055 + rng.random(n) * .045
sig = rng.normal(0, 1, n) * env
c = int(ONSET * n)
L = int(n * .30)
k = np.arange(L)
sig[c:c + L] += np.exp(-k / (L * .10)) * np.sin(k * .16) * .92
sig = np.clip(sig, -1, 1)

ax = wx0 + (wx1 - wx0) * ONSET
bx = wx0 + (wx1 - wx0) * ANLZ
d.rectangle(px(ax, y0 + 52, bx, y1 - 30), fill=(255, 122, 61, 20))   # 분석 구간
d.line(px(wx0, wcy, wx1, wcy), fill="#eef2f7", width=S)
for i in range(n):
    v = abs(sig[i]) * wamp
    xx = wx0 * S + i
    inw = ax * S <= xx <= bx * S
    d.line([xx, wcy * S - v * S, xx, wcy * S + v * S],
           fill=HOT if inw else "#2fc9b6", width=1)

d.line(px(ax, y0 + 46, ax, y1 - 30), fill=HOT, width=int(2 * S))
d.polygon([px(ax - 5, y0 + 42), px(ax + 5, y0 + 42), px(ax, y0 + 50)], fill=HOT)
# 분석 구간 bracket
by = y1 - 26
d.line(px(ax, by, bx, by), fill="#d9612a", width=int(1.5 * S))
for e in (ax, bx):
    d.line(px(e, by - 4, e, by + 4), fill="#d9612a", width=int(1.5 * S))
txt(d, (ax + bx) / 2, by + 5, "분석 구간 320ms", f(KR, 9), "#d9612a", anchor="ma")
txt(d, wx0, by + 5, "-250ms", f(MN, 9), TX3)
txt(d, wx1, by + 5, "+500ms", f(MN, 9), TX3, anchor="ra")


# ═══ 2. 소리의 지도 — 정지, 이 이벤트의 위치 ════════════════════
x0, y0, x1, y1 = head("map", "소리의 지도", "점 = 소리 재생")
gx0, gy0 = x0 + 20, y0 + 44
gw2, gh2 = (x1 - 20) - gx0, (y1 - 46) - gy0
mp = lambda u, v: ((gx0 + u * gw2) * S, (gy0 + v * gh2) * S)

CL = [(.28, .40, .135, (0, 191, 166, 125), 70, True),
      (.50, .80, .095, (124, 140, 255, 115), 40, False),
      (.76, .46, .255, (150, 165, 185, 95), 150, False)]
for cu, cv, r, col, cnt, near in CL:
    if near:                                            # 최근접 참조군만 강조
        cx, cy = mp(cu, cv)
        for m, al in ((1.0, 75), (1.7, 30)):
            rad = r * min(gw2, gh2) * m * S
            for a in range(0, 360, 10):
                a0, a1 = math.radians(a), math.radians(a + 5.5)
                d.line([cx + math.cos(a0) * rad, cy + math.sin(a0) * rad,
                        cx + math.cos(a1) * rad, cy + math.sin(a1) * rad],
                       fill=(0, 191, 166, al), width=int(1.2 * S))
    for _ in range(cnt):
        a, dd = rng.uniform(0, 6.28), math.sqrt(rng.random()) * r
        cx, cy = mp(cu + math.cos(a) * dd, cv + math.sin(a) * dd * 1.25)
        d.ellipse([cx - 2.5 * S, cy - 2.5 * S, cx + 2.5 * S, cy + 2.5 * S], fill=col)

sx, sy = mp(.315, .455)
for rr, al in ((22, 60), (14, 105)):
    d.ellipse([sx - rr * S, sy - rr * S, sx + rr * S, sy + rr * S],
              outline=(255, 122, 61, al), width=int(2 * S))
star = []
for i in range(10):
    a = -math.pi / 2 + i * math.pi / 5
    rr = (3.6 if i % 2 else 9.0) * S
    star.append((sx + math.cos(a) * rr, sy + math.sin(a) * rr))
d.polygon(star, fill=HOT)

# 콜아웃
cxx, cyy = sx / S + 26, sy / S - 30
d.line(px(sx / S + 11, sy / S - 8, cxx - 4, cyy + 12), fill=(217, 97, 42, 150), width=int(1.3 * S))
lbl = "참조군 안쪽 · A프로젝트"
lw = tw(d, lbl, f(KRB, 10.5)) + 18
d.rounded_rectangle(px(cxx, cyy, cxx + lw, cyy + 24), radius=int(7 * S), fill="#fff3ea")
txt(d, cxx + 9, cyy + 6, lbl, f(KRB, 10.5), "#d9612a")

txt(d, x0 + 20, y1 - 38, "2D projection — 축과 거리는 물리적 의미 없음", f(KR, 9), TX3)
lx = x0 + 20
for col, nm in [((0, 191, 166), "A프로젝트"), ((124, 140, 255), "B프로젝트"),
                ((150, 165, 185), "배경"), ((255, 122, 61), "★ 이 이벤트")]:
    d.ellipse(px(lx, y1 - 23, lx + 7, y1 - 16), fill=col)
    txt(d, lx + 10, y1 - 25, nm, f(KR, 9), TX2)
    lx += tw(d, nm, f(KR, 9)) + 18


# ═══ 3. 현장 진단 ═══════════════════════════════════════════════
x0, y0, x1, y1 = head("stat", "현장 진단")
d.rounded_rectangle(px(x1 - 76, y0 + 14, x1 - 22, y0 + 34), radius=int(10 * S), fill="#e9fbef")
txt(d, x1 - 49, y0 + 24, "양호", f(KRB, 10.5), "#16a34a", anchor="mm")
for i, (k2, v, u) in enumerate([("SNR", "18.3", "dB"), ("배경 노이즈", "54.1", "dB"),
                                ("이벤트 밀도", "12", "회/분")]):
    yy = y0 + 62 + i * 74
    txt(d, x0 + 22, yy, k2, f(KR, 11), TX2)
    fs = 30 if i == 0 else 22
    txt(d, x0 + 22, yy + 16, v, f(MNB, fs), ACC if i == 0 else TX)
    txt(d, x0 + 26 + tw(d, v, f(MNB, fs)), yy + (35 if i == 0 else 29), u, f(KR, 11), TX3)
    if i < 2:
        d.line(px(x0 + 22, yy + 58, x1 - 22, yy + 58), fill="#f1f5f9", width=S)
txt(d, x0 + 22, y1 - 30, "출장 체크리스트 §2 자동 기입", f(KR, 9.5), TX3)


# ═══ 4. 스펙트로그램 — 같은 시간창 + attribution ════════════════
x0, y0, x1, y1 = head("spec", "스펙트로그램", "파형과 같은 시간창")
sx0, sy0 = x0 + 20, y0 + 44
sx1, sy1 = x1 - 20, y1 - 52
gw, gh = int((sx1 - sx0) * S), int((sy1 - sy0) * S)

nb, nt = 96, 260
prof = np.linspace(.06, .70, nb)[:, None] ** 1.35
base = prof * (.88 + .12 * np.sin(np.linspace(0, 6, nt)))[None, :] + rng.random((nb, nt)) * .10
for row, amp in [(int(nb * .90), .26), (int(nb * .78), .16)]:
    base[row:row + 2] += amp
cc = int(ONSET * nt)
for j in range(int(nt * .30)):                       # 클릭: 광대역 + 빠른 감쇠
    if cc + j < nt:
        base[:, cc + j] = np.clip(base[:, cc + j] + .85 * math.exp(-j / (nt * .035)), 0, 1)
base = np.clip(base, 0, 1)
lut = np.zeros((nb, nt, 3), np.uint8)
lut[..., 0] = (255 - base * 225).clip(0, 255)
lut[..., 1] = (255 - base * 200).clip(0, 255)
lut[..., 2] = (255 - base * 140).clip(0, 255)
spec = Image.fromarray(lut).resize((gw, gh), Image.BILINEAR)
mask = Image.new("L", (gw, gh), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, gw - 1, gh - 1], radius=int(6 * S), fill=255)
img.paste(spec, (int(sx0 * S), int(sy0 * S)), mask)
d = ImageDraw.Draw(img, "RGBA")

# attribution overlay (목업)
ax2 = sx0 + (sx1 - sx0) * ONSET
ay2 = sy0 + (sy1 - sy0) * .42
for rw, rh, al in ((46, 62, 34), (30, 42, 46), (16, 24, 62)):
    d.ellipse(px(ax2 - rw * .35, ay2 - rh / 2, ax2 + rw * .85, ay2 + rh / 2),
              fill=(255, 122, 61, al))
d.ellipse(px(ax2 - 46 * .35, ay2 - 31, ax2 + 46 * .85, ay2 + 31),
          outline=(217, 97, 42, 170), width=int(1.4 * S))
d.line(px(ax2, sy0, ax2, sy1), fill=(217, 97, 42, 120), width=int(1.4 * S))

txt(d, x0 + 20, y1 - 45, "모델 반응 영역", f(KRB, 9.5), "#d9612a")
txt(d, x1 - 20, y1 - 45, "높음", f(KR, 8.5), TX3, anchor="ra")
gx = x1 - 46 - 5 * 13
for i2, al in enumerate((28, 52, 78, 110, 150)):
    d.rectangle(px(gx + i2 * 13, y1 - 44, gx + i2 * 13 + 11, y1 - 35), fill=(255, 122, 61, al))
txt(d, gx - 6, y1 - 45, "낮음", f(KR, 8.5), TX3, anchor="ra")
txt(d, x0 + 20, y1 - 25, "GradCAM 표시는 목업 — 모델 미연결", f(KR, 9), TX3)


# ═══ 5. 판정 + 시험 성적표 ══════════════════════════════════════
x0, y0, x1, y1 = CARDS["judge"]
txt(d, x0 + 22, y0 + 16, "이 이벤트", f(KRB, 13), TX)
txt(d, x1 - 22, y0 + 19, "14:02:31 · #47", f(MN, 10), TX3, anchor="ra")

txt(d, x0 + 22, y0 + 46, "참조 체결음과 유사", f(KRB, 26), TX)
txt(d, x0 + 22, y0 + 84, "최근접 A프로젝트 클릭  ·  참조군 안쪽  ·  SNR 18.3 dB", f(KR, 11), TX2)

# 버튼 2개 — 소리 재생 / 라이브 재개는 분리
bt = y0 + 112
d.rounded_rectangle(px(x0 + 22, bt, x0 + 222, bt + 44), radius=int(10 * S), fill=SOFT, outline=LINE, width=S)
d.polygon([px(x0 + 44, bt + 14), px(x0 + 44, bt + 30), px(x0 + 57, bt + 22)], fill=TX2)
txt(d, x0 + 68, bt + 14, "방금 소리 재생", f(KRB, 12), TX)

d.rounded_rectangle(px(x0 + 236, bt, x0 + 464, bt + 44), radius=int(10 * S), fill=ACC)
d.polygon([px(x0 + 260, bt + 13), px(x0 + 260, bt + 31), px(x0 + 275, bt + 22)], fill="#ffffff")
txt(d, x0 + 286, bt + 14, "현장 듣기 재개", f(KRB, 12), "#ffffff")

d.line(px(x0 + 22, y0 + 180, x1 - 22, y0 + 180), fill="#f1f5f9", width=S)

# 시험 성적표 — 검출된 것만 세지 않고 분모를 남긴다
txt(d, x0 + 22, y0 + 196, "이번 세션 시험", f(KRB, 12), TX)
txt(d, x1 - 22, y0 + 198, "발표자가 선언한 시험만 집계", f(KR, 9), TX3, anchor="ra")

for i, (nm, val, sub, col, bgc) in enumerate([
        ("체결 시험", "7 / 8", "검출 7 · 무응답 1", ACC, "#effcf9"),
        ("방해음 시험", "0 / 4", "클릭 판정 0건", "#3b82f6", "#eef4ff")]):
    bxx = x0 + 22 + i * 232
    d.rounded_rectangle(px(bxx, y0 + 220, bxx + 210, y0 + 286), radius=int(10 * S), fill=bgc)
    txt(d, bxx + 16, y0 + 231, nm, f(KR, 10.5), TX2)
    txt(d, bxx + 16, y0 + 245, val, f(MNB, 23), col)
    txt(d, bxx + 16, y0 + 271, sub, f(KR, 9.5), TX3)

txt(d, x0 + 22, y1 - 22, "방해음 = 볼펜 · 쇠 두드림 · 손가락 스냅 · 커넥터 충격", f(KR, 9.5), TX3)

img.resize((W // S, H // S), Image.LANCZOS).save(OUT)
shutil.copy(OUT, BASE / "mockup.png")                      # latest 포인터
shutil.copy(__file__, VDIR / f"{VERSION}_{SLUG}.py")       # 그 버전을 만든 스크립트

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
