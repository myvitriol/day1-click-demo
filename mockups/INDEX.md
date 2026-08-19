# 1-Day Demo 목업 버전 기록

각 버전은 PNG + 그 PNG 를 만든 스크립트가 짝으로 남아 있다.

- **v01** [`v01_frozen-event.png`](v01_frozen-event.png) — 정지(freeze) 상태 · 파형 60% + 지도 40% 나란히 · 스펙트로그램 축소 · 검출 로그를 버리고 '이 이벤트 판정 + 시험 성적표'로 교체
- **v02** [`v02_three-panel.png`](v02_three-panel.png) — 세 뷰(파형·스펙트로그램·지도)만으로 완결 · 진단/판정/성적표를 각 뷰 안으로 흡수 · 패널마다 평이한 한 줄 캡션 — 처음 보는 사람 기준
- **v03** [`v03_three-panel-viridis.png`](v03_three-panel-viridis.png) — v02 와 같은 3패널 구성인데 스펙트로그램을 matplotlib viridis 로 교체 — 밝을수록 강함. 흰 점선 attribution + viridis 미니 컬러바 추가
- **v04** [`v04_ci-deeply.png`](v04_ci-deeply.png) — v03 에 DEEPLY CI 적용 — 로고 삽입, Deeply Pink(#FF3D59)=검출/이 이벤트, Deeply Black(#333132)=기존 체결음·본문. 스펙트로그램은 viridis 유지
- **v05** [`v05_ci-header-accent.png`](v05_ci-header-accent.png) — CI 색을 상단 바·로고 줄에만 한정 — 본문 accent 는 teal 로 분리. Pink=브랜드, Teal=검출/이 이벤트, Deeply Black=기존 체결음·본문
- **v06** [`v06_threshold-dial.png`](v06_threshold-dial.png) — v05 + 판정 기준 dial — 지도의 판정 원 반지름을 손으로 옮기며 recall/precision trade-off(operating point)를 그 자리에서 보여준다
- **v07** [`v07_dial-3step.png`](v07_dial-3step.png) — v06 의 dial 을 연속에서 3단(느슨/보통/엄격) 고정으로 — 시험이 12번뿐이라 연속 값은 없는 정밀도를 약속하고, 현장에서 값을 더듬는 모습도 막는다
- **v1.0** [`v1.0_baseline.png`](v1.0_baseline.png) — **확정 baseline (= v07 과 동일 화면)**. 3패널 + DEEPLY CI + 3단 판정 기준 dial. 이후 변경은 v1.1, v1.2 … 로 올린다
