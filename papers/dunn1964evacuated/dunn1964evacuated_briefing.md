---
title: "Pre-reading Briefing: An Evacuated Tower Telescope"
paper_id: "02_dunn_1964"
topic: Solar Observation
date: 2026-04-10
type: briefing
---

# 사전 브리핑: An Evacuated Tower Telescope / Pre-reading Briefing

**논문 / Paper**: "An Evacuated Tower Telescope"
**저자 / Authors**: Richard B. Dunn
**연도 / Year**: 1964
**저널 / Journal**: *Applied Optics*, Vol. 3, No. 12, pp. 1353–1357
**DOI**: 10.1364/AO.3.001353

> **참고**: 이 논문은 Pierce (1964)의 McMath Solar Telescope 논문과 **같은 저널, 같은 호**에 실렸습니다. 두 논문은 대형 태양 망원경의 두 가지 대조적 접근법을 대표합니다.

> **Note**: This paper was published in the **same journal and same issue** as Pierce's (1964) McMath paper. Together they represent two contrasting approaches to large solar telescopes.

---

## 1. 핵심 기여 / Core Contribution

Richard B. Dunn은 태양 망원경의 가장 심각한 문제인 **internal seeing**(망원경 내부의 공기 대류로 인한 이미지 열화)을 근본적으로 해결하기 위해 **광학 경로 전체를 진공으로 만드는** 새로운 설계를 제안하고 구현했습니다. Sacramento Peak Observatory에 건설된 이 76cm(30인치) 구경의 진공 타워 망원경(VTT)은 헬리오스탯 대신 **회전 터렛(rotating turret)**을 사용하여 시야 회전 문제를 해결하고, 광학 경로의 공기를 완전히 제거하여 internal seeing을 0으로 만들었습니다. 이 설계는 이후 모든 현대 고해상도 태양 타워 망원경의 템플릿이 되었습니다.

Richard B. Dunn proposed and implemented a fundamentally new design to solve the most serious problem of solar telescopes — **internal seeing** (image degradation from air convection inside the telescope) — by **evacuating the entire optical path**. This 76-cm (30-inch) vacuum tower telescope (VTT) at Sacramento Peak Observatory used a **rotating turret** instead of a heliostat to solve field rotation, and completely removed air from the optical path, reducing internal seeing to zero. This design became the template for all modern high-resolution solar tower telescopes.

---

## 2. 역사적 맥락 / Historical Context

```
태양 망원경 설계 패러다임 변화 / Solar Telescope Design Paradigm Shift:

1890s ── Hale: 개방형 태양탑 (open-air solar tower) at Mt. Wilson
  │       → 좋은 날씨에 의존, internal seeing 문제 있음
  │
1940s ── McMath-Hulbert Observatory: 수냉식 분광기 발전
  │
1960 ── Pierce: McMath Telescope 계획
  │       → 접근법 A: 수냉식 외피(water-cooled enclosure)로 열 제어
  │       → 70,000L 냉각수, 30톤 구리, but 내부 공기는 여전히 존재
  │
★ 1964 ── Dunn: Evacuated Tower Telescope ← 이 논문 / THIS PAPER
  │       → 접근법 B: 광학 경로의 공기를 완전히 제거 (진공)
  │       → internal seeing = 0
  │
1964 ── Pierce: McMath instrument paper (같은 호 / same issue!)
  │
1969 ── Dunn: "Sacramento Peak's New Solar Telescope" (Sky & Telescope)
  │       → 건설 완료 보고
  │
1985 ── Dunn Solar Telescope로 개명 (Dunn 은퇴 기념)
  │
2002 ── Swedish Solar Telescope (1m) — VTT 설계 + adaptive optics
  │
2020 ── DKIST (4m) — 진공은 아니지만 active thermal control 계승
```

**핵심 대조 / Key Contrast — McMath vs. Dunn:**

| 특성 / Feature | McMath (Pierce) | Dunn VTT |
|---|---|---|
| 구경 / Aperture | 160 cm | 76 cm |
| 초점 거리 / Focal length | 90 m | ~55 m |
| Internal seeing 해결 | 수냉 외피 (부분적) | **진공** (근본적) |
| 빔 유도 / Beam steering | Heliostat (평면 거울) | **Rotating turret** (2 거울) |
| 시야 회전 / Field rotation | 있음 (24시간 1회전) | **없음** (turret이 보상) |
| 분광기 위치 | 지하 관측실 | 지하 관측실 (유사) |
| 이미지 스케일 | ~0.44 mm/arcsec | ~0.27 mm/arcsec |

---

## 3. 필요한 배경 지식 / Prerequisites

### 3.1 Paper #1에서 배운 핵심 개념 / Key Concepts from Paper #1

- **Internal seeing**: 망원경 내부의 열 대류로 인한 이미지 흐림. Pierce는 수냉식 외피로 이를 줄이려 했지만, 돔을 연 후 2분 만에 이미지가 악화되는 문제는 여전했음.
  Image degradation from thermal convection inside the telescope. Pierce tried to reduce this with a water-cooled enclosure, but the 2-minute degradation problem persisted.

- **Heliostat의 한계**: 시야가 24시간에 1회전하고, 입사각-반사각 차이가 커서 거울 왜곡에 민감.
  Heliostat limitations: field rotates once per 24h, sensitive to mirror figure errors.

### 3.2 새로운 개념 / New Concepts

- **Evacuated optical path (진공 광로)**: 망원경 내부의 공기를 완전히 빼서 대류와 굴절률 변동을 원천 차단. 진공 튜브 안에서는 빛이 공기 없이 이동.
  Completely removing air from inside the telescope to eliminate convection and refractive index fluctuations.

- **Rotating turret (회전 터렛)**: 망원경 상단에서 두 개의 거울이 함께 회전하여 태양을 추적. 헬리오스탯과 달리 시야 회전을 자동 보상할 수 있음.
  Two mirrors at the top of the telescope rotate together to track the Sun. Unlike a heliostat, this can automatically compensate for field rotation.

- **Entrance window (입사창)**: 진공 경로의 시작점에 있는 광학 창. 이 창이 새로운 문제를 도입함 — 창 자체의 열 변형과 광학적 결함.
  Optical window at the start of the evacuated path. Introduces new problems — thermal distortion and optical defects of the window itself.

- **Vacuum vs. pressurized**: 진공 대신 헬륨 등 불활성 기체로 채우는 방법도 있음 (헬륨은 굴절률 변화가 공기의 1/8). Dunn은 완전 진공을 선택.
  Alternative: fill with helium (refractive index variation 1/8 of air). Dunn chose full vacuum.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Evacuated path** | 진공으로 만든 광학 경로. 공기 없이 빛만 이동 / Optical path under vacuum — light travels without air |
| **Vacuum tower** | 진공 튜브를 수직 타워 안에 배치한 망원경 형태 / Telescope with a vacuum tube inside a vertical tower |
| **Turret** | 망원경 상단에서 회전하며 태양을 추적하는 거울 조립체 / Rotating mirror assembly at the top that tracks the Sun |
| **Entrance window** | 진공 경로의 입구에 있는 투명한 광학 창 (보통 fused quartz) / Transparent optical window at the vacuum path entrance |
| **Tower seeing** | 타워 주변의 지면 가열에 의한 대기 난류 / Atmospheric turbulence from ground heating around the tower |
| **Gravitational deflection** | 수평 진공 튜브에서 자중에 의한 처짐 → 수직 배치가 유리 / Sag of horizontal vacuum tube under its own weight → vertical is better |
| **Image rotator** | 시야 회전을 보상하기 위한 광학 장치 (프리즘 또는 거울 조합) / Optical device to compensate field rotation |
| **Coelostat** | 두 개의 평면 거울로 태양을 추적하는 시스템. Heliostat의 대안 / Two-flat-mirror system for Sun tracking — alternative to heliostat |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Internal Seeing의 물리학 / Physics of Internal Seeing

공기 중 굴절률 $n$은 온도 $T$와 압력 $P$에 의존:

Refractive index $n$ in air depends on temperature $T$ and pressure $P$:

$$n - 1 \approx 7.9 \times 10^{-5} \frac{P(\text{mbar})}{T(\text{K})}$$

온도 변동 $\Delta T$에 의한 굴절률 변동:

Refractive index fluctuation from temperature variation $\Delta T$:

$$\Delta n \approx -7.9 \times 10^{-5} \frac{P}{T^2} \Delta T$$

경로 길이 $L$에서의 wavefront error:

Wavefront error over path length $L$:

$$\Delta \phi = \frac{2\pi}{\lambda} \Delta n \cdot L$$

McMath에서 $L \approx 155$ m, $\Delta T \approx 0.1°C$이면 wavefront error는 수십 $\lambda$에 달할 수 있음. **진공에서는 $\Delta n = 0$이므로 이 문제가 완전히 사라짐.**

In McMath with $L \approx 155$ m, $\Delta T \approx 0.1°C$, wavefront error can reach tens of $\lambda$. **In vacuum, $\Delta n = 0$, so this problem vanishes completely.**

### 5.2 회절 한계 분해능 / Diffraction-Limited Resolution

$$\theta = 1.22 \frac{\lambda}{D}$$

Dunn VTT: $D = 76$ cm → $\theta = 0.18''$ (at 550 nm). McMath보다 구경은 작지만, 진공 덕분에 실제로 더 나은 분해능을 달성할 수 있음.

Dunn VTT: $D = 76$ cm → $\theta = 0.18''$. Smaller aperture than McMath, but vacuum allows actually achieving better resolution.

### 5.3 진공 튜브의 외압 / External Pressure on Vacuum Tube

대기압이 진공 튜브에 가하는 힘:

Atmospheric pressure force on the vacuum tube:

$$F = P_{\text{atm}} \times A = 101325 \text{ Pa} \times \pi r^2$$

$r = 38$ cm (76 cm 구경)일 때: $F \approx 46{,}000$ N ≈ 4.7 톤. 입사창에도 이 압력이 가해지므로 창의 광학 왜곡이 중요한 설계 문제가 됨.

For $r = 38$ cm: $F \approx 46{,}000$ N ≈ 4.7 tons. This pressure on the entrance window makes its optical distortion a critical design issue.

---

## 6. 읽기 가이드 / Reading Guide

논문을 읽을 때 다음에 주목하세요 / Pay attention to these while reading:

1. **왜 진공인가?**: Dunn이 수냉 외피(McMath 방식)나 헬륨 충전 대신 완전 진공을 선택한 근거는?
   Why full vacuum instead of water-cooling (McMath) or helium filling?

2. **입사창 문제**: 진공을 유지하려면 입구에 창이 필요한데, 이 창이 도입하는 새로운 문제는?
   The entrance window is needed for vacuum but introduces new problems — what are they?

3. **Turret vs. Heliostat**: 회전 터렛이 헬리오스탯에 비해 갖는 장점과 단점은?
   Advantages and disadvantages of rotating turret vs. heliostat?

4. **수직 배치의 이유**: 왜 McMath처럼 경사면이 아닌 수직 타워인가?
   Why a vertical tower instead of an inclined structure like McMath?

5. **실제 성능**: 진공이 internal seeing을 완전히 해결했는가? 남아있는 한계는?
   Did vacuum completely solve internal seeing? What limitations remain?

---

## Q&A

*(읽기 중 질문이 추가됩니다 / Questions during reading will be appended here)*
