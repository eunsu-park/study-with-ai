---
title: "Pre-Reading Briefing: Deep Space Climate Observatory: The DSCOVR Mission"
paper_id: "44_burt_2012"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# DSCOVR Mission: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Burt, J. and Smith, B., "Deep Space Climate Observatory: The DSCOVR Mission," 2012 IEEE Aerospace Conference, 2012. DOI: 10.1109/AERO.2012.6187025
**Author(s)**: Joe Burt, Bob Smith (NASA Goddard Space Flight Center)
**Year**: 2012

---

## 1. 핵심 기여 / Core Contribution

이 논문은 1998년 Al Gore 부통령이 제안한 Triana 지구 관측 임무가 우주왕복선 발사 취소 후 2001년부터 "Stable Suspension" 상태로 GSFC 청정실에 보관되어 오던 위성을, NOAA·NASA·USAF의 공동 노력(이른바 "Serotine Report")을 통해 우주기상 모니터링 임무인 DSCOVR로 재정의(repurpose)하여 2014년경 발사 가능성을 평가한 공식 보고서이다. 16년간 보관된 우주선의 시스템·서브시스템·계기(EPIC, NISTAR, PlasMag) 상태 점검, 위험요소(태양전지 접착제 산화, 점화 모터 그리스 등), 환경시험 재수행 계획, 새로운 ELV(Taurus II / Falcon 9) 통합 작업, 그리고 L1 Lissajous 궤도 진입을 위한 ΔV·연료 예산을 제시한다.

This paper documents the formal "Serotine" study by NASA, NOAA, and USAF that reassessed the feasibility of refurbishing and launching the long-mothballed Triana spacecraft as the Deep Space Climate Observatory (DSCOVR) — repurposed from an Earth-observing platform into the operational solar-wind monitor at the Sun–Earth L1 Lagrange point. After more than seven years in clean-room storage, the observatory was powered up in 2008 and found nearly intact; the paper details the status of every subsystem (mechanical, GN&C, propulsion, C&DH, power, comms), the residual risks (solar-cell adhesive oxidation, reaction-wheel grease, propellant heater anomaly), the refurbishment plan for the EPIC and NISTAR Earth-science instruments and the PlasMag space-weather suite (Faraday cup, magnetometer, electron spectrometer), the new launch-vehicle accommodations (Taurus II / Falcon 9, since the Shuttle was retired), the ground-system architecture using NOAA's RTSW network, and the ΔV/mass budget for direct insertion into a large-amplitude L1 Lissajous orbit.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1998년 Triana는 Christopher Columbus 함대의 망루지기 이름을 따 명명된, 지구 전체의 일조면 영상을 인터넷에 실시간 공개하기 위한 임무로 제안되었다. 21개월간 $249M(FY07$)이 투입되어 거의 완성된 상태였으나 정치적 논란과 STS 발사 일정 우선순위로 2001년 11월 보관에 들어갔다. 2003년 Columbia 사고 이후 STS 발사는 사실상 불가능해졌고, 2008년 NOAA·USAF의 우주기상 자산 노후화(특히 1997년 발사된 ACE)에 대한 우려가 DSCOVR 부활의 직접적 동력이 되었다.

In 1998 then-Vice President Al Gore proposed Triana to provide a continuous, internet-streamed view of the sunlit Earth from L1. The spacecraft was nearly built ($249M FY07, 21 months of integration) when politics and Shuttle manifest pressure forced it into "Stable Suspension" in November 2001. The Columbia accident (2003) made the original Shuttle launch path moot. By 2008, NOAA's reliance on the aging ACE spacecraft (launched 1997, operating beyond its design life at L1) for real-time solar-wind warnings created a clear operational gap, motivating the Serotine study and the rebranding of Triana as DSCOVR.

### 타임라인 / Timeline

```
1997 ─ ACE launched (real-time solar-wind monitor at L1)
1998 ─ Gore proposes Triana
2000 ─ Triana spacecraft fully integrated and tested
2001 Nov ─ Stable Suspension begins (clean-room storage, GSFC)
2003 ─ Columbia accident → STS path effectively closed
2008 ─ Aliveness test; Triana found nearly intact
2009 ─ NASA funds EPIC/NISTAR refurbishment
2011 ─ NOAA/NASA/USAF align for ELV launch ~2014
2012 ─ This paper (Burt & Smith)
2015 Feb ─ DSCOVR launched on Falcon 9 (post-paper)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- 천체역학 / Astrodynamics: Sun–Earth circular restricted three-body problem (CR3BP), Lagrange points L1–L5, halo / Lissajous orbits, station-keeping ΔV. / 태양–지구 제한 삼체문제, 라그랑주 점, Lissajous 궤도, 정지유지 ΔV.
- 우주기상 / Space weather: solar wind plasma parameters (n, V, T, B), why ~30–60 min advance warning is possible from L1 to Earth (1.5×10⁶ km / V_sw ≈ 30–60 min). / L1에서 지구까지 30–60분 선행 경고 원리.
- 우주선 시스템공학 / Spacecraft systems: subsystem taxonomy (Mechanical, GN&C, Propulsion, C&DH, Power, Comm, Thermal), TRL, environmental test (vibration, EMI/EMC, thermal-vacuum). / 서브시스템 분류, 환경시험.
- 계기 원리 / Instrument physics: Faraday cup ion energy/charge spectrometer, fluxgate magnetometer, "tophat" electrostatic analyzer for electrons, CCD photometry & filter-wheel imaging spectrometry, cavity radiometers for total irradiance. / 패러데이 컵, 플럭스게이트 자력계, 톱햇 분석기, CCD 광도계, 공동 복사계.
- 발사체 능력 / Launch vehicle performance: C3 vs Δv to L1, payload mass to high-energy escape orbits (Taurus II ~1280 kg to L1, Falcon 9 ~2000 kg, Delta II ~692/754 kg). / 발사체별 L1 도달 능력.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **DSCOVR** | Deep Space Climate Observatory — Triana 부활 임무명. NOAA 우주기상 모니터링 + EPIC/NISTAR 지구과학. / Repurposed Triana mission for L1 solar-wind monitoring plus Earth science. |
| **Triana** | 1998 NASA 임무 원형. Columbus 함대 망루지기 Rodrigo de Triana에서 명명. / Original 1998 NASA mission name, after Columbus's lookout. |
| **Stable Suspension** | 임무가 취소되지는 않은 채 청정실에 N₂ purge로 장기 보관된 상태. / Long-term clean-room storage with continuous nitrogen purge while mission stays officially active. |
| **L1 Lissajous orbit** | Sun–Earth L1 주변의 준주기 궤도; ACE, SOHO, Wind 등이 사용. / Quasi-periodic orbit around Sun–Earth L1 used by ACE, SOHO, Wind. |
| **EPIC** | Earth Poly-Chromatic Imaging Camera — 30.5 cm Ritchey–Chrétien, 2048×2048 CCD, 10 필터. / 30.5-cm telescope with 10-band filter wheel imaging the sunlit Earth. |
| **NISTAR** | NIST Advanced Radiometer — 3-cavity radiometer for Earth radiation budget (0.2–100 μm). / 지구 복사 수지 측정 3-공동 복사계. |
| **PlasMag** | Plasma-Magnetometer suite: Faraday Cup + Magnetometer + Electron Spectrometer + PHA. / 패러데이 컵, 자력계, 전자 분광기, PHA로 구성된 플라즈마-자기장 패키지. |
| **Faraday Cup** (FC) | MIT-built ion analyzer; 90 ms cadence solar-wind speed/density/temperature. / 90 ms 시간분해능으로 태양풍 속도·밀도·온도 측정. |
| **Serotine Team / Report** | "늦게 열리는" 솔방울에서 따온 NASA Serotine 팀의 재시작 평가 보고서. / NASA team named after a "late-opening" pinecone, who authored the restart assessment. |
| **GUS / IRIS / HASE** | Gyroscopic Upper Stage / Interface for Reusable IUS / HASE — STS 발사용이라 ELV 전환 시 제거. / Shuttle-era upper stage and adapters removed for ELV launch. |
| **RTSW Network** | Real-Time Solar Wind ground network for ACE → DSCOVR. / ACE/DSCOVR 실시간 태양풍 지상국 네트워크. |
| **Aliveness Test** | 보관된 우주선의 전원 인가 후 모든 서브시스템 동작 점검. / Power-on functional check of stored spacecraft. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) L1 거리 (Hill 근사) / L1 distance (Hill's approximation)**

$$ r_{L1} \approx R \left( \frac{m_2}{3 M_\odot} \right)^{1/3} $$

지구–태양 거리 $R = 1$ AU, 지구질량비 사용 → $r_{L1} \approx 0.01$ AU ≈ 1.5×10⁶ km. 논문은 "1/100 AU ≈ 1.5×10⁶ km ≈ 0.93×10⁶ mi"로 기술. / The paper quotes L1 at ~1/100 AU, matching this Hill estimate.

**(2) 로켓 방정식 / Tsiolkovsky rocket equation**

$$ \Delta V = I_{sp} g_0 \ln\!\left(\frac{m_0}{m_f}\right) $$

논문: 단발성 모노프로필 hydrazine, $I_{sp} = 220$ s, ΔV ≈ 200 m/s 목표 → 145 kg 연료를 ~40 kg로 감축 검토. / Used to size the propellant load for L1 insertion and stationkeeping.

**(3) L1 선행 경고 시간 / L1 warning lead time**

$$ \tau = \frac{r_{L1}}{V_{sw}} \approx \frac{1.5 \times 10^9 \text{ m}}{4 \times 10^5 \text{ m/s}} \approx 60 \text{ min} $$

태양풍 속도 400 km/s 가정 시 ~60분 선행. CME(>1000 km/s) 시 15–20분으로 단축. / Lead time scales inversely with V_sw.

**(4) 데이터 율 vs 안테나 크기 (링크 예산 정성식) / Data rate vs aperture (qualitative link budget)**

$$ \mathrm{SNR} \propto \frac{P_t G_t G_r}{L_{path}} \quad,\quad G_r \propto \left(\frac{\pi D}{\lambda}\right)^2 $$

논문은 10 m 이상 안테나는 32 kbps 전체 RTSW를, 6.3 m Boulder 국은 16 kbps 계기 데이터만 지원함을 명시. / Larger ground antennas needed for full real-time rate at 1.5×10⁶ km.

---

## 6. 읽기 가이드 / Reading Guide

- **§1–2 (Intro/Status)**: 임무 역사와 보관 상태 — 빠르게 통독. / Skim for context.
- **§3 (Risks)**: 태양전지 접착제(VCL/Glory 사례), 그리스, GIDEP — 장기 보관의 hardware degradation 사례 학습. / Detailed lessons on storage risks.
- **§4 (Subsystems)**: 가장 길고 핵심. 각 서브시스템(GN&C, 추진, C&DH, Power, Comm, Thermal) 별 상태와 refurbishment 작업 정독. / Read carefully — heart of the paper.
- **§6 (Instruments)**: EPIC 10개 필터(Table 1), NISTAR 3 채널, PlasMag 3 sensor를 표로 정리하며 읽기. / Tabulate instrument specs.
- **§9 (Launch Accommodations)**: Table 3–6 (ΔV, mass, launch vehicle 비교) — Minotaur V 검토와 거부 이유, Taurus II/Falcon 9 채택 이유. / Pay attention to mass-budget logic.
- **§10 (Conclusion)**: 한 단락만 읽어도 "feasible & cost-effective" 결론 확인. / Short.

---

## 7. 현대적 의의 / Modern Significance

DSCOVR는 2015년 2월 11일 SpaceX Falcon 9으로 실제 발사되어, 2016년 6월부터 ACE를 대체하는 NOAA의 운영 우주기상 자산이 되었다. 본 논문은 (1) 거의 완성된 우주선을 10년 이상 후 재가동한 매우 드문 사례 보고, (2) 임무 목적 자체를 지구과학(Triana)에서 우주기상(DSCOVR)으로 재정의하면서도 동일한 하드웨어를 활용한 사례, (3) STS 시대 설계를 ELV로 이식하는 실무적 노하우를 담고 있다.

Published three years before launch, the paper is the canonical engineering record of how a "mothballed" spacecraft can be brought back to flight: it documents the test methodology, the subsystem-level risk register, and the launch-vehicle trade study. DSCOVR has since become NOAA's primary real-time solar-wind monitor, providing the 30-to-60-minute geomagnetic-storm warning that protects the U.S. power grid, satellites, and aviation. EPIC's daily full-Earth images and NISTAR's radiation-budget data also realized — 17 years late — Gore's original vision. The paper is a practical playbook for repurposing legacy hardware, a topic again relevant as agencies consider extending or refurbishing aging assets (e.g., Voyager, Hubble servicing studies).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
