---
title: "The SOLAR-A Mission: An Overview"
authors: [Y. Ogawara, T. Takano, T. Kato, T. Kosugi, S. Tsuneta, T. Watanabe, I. Kondo, Y. Uchida]
year: 1991
journal: "Solar Physics 136, 1-16"
doi: "10.1007/BF00151692"
topic: Solar_Observation
tags: [SOLAR-A, YOHKOH, solar flares, X-ray imaging, HXT, SXT, BCS, WBS, ISAS, mission overview]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 41. The SOLAR-A Mission: An Overview / SOLAR-A 임무 개요

---

## 1. Core Contribution / 핵심 기여

본 논문은 1991년 8월 일본 ISAS가 가고시마 우주센터에서 M-3S-II 발사체로 발사한 SOLAR-A(궤도 진입 후 YOHKOH로 개명) 위성의 종합적 개요를 제공한다. SOLAR-A는 HINOTORI(1981)의 후계 임무로 태양활동 극대기(1989–1992) 중 약 2년간(궤도 수명 3–4년) X선·감마선 영역에서 태양 플레어의 고에너지 현상을 관측하기 위한 국제 협력 우주 천문대이다. 4개 핵심 기기 — HXT(Hard X-ray Telescope, 푸리에 합성형 격자 콜리메이터, 15–100 keV, ~5″), SXT(Soft X-ray Telescope, 변형 Wolter-I + 1024×1024 CCD, 3–60 Å, ~2.5″), BCS(Bragg Crystal Spectrometer, S XV/Ca XIX/Fe XXV/Fe XXVI 라인), WBS(Wide Band Spectrometer, SXS+HXS+GRS+RBM, 2 keV–100 MeV) — 가 동시에 동작하여 단일 플레어 사건을 다파장 일관성 있게 추적한다. 위성체는 무게 ~400 kg, 600 km × 31° × 97분 궤도, 3축 안정화(Z축 ~1″), 10 Mbyte 자기 버블 데이터 리코더, 32/4/1 kbps 텔레메트리를 갖추며, 4개 운용 모드(flare/quiet/night/BCS-out)를 자동·수동 전환한다.

This paper presents the complete mission design of SOLAR-A (renamed YOHKOH after launch) — a Japanese space solar observatory launched on 30 August 1991 by an M-3S-II rocket from Kagoshima Space Center as the successor to HINOTORI. The mission targeted high-energy flare phenomena (X- and gamma-rays) during solar cycle 22 maximum with a coordinated four-instrument payload: HXT (Fourier-synthesis bigrid collimator, 15–100 keV in 4 bands, ~5″ resolution, full-disk FOV); SXT (modified Wolter-I grazing-incidence telescope with 1024×1024 CCD, 3–60 Å soft X-rays plus optical filters, ~2.5″); BCS (four bent-crystal spectrometers for S XV, Ca XIX, Fe XXV, Fe XXVI lines at mÅ resolution); and WBS (gas SXS 2–30 keV, NaI HXS 20–400 keV, BGO GRS 0.2–100 MeV, plus radiation belt monitor RBM). The 400 kg three-axis-stabilized spacecraft sits in a 600 km / 31° / 97-min near-circular orbit, achieves 1″ Z-axis pointing accuracy, stores 10 Mbyte on a magnetic-bubble recorder, and dynamically switches among flare/quiet/night/BCS-out modes via onboard counting-rate thresholds. The paper emphasizes the synergy of coordinated instruments, automated flare detection, and an extensive list of scientific objectives focused on flare onset, particle acceleration, chromospheric evaporation, and quiet-Sun coronal evolution.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Section 1, pp. 1–2) / 서론

논문 도입부는 SOLAR-A의 동기를 제공한다. 직전 활동 극대기에 SMM(1980)과 HINOTORI(1981)가 hard X-ray 영상을 처음으로 만들었고, Tanaka(1987)의 type B(impulsive, 다중 footpoint 동시 점화)와 type C(gradual, 코로나에 위치한 ~수×10⁴ km 고도의 hard X-ray 소스)가 발견되었다. P78-1, Tansei IV, SMM, HINOTORI의 Bragg 분광계는 청색편이된 emission line을 통해 chromospheric evaporation을 직접 관측했고, T > 3×10⁷ K의 superhot 플라즈마(type A)에서 수소형 Fe 방출(Fe XXVI)이 두드러짐을 보였다. 감마선 플레어들은 ion이 수 MeV로 수 초 안에 전자와 동시에 가속되며, solar limb 근처에서 우선적으로 관측되어 방출의 비등방성을 시사했다.

The introduction motivates SOLAR-A. During the previous solar maximum, SMM (1980) and HINOTORI (1981) produced the first hard X-ray flare images, revealing Tanaka's (1987) type-B (impulsive, simultaneous multi-source footpoints) and type-C (gradual, coronal source at altitudes of a few ×10⁴ km) flares. Bragg spectrometers on P78-1, Tansei IV, SMM, and HINOTORI directly observed chromospheric evaporation via blueshifted emission lines and found H-like Fe XXVI emission from superhot plasmas (type A, T > 3×10⁷ K). Gamma-ray flares showed ions accelerated to MeV in seconds simultaneously with electrons, and limb-preference suggested anisotropy.

**한계 / Identified deficiencies (motivating SOLAR-A)**:
1. Hard X-ray imager의 공간 분해능이 낮고 에너지 범위가 ~30 keV 미만 → thermal source의 오염. Low spatial resolution and energy range below ~30 keV led to thermal contamination.
2. Soft X-ray 반사망원경 부재(SKYLAB 1973–74 이후 처음). No grazing-incidence soft X-ray telescope had flown since SKYLAB.
3. Bragg 분광계의 낮은 감도로 플레어 onset 데이터 없음. Low sensitivity prevented clean onset-phase data.
4. Hard X-ray와의 동시 영상 부재로 감마선 플레어 위치 불명. No simultaneous hard X-ray imaging at gamma-ray energies.

이 4가지 결점을 동시에 해결하는 것이 SOLAR-A의 설계 철학이다 (Table I 요약: 발사 1991-08, 궤도수명 3–4년, PI 표).

These four shortcomings collectively defined SOLAR-A's design philosophy. Table I lists key personnel: project manager Y. Ogawara (ISAS), project scientist Y. Uchida (Univ. Tokyo), and PIs K. Kai† & K. Makishima (HXT), T. Hirayama & L. W. Acton (SXT), J. Nishimura (WBS), E. Hiei & J. L. Culhane (BCS).

### Part II: Scientific Instruments (Section 2, pp. 3–5) / 과학 기기

#### 2.1 HXT — Hard X-ray Telescope (p. 3)

HXT(Kosugi et al., 1991)는 64개 bigrid modulation collimator 어레이로 구성된 푸리에 합성 망원경이다. 각 colimator subunit은 2.3 × 2.3 cm 단면, 0.5 cm 두께 NaI(Tl) 결정과 1인치 포토튜브로 구성된다. 64개 격자쌍이 적절한 위치각과 파장수 $(u,v)$-평면 위치에 배치되어 32개 복소 푸리에 성분을 동시에 측정한다 (각 격자쌍은 sin/cos 두 위상으로 분할). 합성 개구는 grid pitch에 의한 fundamental wave number로 결정되어 약 2'×2' field-of-view를 제공하지만 전체 태양면을 커버한다. 각분해능 ~5″, 4개 에너지 대역(15–24, 24–35, 35–57, 57–100 keV), 시간 분해능 0.5 s.

HXT (Kosugi et al., 1991) is a Fourier-synthesis telescope of 64 bigrid modulation collimators. Each subunit (2.3×2.3 cm cross-section) has a 0.5-cm NaI(Tl) crystal coupled to a 1-inch phototube. Pairs of grids sample 32 complex Fourier components in $(u,v)$-space (each pair gives two phases). The synthesis aperture set by the fundamental wave number gives ~2'×2' FOV, but the FOV covers the whole Sun. Angular resolution ~5″ in 4 energy bands (15–24, 24–35, 35–57, 57–100 keV) at 0.5 s temporal resolution. Effective area ~70 cm².

#### 2.2 SXT — Soft X-ray Telescope (pp. 3–4)

SXT(Tsuneta et al., 1991)는 grazing-incidence 반사 망원경(modified Wolter type I)으로 3–60 Å 영역에서 동작한다. 검출기는 1024×1024 CCD, 광학계의 각분해능은 ~2 arc sec(소수점 슬릿 분해능 한계는 2.4″ 픽셀이 결정). 두 개의 필터 휠과 셔터 장치로 에너지 대역과 노출 시간을 조정하며, 광학(4600–4800 Å, 4293–4323 Å) 및 X-선 모드를 모두 갖춘다. 데이터 획득은 스페이스크래프트 데이터 프로세서로부터의 명령에 따라 전용 마이크로프로세서가 제어한다.

SXT (Tsuneta et al., 1991) is a grazing-incidence modified Wolter-I reflecting telescope (3–60 Å) with a 1024×1024 CCD. Optical-system resolution ~2″, slightly better than the 2.4″ pixel size. Two filter wheels and a shutter mechanism select energy band and exposure. The CCD also operates in optical (4600–4800 Å continuum, 4293–4323 Å Hα region). A dedicated microprocessor controls filter/exposure selection.

**관측 모드 / Observing modes**: 조용한 태양 시 whole-Sun 영상 + 최대 4개 active region partial-frame; 플레어 시 가장 밝은 영역의 partial-frame을 0.5 s 단위로 집중 촬영. When quiet, whole-Sun + up to 4 active region partial frames; when flaring, partial frames at up to 0.5 s cadence on the brightest region.

#### 2.3 WBS — Wide Band Spectrometer (pp. 4–5)

WBS(Yoshimori et al., 1991)는 4종 검출기로 이루어진다:
- **SXS** (Soft X-ray Spectrometer): xenon 가스 비례계수기, 2–30 keV. 0.25 s마다 2채널 계수율, 1 s마다 128채널 펄스높이 스펙트럼.
- **HXS** (Hard X-ray Spectrometer): NaI(Tl), 20–400 keV. 0.125 s 2채널, 1 s 32채널 PHS.
- **GRS** (Gamma-Ray Spectrometer): 두 개의 동일 BGO scintillator, 0.2–100 MeV. 0.25 또는 0.5 s 6채널 계수, 4 s 128채널 PHS.
- **RBM** (Radiation Belt Monitor): 태양 방향에 수직, Si diode + NaI scintillator. >20 keV 입자 0.25 s 단위 측정 — 플레어 트리거 시 우주선 입자 배경 변동을 모니터링.

WBS detects all four target spectral domains at high cadence and is also used during night to detect cosmic gamma-ray bursts.

#### 2.4 BCS — Bragg Crystal Spectrometer (p. 5)

BCS(Culhane et al., 1991)는 4개의 굽은(bent) 결정 분광계로 위치-감응 비례계수기를 사용한다. 측정 라인:
- S XV @ 5.0385 Å (5.0160–5.1143 Å, 분해능 3.232 mÅ)
- Ca XIX @ 3.1769 Å (3.1631–3.1912 Å, 0.918 mÅ)
- Fe XXV @ 1.8509 Å (1.8298–1.8942 Å, 0.710 mÅ)
- Fe XXVI @ 1.7780 Å (1.7636–1.8044 Å, 0.565 mÅ)

각 채널 최대 256 spectral bin, 자체 384 kbyte 큐 메모리(0.125 s 시간 분해능 가능)에 초기 phase 데이터를 임시 저장 후 main DP가 일정한 속도로 읽어 telemetry로 보낸다 — 이로써 플레어 onset의 고시간분해능 데이터가 보존된다.

BCS uses four bent crystals with position-sensitive proportional counters, providing up to 256 spectral bins per band. Each band has its own 384-kbyte queue memory (storing 0.125-s onset data) read out at a fixed rate by the main DP. This guarantees onset capture even when telemetry is bandwidth-limited.

### Part III: The Spacecraft (Section 3, pp. 5–8) / 위성체

#### 3.1 General (p. 5)

발사 시간/장소: 1991년 8월, 가고시마 우주센터(31°N, 131°E), M-3S-II 발사체. 궤도: ~600 km 고도 거의 원궤도, 31° 경사각, 97분 주기. 위성체 치수 100×100×200 cm, 외부 태양전지판 150×200 cm 두 장, 무게 ~400 kg. H자 구조: center panel(SXT, HXT, BCS 광학 벤치), 두 개 측면 panel, top panel(WBS 검출기 + SXT/HXT/BCS aperture window), bottom panel(전자기기). 태양 전지판 ~570 W, 낮 220 W / 밤 180 W 가용(NiCd 배터리 충전).

Launched August 1991 from Kagoshima (31°N, 131°E) by M-3S-II into a 600-km / 31°-inclination / 97-min near-circular orbit. Spacecraft is 100×100×200 cm with two 150×200 cm solar panels, total ~400 kg. The H-shaped structure has a central panel hosting SXT, HXT, and BCS as the optical bench; the top panel houses WBS and instrument aperture windows; bottom panel holds electronics. Solar arrays generate ~570 W; ~220 W (day) / 180 W (night) usable after NiCd charging.

#### 3.2 Attitude Control (p. 6)

3축 안정화. Z축은 태양 중심, 안정도 ~1 arc sec s⁻¹ 및 수 arc sec min⁻¹; Y축은 천체 북쪽. 액추에이터: momentum wheels, magnetic torquers, control-moment gyros. 센서: Sun sensor 2개 + star tracker, geomagnetic sensor, 4-자이로 관성 기준 유닛. Z축 결정 정확도 ≤1″, fine sun sensor의 misalignment 제외. 두 영상 망원경(HXT, SXT)은 자체 aspect sensor를 가짐.

Three-axis stabilization. Z-axis points sunward to ~1″ s⁻¹ and a few ″ min⁻¹ stability; Y-axis points celestial north. Actuators are momentum wheels, magnetic torquers, and control-moment gyros; sensors are 2 Sun sensors, a star tracker, geomagnetic sensors, and a 4-gyro IRU. Z-axis determination ≤1″. Two imagers carry independent aspect sensors.

#### 3.3 Onboard Data Processing (pp. 7–8)

DP 유닛은 이중 중복 마이크로컴퓨터 시스템. 핵심 기능을 hardwired logic으로도 구현하여 양 컴퓨터 모두 고장 시 대비. 기능: (1) 모든 기기로부터 데이터 수집, (2) 텔레메트리 스트림 편집, (3) 데이터 리코더 기록/재생, (4) 자동 관측 모드 제어(SXT 위주). 텔레메트리 32/4/1 kbps. 10 Mbyte 자기 버블 리코더가 우주선 식 동안 데이터 저장, 지상국 접촉 시 262 kbps로 덤프.

The DP unit consists of dual-redundant microcomputer systems with hardwired-logic backup. It (1) gathers data from all instruments, (2) edits the telemetry stream, (3) controls the BDR record/playback, and (4) automates observing modes (notably SXT). Real-time telemetry at 32/4/1 kbps; 10-Mbyte BDR dumps at 262 kbps during ground contact.

#### 3.4 Command System (p. 8)

Kagoshima Space Center에서 5 orbits/day × ~10 min 접촉 동안만 명령 송신. Organized Group(OG): 최대 32 명령, 128 OG 저장. Operation Program(OP): 일련의 OG를 시간 간격과 함께 자동 디스패치, 최대 ~10일 운용 가능.

Commands are uplinked from Kagoshima during 5 contacts/day × ~10 min each. Up to 128 'Organized Groups' (OGs) of 32 commands each can be stored, and OGs can be sequenced as 'Operation Programs' (OPs) lasting up to ~10 days, enabling pre-programmed multi-day operations.

#### 3.5 Telemetry (pp. 8–9)

Deep Space Network 3국(Goldstone, Madrid, Canberra) + Kagoshima. 두 채널: S-band(2.2 GHz) 실시간, X-band(8.4 GHz) BDR 재생. Kagoshima는 두 채널 동시 수신; DSN국은 X-band 재생만. 데이터는 Kagoshima 실시간 → ISAS 사가미하라; DSN 데이터는 JPL 경유 NASCOM 라인으로 ISAS로 전송. 최종 ISAS 데이터베이스에 보관(Morrison et al., 1991 참조).

Telemetry uses S-band (2.2 GHz, real-time) and X-band (8.4 GHz, recorder playback) to four ground stations: Kagoshima plus the DSN stations at Goldstone, Madrid, Canberra. All real-time data flow to ISAS via dedicated link; DSN data come through NASCOM. All archives reside at ISAS (see Morrison et al., 1991).

### Part IV: Operations (Section 4, pp. 9–13) / 운용

#### 4.1 Observing Modes (p. 9)

4개 모드 × 3 텔레메트리 데이터율(high 32, medium 4, low 1 kbps):

| Mode | Rate | Purpose |
|---|---|---|
| Flare | 32 or 4 kbps | All four instruments share telemetry |
| Quiet | 32 or 4 kbps | Mostly SXT (with HXT lowest band) |
| Night | 1 kbps | HXS+RBM only (cosmic GRB hunting) |
| BCS-out | 32 kbps | Sweep BCS queue memory after flare |

플레어 모드에서는 4기기가 텔레메트리를 분배(HXT 16, SXT 64, WBS 8, BCS 8 byte/128-byte frame). 조용 모드에서는 SXT가 64+16 byte를 사용하며 두 부분으로 나뉨(full frame + partial frame).

Four modes × three telemetry rates. In flare mode, the four instruments share the 128-byte frame as HXT 16 / SXT 64 / WBS 8 / BCS 8 (basic 32 byte). In quiet mode, SXT takes 64+16 bytes split into full-frame (whole-Sun) and partial-frame (active regions); the 16-byte section yields whole-Sun cadence ~10 min at high data rate.

**HXT 사전플레어 데이터 / HXT preflare data**: lowest energy band(15–24 keV) data가 'basic' 텔레메트리 섹션에 항상 기록되어, 플레어 시작 직전의 hard X-ray 영상이 보존된다.

HXT lowest-band (15–24 keV) data is always logged in the 'basic' section, preserving preflare hard X-ray images even when other bands have no significant signal.

#### 4.2 Automated Mode Control (pp. 10–12)

Fig. 2의 상태도: spacecraft day 시작 시 quiet mode (high 또는 medium rate). HXS, SXS, BCS 중 하나(선택 가능)에서 'flare threshold' 초과 + RBM 정상 → flare flag set, 다음 2초 이내에 flare mode + high rate. 계수율 추가 단서:
- 'flare end threshold' 미만 → quiet 복귀 (case A)
- 위에 있지만 'great flare threshold' 미만 → flare-medium rate 지속 (case B)
- great flare threshold 초과 → flare-high rate 지속 (case C, 'great flare')

Fig. 2 shows the state machine: quiet → flare on threshold crossing (provided RBM does not also rise — that would indicate trapped radiation, not a real flare). Three thresholds (Fig. 3) classify flares: case A (sub-threshold tail returns to quiet), case B (above flare-end threshold but below great-flare → continue flare-medium), case C (above great-flare → continue flare-high). The 'flare minimum duration' enforces a guaranteed coverage interval.

일몰 시 BCS-out 모드 또는 직접 night 모드로 전환; 일출 시 quiet로 복귀. 플레어 모드 종료 후 BCS 큐에 데이터가 남아 있으면 BCS-out으로 비운 후 quiet로 간다.

At sunset the controller chooses BCS-out (if BCS queue has flare data) then night, or directly night. At sunrise it returns to quiet.

#### 4.3 Bubble Data Recorder Storage Control (pp. 12–13)

BDR 10 Mbyte는 high rate에서 ~40 분이면 채워진다(궤도 일조 ~60분 이상). 따라서 overwrite가 불가피. DP는 새로 쓸 데이터의 중요도와 이미 저장된 데이터의 중요도를 비교하여 보호한다:

$$\text{great flare} > \text{normal flare} > \text{quiet/night}$$

게다가 'campaign observation' 명령을 통해 운영자가 특정 기간 데이터를 무조건 보호할 수 있다 (예: SXT whole-Sun synoptic movie). 이는 HINOTORI 임무에서 검증된 우선순위 시스템의 진화형이다(Kondo, 1983).

The 10-Mbyte BDR fills in ~40 min at 32 kbps but each daylit pass exceeds 60 min, forcing overwrite. The DP compares importance of incoming vs. stored frames using priority great-flare > normal-flare > quiet/night, an evolution of HINOTORI's logic (Kondo 1983). 'Campaign observations' unconditionally protect data over a specified interval, e.g., for SXT synoptic movies.

### Part V: Scientific Topics (Section 5, pp. 13–15) / 과학 주제

논문은 14개의 과학 목표를 4개 카테고리로 정리:

**(A) Flare-related phenomena (9개)**: active region 진화(특히 preflare, coronal loop 자기 연결성); flare onset(impulsive phase 직전); 고온 loop/arcade 형성; electron 가속 위치/시간; ion 가속 + gamma-ray line flares; loop footpoint 동역학(질량 분출/evaporation); flare ejecta/shock/plasmoid를 X선으로 관측 가능한지; hard X-ray/soft X-ray/optical/radio 소스 간 관계; white-light flares (SXT aspect sensor로 관측).

**(B) Non-flare dynamics (3개)**: surge/Brueckner jets; quiescent filament 소멸 + 저에너지 two-ribbon flare; coronal mass ejections.

**(C) Other activity (3개)**: X-ray bright points + solar-cycle 변화; micro/nano flares; active-region loop 형성 진화.

**(D) Global structure (4개)**: quiet coronal loop 형성; coronal hole 거동; solar oscillations(SXT aspect sensor); 단순 photospheric imaging.

The 14 science objectives are grouped: (A) flare-related — 9 items including preflare evolution, onset, hot loop/arcade formation, electron/ion acceleration, footpoint dynamics, plasmoids, multi-wavelength source relations, white-light flares; (B) non-flare dynamics — surges/jets, filament disappearance, CMEs; (C) other — X-ray bright points, micro/nano flares, AR loop evolution; (D) global structure — quiet loops, coronal holes, solar oscillations, photospheric imaging.

**선택 사례 / Worked example: flare onset**: HINOTORI/SMM 결과 — (a) Fe XXV/Ca XIX 라인에서 300–400 km/s의 격렬한 상승운동이 onset 직전 감지(Tanaka et al. 1983; Antonucci 1983); (b) X선 소스가 이미 고온 상태로 loop 정상에서 처음 출현, 급격한 팽창/냉각 없음 — 봉입(confinement)과 절연 메커니즘 필요(Tsuneta et al. 1984); (c) gamma-ray 라인이 hard X-ray impulsive burst와 동시 출현 — 거의 즉각적 ion 가속 필요(Forrest & Chupp 1983; Nakajima et al. 1983). (a)+(b)는 impulsive phase 직전에 매우 중요한 동역학적 phase 존재 가능성 시사(Uchida & Shibata 1988). SOLAR-A는 어디·언제·어떻게 질량과 에너지가 flaring loop에 들어오는지 정밀 답변할 수 있고, 자기 재결합 발생 여부도 magnetic connectivity 변화로 검증 가능.

The paper picks **flare onset** as the worked example. Tanaka et al. (1983) and Antonucci (1983) found 300–400 km/s upflows in Fe XXV / Ca XIX immediately before onset; Tsuneta et al. (1984) found high-T sources confined at looptops without rapid expansion (requiring special confinement); Forrest & Chupp (1983) and Nakajima et al. (1983) found gamma-ray line emission simultaneous with the impulsive hard X-ray burst, requiring near-instantaneous ion acceleration. Together these motivate Uchida & Shibata's (1988) hypothesized pre-impulsive dynamical phase that SOLAR-A can directly test. The mission can answer 'where, when, how' mass and energy enter flaring loops and whether magnetic reconnection is responsible.

### Part VI: Concluding Remarks (Section 6, pp. 15–16)

활동 극대기에 SOLAR-A만이 태양 플레어 전용 위성이며, soft X-ray부터 gamma-ray까지의 광범위 에너지 범위에 걸친 체계적 데이터를 제공한다. HINOTORI/SMM 대비 큰 개선. X-/gamma-ray 만으로는 플레어를 완전 이해할 수 없으므로 지상 광학·전파 관측소와의 협력, 이론 연구가 필수적임을 강조. 향후 SOLAR-B(현재의 Hinode)도 언급(논문 발표 당시는 미래 계획).

SOLAR-A is the unique flare-dedicated mission during the current maximum, providing systematic broadband data over X- and gamma-ray ranges with substantial improvements over HINOTORI/SMM. Coordinated ground-based optical/radio observations and theory are essential. The authors anticipate a future SOLAR-B (now Hinode, launched 2006).

---

## 3. Key Takeaways / 핵심 시사점

1. **Coordinated four-instrument design / 4기기 협업 설계** — SOLAR-A의 핵심 혁신은 단일 플레어를 hard X-ray imaging(HXT), soft X-ray imaging(SXT), high-energy spectroscopy(WBS), 고분해능 라인 분광(BCS)으로 동시에 관측하여, 입자 가속/플라즈마 가열/chromospheric evaporation의 시간적 인과 관계를 풀 수 있다는 점이다. SOLAR-A's signature innovation is simultaneously imaging a single flare in hard X-rays (HXT), soft X-rays (SXT), broadband spectroscopy (WBS), and resolved line profiles (BCS) so the temporal causality among acceleration, heating, and evaporation can be untangled.

2. **Fourier-synthesis hard X-ray imaging / 푸리에 합성 영상** — HXT의 64개 bigrid collimator pair는 각각 $(u,v)$-평면의 한 점을 표본화하여 32개 복소 visibility를 형성, 이로부터 영상을 합성한다. 이는 전파 간섭계의 X-선 버전이며, ~5″ 분해능은 SMM/HINOTORI보다 한 자릿수 향상. HXT pioneers the X-ray analogue of radio aperture synthesis: 64 bigrid pairs sample 32 complex Fourier components, reconstructing images with ~5″ resolution — an order of magnitude better than SMM/HINOTORI hard X-ray imagers.

3. **Wolter-I + CCD soft X-ray imaging / 그레이징 입사 + CCD 영상** — SXT는 SKYLAB 이후 첫 본격적 grazing-incidence 태양 X-선 망원경이자 CCD 검출기 사용 첫 사례. 1024×1024 픽셀, 2.5″ 분해능, 3–60 Å + 광학 모드 동시 가능 — coronal loop 구조와 자기 연결성을 시각적으로 드러내는 능력의 도약. SXT — first serious grazing-incidence solar X-ray imager since SKYLAB and the first to use a CCD — produces 1024² × 2.5″ images covering 3–60 Å plus optical, enabling visual mapping of coronal loop structures and magnetic connectivity.

4. **Onboard automation as scientific necessity / 과학적 필수의 자동화** — 5번/일 × 10분 지상 접촉, 10 Mbyte BDR, 60+분 일조 시간이라는 운용 제약 하에서, DP의 자동 flare 감지(threshold crossing + RBM cross-check)와 우선순위 기반 overwrite 보호가 없다면 'great flare' 데이터를 놓칠 수 있다 — 자동화는 단순한 편의가 아니라 과학적 필수 조건. Given limited ground contacts (5/day × 10 min), 10 Mbyte BDR, and 60+-min daylit orbits, autonomous flare detection (threshold crossing with RBM cross-check) plus priority-based overwrite protection is not a convenience but a scientific necessity to capture great flares.

5. **Triple-threshold flare classification / 삼중 임계값 분류** — flare/flare-end/great-flare 임계값으로 플레어를 case A(미발달), B(보통), C(거대)로 분류하여 데이터율과 데이터 보호 우선순위를 자동 결정. 이 로직은 HINOTORI(Kondo 1983)의 발전형으로 현대 우주 미션 표준의 원형. The flare/flare-end/great-flare triple-threshold scheme classifies events into cases A (sub-threshold), B (normal), C (great), automatically selecting data rate and overwrite priority — a HINOTORI-derived (Kondo 1983) blueprint for modern mission autonomy.

6. **Flare onset as the central physics target / 플레어 onset이 핵심 물리 목표** — 14개 과학 목표 중 저자들이 단 하나만 자세히 다룬 것이 flare onset(질량/에너지 유입, 자기 재결합 검증)이며, SOLAR-A의 다중 분광·영상 기능이 'where, when, how' 질문에 답하기 위한 것임을 명시. The authors' single worked example — flare onset — emphasizes that SOLAR-A's coordinated capabilities exist to answer 'where, when, how' mass/energy enter flaring loops and to test whether magnetic reconnection drives flares.

7. **International collaboration and engineering legacy / 국제 협력과 엔지니어링 유산** — Japan(ISAS, NAOJ, 도쿄대 등) + US(LPARL, Stanford, UC Berkeley, U. Hawaii) + UK(MSSL, RAL) 협력 모델, M-3S-II 발사체, 자기 버블 데이터 리코더, 변조 콜리메이터의 푸리에 합성, S-band+X-band 텔레메트리 — Hinode와 후속 임무들의 엔지니어링 청사진을 제공. The Japan–US–UK collaboration model (ISAS+NAOJ + LPARL/Stanford/UCB/Hawaii + MSSL/RAL), M-3S-II launcher, magnetic-bubble recorder, modulation-collimator Fourier synthesis, and dual S/X-band telemetry constitute an engineering blueprint inherited by Hinode (SOLAR-B) and beyond.

8. **Bridging two solar maxima / 두 태양극대기 연결** — 1980 SMM/1981 HINOTORI에서 발견된 의문(type B/C, superhot, gamma-ray line)을 다음 극대기(1989–1992)에서 SOLAR-A가 해결하도록 설계된 명시적 임무 — '한 극대기의 발견이 다음 극대기의 임무를 낳는' 태양물리 임무 사이클의 모범. SOLAR-A explicitly addresses the questions posed by the prior maximum's SMM/HINOTORI discoveries (type B/C flares, superhot plasmas, gamma-ray lines), exemplifying the solar-physics mission cycle in which one maximum's discoveries shape the next maximum's mission.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Fourier-Synthesis Imaging (HXT) / 푸리에 합성 영상

각 modulation collimator pair는 sky brightness $I(x,y)$의 푸리에 변환에서 한 점을 측정한다:

Each modulation collimator pair samples one point of the sky-brightness Fourier transform:

$$V(u_i, v_i) = \iint I(x,y) \exp\left[-2\pi j (u_i x + v_i y)\right] dx\,dy$$

64개 격자쌍 → 32개 독립적 $(u_i, v_i)$ 점, 각각 cos/sin (real/imaginary) 위상으로 측정. 영상 재구성:

64 grid pairs → 32 independent $(u,v)$ samples, each measured with both phases. Image reconstruction:

$$\hat{I}(x,y) = \sum_{i=1}^{32} V_i \exp\left[2\pi j (u_i x + v_i y)\right]$$

- $u_i, v_i$: spatial-frequency coordinates determined by grid pitch $p_i$ and orientation: $|u_i| = 1/p_i$
- 가장 미세한 grid pitch가 분해능 결정: $\theta_{\min} \sim 1/u_{\max} \sim 5''$
- The finest grid pitch sets the angular resolution $\theta_{\min} \sim 1/u_{\max} \sim 5''$
- FOV는 fundamental wave number(가장 굵은 grid)가 결정: ~2'×2' synthesis aperture, 그러나 collimator 시야 자체는 전체 태양

### 4.2 Bragg Spectroscopy (BCS) / Bragg 분광법

$$n\lambda = 2d \sin\theta$$

- $n$: 차수(주로 1차)
- $d$: 결정 격자 간격
- $\theta$: 입사각
- $\lambda$: X-선 파장

BCS 분해능: 4개 라인의 공칭/측정 범위/분해능 (Table II)

| Line | Nominal $\lambda_0$ (Å) | Range (Å) | $\Delta\lambda$ (mÅ) | $\lambda/\Delta\lambda$ |
|---|---|---|---|---|
| S XV | 5.0385 | 5.0160–5.1143 | 3.232 | ~1560 |
| Ca XIX | 3.1769 | 3.1631–3.1912 | 0.918 | ~3460 |
| Fe XXV | 1.8509 | 1.8298–1.8942 | 0.710 | ~2610 |
| Fe XXVI | 1.7780 | 1.7636–1.8044 | 0.565 | ~3150 |

도플러 속도 분해능 / Doppler velocity resolution:
$$\frac{\Delta v}{c} = \frac{\Delta\lambda}{\lambda} \implies \Delta v_{\text{FeXXV}} \sim \frac{0.71\times 10^{-3}}{1.85} c \approx 115~\text{km/s}$$

이는 chromospheric evaporation의 300–400 km/s upflow를 잘 분해할 수 있음을 의미.

This easily resolves the 300–400 km/s chromospheric evaporation upflows reported by Tanaka et al. (1983).

### 4.3 Thermal Bremsstrahlung Diagnostic / 열적 제동복사

광학적으로 얇은 열 플라즈마의 X-선 방출:

Optically thin thermal X-ray emission:

$$\epsilon_\nu \propto n_e n_i T^{-1/2} \exp(-h\nu/kT)\,\bar{g}_{ff}$$

전체 플럭스 (단위 면적당): 

$$F = \int_V \epsilon_\nu \, dV \, / \, (4\pi D^2)$$

emission measure와 결합:
$$EM = \int n_e^2 dV$$
$$F \propto \frac{EM \cdot T^{-1/2} \exp(-h\nu/kT)}{D^2}$$

SXT(soft X-ray)와 BCS는 다른 $T$ 영역에 민감 — SXT는 ~10⁶–10⁷ K coronal plasma, BCS Fe XXVI는 superhot $T > 3\times 10^7$ K.

SXT samples ~10⁶–10⁷ K coronal plasma; BCS Fe XXVI is sensitive to superhot T > 3×10⁷ K plasmas.

### 4.4 Orbital Mechanics / 궤도 역학

Kepler's third law for circular LEO:
$$T = 2\pi \sqrt{\frac{(R_\oplus + h)^3}{GM_\oplus}}$$

대입 / Substituting $h = 600$ km, $R_\oplus = 6378$ km, $GM_\oplus = 3.986\times 10^{14}$ m³/s²:
$$T = 2\pi \sqrt{\frac{(6.978\times 10^6)^3}{3.986\times 10^{14}}} \approx 5800~\text{s} \approx 96.7~\text{min}$$

논문의 97 min 일치. 일조/일식 비율은 31° 경사각에서 시간에 따라 변동하지만 평균적으로 ~60% 일조 (~58분 day, ~39분 night).

Matches the paper's quoted 97-min period. At 31° inclination the sunlit fraction is ~60% on average (~58 min day, ~39 min night per orbit).

### 4.5 Telemetry Budget / 텔레메트리 예산

기록기 수명 / Recorder fill time at high rate:
$$\tau_{\text{fill}} = \frac{C_{\text{BDR}}}{R_{\text{high}}} = \frac{10\times 8\times 10^6 \text{ bits}}{32\times 10^3 \text{ bits/s}} = 2500~\text{s} \approx 41.7~\text{min}$$

논문의 ~40 min 일치. 일조 60+분이므로 overwrite 우선순위 로직 필수.

Matches the paper's ~40 min. With 60+-min daylit passes, overwrite-priority logic is unavoidable.

지상국 덤프 시간 / Dump time at 262 kbps:
$$\tau_{\text{dump}} = \frac{10\times 8 \times 10^6}{262\times 10^3} = 305~\text{s} \approx 5~\text{min}$$

10분 접촉 윈도우 내에서 안정적으로 가능. Comfortably fits within 10-min contact windows.

### 4.6 Frame Allocation / 프레임 할당

Flare-mode 128-byte frame:

$$N_{\text{basic}} + N_{\text{HXT}} + N_{\text{SXT}} + N_{\text{WBS}} + N_{\text{BCS}} = 32 + 16 + 64 + 8 + 8 = 128~\text{bytes}$$

Quiet-mode: SXT 64+16=80 byte (full + partial), 32 + 0 + 80 + 8 + 8 = 128 byte. (HXT는 'basic' 32바이트 안에 lowest-band 데이터로 항상 존재.)

In quiet mode, SXT takes 80 bytes (full+partial frame) and HXT lowest-band data is folded into the 32-byte basic section.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1962 ─── First solar X-ray observations (Aerobee rockets, Friedman)
1968 ─── OSO-3 first hard X-ray flare detection
1973–74 ─ SKYLAB ATM (S-054 grazing-incidence soft X-ray telescope, Vaiana)
1977 ─── Tanaka & Zirin: type-A superhot plasmas reported
1980 ─── SMM launched (Solar Maximum Mission, NASA)
1981 ─── HINOTORI launched (Japan, hard X-ray modulation collimator imaging)
1983 ─── Tanaka et al. blueshifts in Fe XXV / Ca XIX (HINOTORI)
1983 ─── Forrest & Chupp / Nakajima et al. simultaneous γ-ray + hard X-ray
1984 ─── Tsuneta et al. confined hot looptop sources
1987 ─── Tanaka K. PASJ flare classification (type A/B/C)
1988 ─── Uchida & Shibata pre-impulsive dynamical phase hypothesis
─────► 1991 ─ SOLAR-A → YOHKOH launched (Aug 30)  *** THIS PAPER ***
1991 ─── Solar Phys. 136 special issue (Kosugi HXT, Tsuneta SXT, Culhane BCS,
         Yoshimori WBS, Morrison data analysis)
1992 ─── First YOHKOH science (Tsuneta et al. 1992 SXT, cusp loops)
1994 ─── Masuda et al. above-the-loop-top hard X-ray source
1996 ─── Tsuneta cusp-shaped reconnection (YOHKOH SXT)
2001 ─── YOHKOH ends (Dec, battery failure during eclipse)
2002 ─── RHESSI launched (NASA, hard X-ray imaging spectroscopy)
2006 ─── Hinode (SOLAR-B) launched, successor to YOHKOH
2010 ─── SDO launched (full-disk EUV/magnetograms)
2020 ─── Solar Orbiter launched (ESA/NASA)
```

YOHKOH는 두 태양 극대기를 잇는 결정적 임무였다. SMM/HINOTORI 시대의 의문을 해결하고, Hinode/RHESSI 시대의 표준 모델(자기 재결합 + 고온 looptop + footpoint)을 확립.

YOHKOH bridges two solar maxima, resolving the questions posed by SMM/HINOTORI and establishing the standard flare model (reconnection + hot looptop + footpoints) that defined the Hinode/RHESSI era.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Vaiana et al. (1973) SKYLAB X-ray telescope | First grazing-incidence solar X-ray imager — direct technological ancestor of SXT / SXT의 직계 기술적 선조 | High |
| Kondo (1983) HINOTORI proceedings | Predecessor mission's data-priority logic explicitly inherited by SOLAR-A DP / DP가 명시적으로 계승한 데이터 우선순위 로직 | High |
| Tanaka K. (1987) PASJ flare classification | type A/B/C flare taxonomy that SOLAR-A is designed to test and refine / SOLAR-A가 검증·정교화 목표로 하는 분류 체계 | High |
| Tsuneta et al. (1984) ApJ confined hot sources | HINOTORI result motivating SOLAR-A's high-resolution coordinated imaging / SOLAR-A 협업 영상의 직접 동기 | High |
| Uchida & Shibata (1988) pre-impulsive phase | Theoretical hypothesis SOLAR-A is built to verify / SOLAR-A가 검증 대상으로 하는 이론 가설 | High |
| Kosugi et al. (1991) Solar Phys. 136, 17 | HXT instrument paper — companion in same special issue / 같은 특집호 HXT 기기 논문 | Highest |
| Tsuneta et al. (1991) Solar Phys. 136, 37 | SXT instrument paper — companion / SXT 기기 논문 동반자 | Highest |
| Culhane et al. (1991) Solar Phys. 136, 89 | BCS instrument paper — companion / BCS 기기 논문 | Highest |
| Yoshimori et al. (1991) Solar Phys. 136, 69 | WBS instrument paper — companion / WBS 기기 논문 | Highest |
| Morrison et al. (1991) Solar Phys. 136, 105 | Data analysis system paper — companion / 데이터 분석 시스템 논문 | High |
| Forrest & Chupp (1983) Nature γ-ray timing | SMM result motivating SOLAR-A WBS GRS / WBS GRS의 동기 SMM 결과 | Medium |
| Masuda et al. (1994) above-the-loop source | Future YOHKOH discovery enabled by HXT / HXT가 가능케 한 미래 YOHKOH 발견 | High |
| Kosugi & Acton et al. (1992) early YOHKOH | Initial mission results fulfilling SOLAR-A objectives / SOLAR-A 목표를 실현한 초기 결과 | High |

---

## 7. References / 참고문헌

- **Primary / 주논문**: Ogawara, Y., Takano, T., Kato, T., Kosugi, T., Tsuneta, S., Watanabe, T., Kondo, I., Uchida, Y. (1991). "The SOLAR-A Mission: An Overview", *Solar Physics* **136**, 1–16. DOI: 10.1007/BF00151692

- **Companion instrument papers / 동반 기기 논문 (Solar Phys. 136 special issue)**:
  - Kosugi, T. et al. (1991). HXT instrument paper. *Solar Phys.* **136**, 17.
  - Tsuneta, S. et al. (1991). SXT instrument paper. *Solar Phys.* **136**, 37.
  - Yoshimori, M. et al. (1991). WBS instrument paper. *Solar Phys.* **136**, 69.
  - Culhane, J. L. et al. (1991). BCS instrument paper. *Solar Phys.* **136**, 89.
  - Morrison, M. et al. (1991). Data analysis system. *Solar Phys.* **136**, 105.

- **Cited prior work / 인용된 선행 연구**:
  - Antonucci, E. (1983). *Solar Phys.* **86**, 67.
  - Forrest, D. J. & Chupp, E. L. (1983). *Nature* **305**, 291.
  - Kondo, I. (1983). In Y. Tanaka et al. (eds.), *Proc. HINOTORI Symposium on Solar Flares*, ISAS, Tokyo, p. 3.
  - Kundu, M. R. & Woodgate, B. (eds.) (1986). *Energetic Phenomena on the Sun*, NASA-CP 2439.
  - Nakajima, H., Kosugi, T., Kai, K., Enome, S. (1983). *Nature* **305**, 292.
  - Tanaka, K. (1987). *Publ. Astron. Soc. Japan* **39**, 1.
  - Tanaka, K., Nitta, N., Akita, K., Watanabe, T. (1983). *Solar Phys.* **86**, 91.
  - Tsuneta, S. et al. (1984). *Astrophys. J.* **284**, 827.
  - Uchida, Y. & Shibata, K. (1988). *Solar Phys.* **116**, 291.

- **Subsequent landmark YOHKOH/Hinode results / 후속 주요 결과**:
  - Masuda, S. et al. (1994). *Nature* **371**, 495 — above-the-loop-top hard X-ray source.
  - Tsuneta, S. (1996). *Astrophys. J.* **456**, 840 — cusp-shaped reconnection.
  - Acton, L. W. et al. (1992). *Science* **258**, 618 — first SXT science.
  - Kosugi, T. et al. (2007). *Solar Phys.* **243**, 3 — Hinode (SOLAR-B) overview.
