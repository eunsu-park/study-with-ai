---
title: "An Ultraviolet Auroral Imager for the Viking Spacecraft"
authors: ["C. D. Anger", "S. K. Babey", "A. L. Broadfoot", "R. G. Brown", "L. L. Cogger", "R. Gattinger", "J. W. Haslett", "R. A. King", "D. J. McEwen", "J. S. Murphree", "E. H. Richardson", "B. R. Sandel", "K. Smith", "A. Vallance Jones"]
year: 1987
journal: "Geophysical Research Letters"
doi: "10.1029/GL014i004p00387"
topic: Space_Weather
tags: [auroral_imaging, ultraviolet, MCP, CCD, viking, instrumentation, dayglow]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 48. An Ultraviolet Auroral Imager for the Viking Spacecraft / Viking 위성을 위한 자외선 오로라 영상기

---

## 1. Core Contribution / 핵심 기여

**English** — Anger and colleagues describe the Viking V5 ultraviolet auroral imager, the first space instrument explicitly engineered to deliver global, high-resolution images of the auroral oval simultaneously on the dark and the sunlit side of Earth. The instrument is two co-bore-sighted f/1 inverse-Cassegrain Burch cameras with 25° × 20° fields of view, complementary far-UV passbands (1235–1600 Å and 1340–1800 Å), curved single-stage microchannel-plate intensifiers carrying KBr and CsI photocathodes deposited directly on the front of each MCP, and tapered fiber-optic bundles that re-image the spherical focal surface onto a planar 288 × 385 frame-transfer CCD. A combination of optical (BaF₂/CaF₂ filters in the central hole of the secondary mirror, selective Acton coatings on the secondary), detector-level (intrinsic UV-only photocathode response, ~5000× MCP visible-light scatter suppression, a ~200 Å aluminum phosphor overcoat) and mechanical (knife-edge baffle vanes, < 3 % diffuse reflectance) measures together suppress visible-light leakage by many orders of magnitude — enough to image sub-kilo-rayleigh aurora against a fully sunlit Earth. From the spinning Viking platform a 1-s effective integration is realized by stepping CCD column charges in synchronism with the image's spin-induced motion (electronic despinning), provided the columns are aligned with the spin equator to ±0.2°. A 5.1 kbps telemetry baseline (30.7 kbps bursts via the V4 plasma instrument) feeds a VAX 11/750 ground station at Kiruna for archiving and near-real-time interactive control.

**한국어** — Anger 등은 Viking V5 자외선 오로라 영상기를 설계 단계부터 기술한다. 이는 어두운 면과 태양빛이 비치는 면(sunlit Earth) 모두에서 오로라 oval을 동시에 전역(global)으로, 또 고해상도로 영상화할 목적으로 설계된 최초의 우주 기기다. 기기는 시야 25° × 20°의 광각 f/1 역(inverse) Cassegrain Burch 카메라 두 대로 구성되며, 통과대역은 1235–1600 Å와 1340–1800 Å로 상보적이고, 단단(single stage) 곡면 MCP(microchannel plate) 광증폭기를 장착해 KBr·CsI 광음극(photocathode)을 MCP 전면에 직접 증착했다. 구면 초점면을 평면 288 × 385 프레임 전송(frame-transfer) CCD에 사상하기 위해 테이퍼드 광섬유 다발을 사용한다. 광학적(부거울 중앙공의 BaF₂/CaF₂ 필터, Acton 선택 반사 코팅), 검출기 차원(광음극 자체의 UV 전용 응답, MCP 가시광 다중산란 ~5 × 10³ 억제, 형광체 위 ~200 Å Al 오버코트), 기계적(knife-edge 베인 배플, 확산 반사율 < 3 %) 대책을 결합하여 가시광 누설을 수많은 자릿수만큼 억제하며, 이로 인해 sunlit Earth 배경에서도 서브-킬로레일리(sub-kR) 오로라가 검출 가능하다. 회전하는 Viking 위성에서는 CCD 컬럼 전하를 영상 이동과 동기 이송하는 전자식 despinning으로 1초 유효 적분을 얻는다(컬럼은 스핀 적도면에 ±0.2°로 정렬). 기본 5.1 kbps 텔레메트리(필요 시 V4 채널을 빌려 30.7 kbps)로 Kiruna 지상국 VAX 11/750으로 전송, 보관·실시간 제어된다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Mission setting and design considerations / 임무 배경과 설계 고려사항

**English** — The 1980 Swedish Space Corporation decision committed to a high-eccentricity polar orbit specifically to dwell over the 15 000 km auroral acceleration region. Because optical context is essential for interpreting in-situ particle/field experiments, the mission demanded an imager that could see the oval continuously — including in daylight when the auroral region is partially or fully sunlit. The introduction lists eight numbered considerations that frame every design choice:

1. **Inheritance** — V5 builds on ISIS-II (Anger 1973), DMSP (Rogers 1974), Kyokko (Hirao 1978), Dynamics Explorer (Frank 1981), and HILAT (Meng 1984). Each prior instrument informed a specific subsystem: ISIS-II demonstrated UV photometry from a spinning platform; DE-1 SAI demonstrated true global imaging; HILAT proved UV imaging works under full sunlight.
2. **Wavelength choice** — Visible imaging is impossible on the dayside; even nightside imaging suffers because the sunlit limb scatters into the optics. UV wavelengths below ~1800 Å exploit low atmospheric reflectance.
3. **Optics** — A 25° FOV, fast (f/1), reflecting design with the fewest possible elements. The Burch concentric two-mirror inverse-Cassegrain (radius of curvature 22.4 mm focal sphere) satisfies these.
4. **Two passbands** — Imaging the LBH bands and the OI 1304/1356 Å multiplets separately allows the precipitating electron mean energy to be inferred from intensity ratios.
5. **Curved MCPs** — A planar MCP would defocus across a 25° spherical focal surface; the MCP is curved to match (R = 22.4 mm, 24.8 mm diameter). Open intensifiers with the photocathode on the front MCP face avoid an entrance window's UV losses but require permanent vacuum/dry-N₂ environment because KBr and CsI are hygroscopic.
6. **Single-stage MCP** — Adequate gain (~2 × 10⁴ at 1000 V) without the resolution / lifetime cost of multi-stage designs. Direct fiber-optic coupling between MCP output and CCD; the same fiber assembly performs distortion correction.
7. **One-second exposure on a spinning satellite** — Achieved by stepping CCD pixel charges along columns in synchrony with image motion. The CCD columns must be aligned to the spin equator within ±0.2° or smearing dominates.
8. **Operational flexibility** — The instrument can be re-pointed, re-windowed, and re-binned in near real time from Kiruna, exploiting a reference pulse from the spacecraft's limb / sun sensors to trigger exposures at any spin phase.

**한국어** — 1980년 스웨덴 우주공사의 결정은 15 000 km 오로라 가속 영역 위에서 오래 머무르는 고이심률 극궤도를 채택했다. 입자·자기장 동시 측정을 해석하려면 광학적 맥락이 필요하므로 oval을 연속 영상화하는 기기가 요구됐다. 서론은 설계 결정을 이끄는 8가지 고려사항을 번호로 정리한다.

1. **선행 자산 계승** — ISIS-II(Anger 1973), DMSP(Rogers 1974), Kyokko(Hirao 1978), Dynamics Explorer(Frank 1981), HILAT(Meng 1984) 경험을 흡수. 각각 회전 플랫폼 UV 측광, 전역 영상화, sunlit 영상화의 선례를 제공.
2. **파장 선택** — 가시광 영상화는 dayside에서 불가능하며 nightside에서도 sunlit 림(limb) 산란이 광학계로 들어와 곤란. ~1800 Å 이하는 대기 반사율이 낮아 유리.
3. **광학계** — 25° 시야, 빠른 광학(f/1), 반사 광학 + 최소 요소. 두 구면거울 동심 역 Cassegrain Burch 광학(초점 구면 R = 22.4 mm)으로 충족.
4. **두 통과대역** — LBH 밴드와 OI 1304/1356 Å 다중선을 분리 영상하여 강하 전자의 평균 에너지를 비율로 추정.
5. **곡면 MCP** — 평판 MCP는 25° 구면 초점에서 초점이 흐려진다. R = 22.4 mm, 직경 24.8 mm로 곡면화. KBr·CsI 광음극이 흡습성이라 진공/건조 N₂ 환경에서만 다룬다.
6. **단단 MCP** — 1000 V에서 ~2 × 10⁴ 이득으로 충분. 다단 MCP의 분해능·수명 손실을 회피. MCP 출력과 CCD를 광섬유로 직접 결합, 같은 다발이 왜곡 보정도 수행.
7. **회전 위성에서 1초 노출** — CCD 픽셀 전하를 영상 이동과 동기 이송. 단, CCD 컬럼은 스핀 적도면에 ±0.2°로 정렬해야 번짐이 없음.
8. **운영 유연성** — Kiruna 지상국에서 거의 실시간으로 가리킴, 윈도우, 비닝(binning) 변경. 림·태양 센서 펄스로 임의 스핀 위상에서 노출 트리거.

### Part II: Optics and image mapping / 광학과 영상 사상

**English** — The Burch concentric optical system uses two spherical mirrors with a common center of curvature, originally proposed in 1947 as a reflecting microscope objective. With f/1 and a 25° FOV the design gives essentially perfect spherical-aberration correction (the mirrors are concentric so the chief ray is symmetric) and minimal off-axis aberrations because every field point sees an equivalent symmetric system. The penalty: the focal surface is a sphere of radius equal to the mirror radius (here 22.4 mm). Resolution is limited by the intensifier to ~0.076°.

The concentric design also implies that the mapping from sky to sphere is angularly true: each angular pixel on the sphere corresponds to the same solid angle on the sky. If the spherical image were projected onto a planar CCD by parallel fibers, the result would be the orthographic projection of a globe onto a plane, with severe distortion and a multi-pixel point-spread function near the field edges. To eliminate this, the V5 optical chain inserts a tapered fiber-optic distortion corrector between the curved MCP output (covered with the phosphor screen) and the planar CCD. The entrance face of the corrector is figured to match the sphere; the exit face is flat; interior fibers curve to map sphere to plane while preserving local fiber-to-fiber neighbor relations. Since spin compensation works only if every image point moves in a straight column at uniform rate, distortion correction is critical for the despinning scheme.

**한국어** — Burch 동심 광학계는 두 구면거울이 공통 곡률중심을 갖는다(1947년 반사 현미경 대물렌즈). f/1, 25° 시야에서 구면 수차는 동심성에 의해 거의 완전히 보정되고, off-axis 수차도 모든 시야점이 동일한 대칭계를 보므로 최소화된다. 대가는 초점면이 거울과 동일 반지름의 구면(여기 R = 22.4 mm)이라는 점. 전체 분해능은 광증폭기에 의해 ~0.076°로 제한된다.

동심 설계는 하늘 → 구면 사상이 각도적으로 진실(angular-true)임을 뜻한다. 만약 평행 광섬유로 그 구면을 평면 CCD에 투사하면 결과는 지구의 위경도를 평면 지도에 정사 투영(orthographic)한 것과 같아, 시야 가장자리에서 점확산함수(PSF)가 수 픽셀로 퍼진다. V5는 곡면 MCP 출력(형광체) 뒤에 테이퍼드 광섬유 왜곡 보정기를 둔다. 입사면은 구면, 출사면은 평면, 내부 광섬유는 구면→평면 사상에 맞춰 곡선 경로로 배치된다. 회전 보정이 가능하려면 각 영상 점이 한 컬럼을 따라 직선·등속으로 움직여야 하므로 왜곡 보정은 필수적이다.

### Part III: Thermal stability / 열적 안정성

**English** — V5 must operate down to −30 °C but is focused at room temperature. The Burch design is sensitive to mirror separation. Two material choices keep it in focus across that range: a graphite-fiber-epoxy composite (GFEC) housing with near-zero CTE along the optic axis, and Zerodur mirrors (also very low CTE). The dust cover is removed several hours after orbit insertion; until then a slight positive nitrogen pressure protects the hygroscopic photocathodes.

**한국어** — V5는 −30 °C까지 작동해야 하지만 초점 조정은 상온에서만 한다. Burch 설계는 거울 간격에 민감하므로, 광축 방향 열팽창이 거의 0인 GFEC(흑연 섬유 에폭시 복합) 하우징과 마찬가지로 매우 낮은 CTE의 Zerodur 거울을 사용해 온도 범위 전체에서 초점을 유지한다. 먼지 덮개는 궤도 진입 수 시간 후 분리되며, 그 전까지는 흡습성 광음극 보호를 위해 미양압 N₂ 환경을 유지한다.

### Part IV: Camera passbands and dayglow rejection / 카메라 통과대역과 dayglow 억제

**English** — The two cameras differ only in their passband.

- **Camera 0** (1340–1800 Å): BaF₂ filter + CsI photocathode. The BaF₂ filter cuts off around 1340 Å (its short-λ edge) and the CsI cathode falls steeply above 1800 Å. The passband emphasizes LBH bands and the 1493 Å NI line; OI 1304 Å is suppressed. Because Camera 0 was prioritized for sensitivity in the far-UV, its secondary mirror is overcoated only with aluminum (highest reflectance).
- **Camera 1** (1235–1600 Å): CaF₂ filter + KBr photocathode + a selective reflective coating (Acton Research) on the secondary mirror that further attenuates wavelengths longer than ~1600 Å. The passband captures the OI 1304 Å resonance triplet, the 1356 Å forbidden multiplet, and some short-wavelength LBH bands. Because Camera 1's window includes the bright 1304 Å Earth dayglow, the extra attenuation is needed to keep visible-light leakage from dominating in IBC II conditions.

The filters sit in the central hole of the secondary mirror — a clever placement that avoids any image-quality penalty because the central obscuration is already lost. They are polished for low UV scattering. The photocathodes set the long-wavelength cutoff and contribute the bulk of the visible-light rejection: residual quantum efficiency at 3000 Å is below 2 × 10⁻⁸, falling 10× per 290 Å. Without further suppression the residual visible signal from the sunlit Earth would still be comparable to an IBC II auroral signal (~10 kR) for Camera 1 — hence the additional Acton coating.

The authors acknowledge a limitation: under daylit conditions the OI 1304 Å dayglow contaminates Camera 1 so badly that the LBH/OI ratio cannot reliably yield mean electron energy on the dayside. Preliminary observations of dayglow at 1304 Å are consistent with expected fluxes, validating the radiometry.

**한국어** — 두 카메라는 통과대역만 다르다.

- **Camera 0**(1340–1800 Å): BaF₂ 필터 + CsI 광음극. BaF₂는 ~1340 Å에서 단파 컷오프, CsI는 1800 Å 이상에서 급락. LBH 밴드와 1493 Å NI를 강조, OI 1304 Å은 억제. 원-UV 감도 우선으로 부거울은 알루미늄만 코팅(최대 반사율).
- **Camera 1**(1235–1600 Å): CaF₂ 필터 + KBr 광음극 + Acton 선택 반사 코팅(부거울)으로 ~1600 Å 이상을 추가 감쇠. OI 1304 Å, 1356 Å 그리고 일부 단파 LBH를 잡는다. 1304 Å dayglow가 강하므로 추가 감쇠 없이는 IBC II 신호 수준의 가시광 누설이 발생할 수 있어 Acton 코팅이 필수.

필터를 부거울 중앙공에 배치한 것은 영상 품질에 영향을 주지 않는(이미 가려진 부분이므로) 영리한 결정이며, 필터 자체는 저 UV 산란용으로 연마됐다. 광음극이 장파 컷오프와 가시광 거부 대부분을 담당: 3000 Å 잔류 양자효율 ≤ 2 × 10⁻⁸, 이후 290 Å마다 10배씩 감소. 그래도 sunlit Earth로부터의 잔류 가시광이 IBC II 오로라 신호(~10 kR)와 비슷할 수 있어 Acton 코팅이 필요했다. 저자들은 한계도 인정한다: dayside에서는 1304 Å dayglow가 Camera 1을 오염시켜 LBH/OI 비율로 강하 전자 평균 에너지를 추출할 수 없다. 1304 Å dayglow의 초기 관측은 예측 플럭스와 일치해 측광 보정의 타당성을 입증.

### Part V: Baffle system / 배플 시스템

**English** — When the Sun's vector lies < 90° from the optic axis, off-axis light scatters into the optics. The baffle prevents the Sun from directly illuminating the filter, the mirrors, or the stop at the filter edge. Knife-edge vanes block the worst direct paths, leaving two residual scatter sources:

1. Scatter from the tips of the knife edges themselves.
2. Multiple scatter from the front of one vane to the back of the preceding vane.

Following Breault (1977), if the diffuse reflectance of vane and tube surfaces is below 3 %, the multiple-scatter contribution is held below the knife-edge contribution. Quantitatively, with all knife edges directly illuminated and a 5 % diffuse reflectance at the knife edges, total baffle scatter is < 20 % of the IBC II signal for Camera 1; for Camera 0 (no Acton coating) the scatter could be comparable to the auroral signal in this most unfavorable case. The critical sun angle for the baffle is ~45°. Operationally this means baffle scatter limits the noon-midnight sector of the orbit.

**한국어** — 태양 벡터가 광축에 대해 90° 미만일 때 외곽 광이 광학계로 산란된다. 배플은 태양이 필터, 거울, 필터 가장자리 stop을 직접 비추지 못하게 한다. Knife-edge 베인이 직접 경로를 막은 뒤 남는 잔류 산란은 두 가지:

1. Knife-edge 자체 끝에서의 산란.
2. 한 베인 앞면에서 직전 베인 뒷면으로의 다중 산란.

Breault(1977)에 따르면 베인·튜브 표면 확산 반사율이 3 % 미만이면 다중 산란 기여가 knife-edge 기여 아래로 억제된다. 모든 knife-edge가 직접 비춰지고 5 % 확산 반사율이라는 최악 조건에서도 Camera 1의 총 배플 산란은 IBC II 신호의 20 % 미만; Camera 0(Acton 코팅 없음)은 이 최악 조건에서 신호와 비슷할 수 있다. 배플 임계 태양각은 ~45°이며, 운영상 noon-midnight sector 시야가 제한된다.

### Part VI: Image intensifiers / 영상 광증폭기

**English** — Each intensifier is windowless and curved (R = 22.4 mm, 24.8 mm diameter). The housing is Macor machinable ceramic; the MCP and phosphor are held by ceramic rings; flat circular electrodes contact MCP and phosphor edges. The output faceplate (permanently bonded) is a fiber-optic distortion corrector with a concave depression matching the MCP back surface, on which the P-20 phosphor is deposited. A −25 V screen in front of the MCP, paired with a +25 V MCP front, modestly excludes low-energy charged particles and recovers escaped photoelectrons.

The custom MCP from Galileo Electro-Optics has 12 µm channels parallel to the camera optic axis. Because rays at the spherical focal surface are not parallel to the optic axis, visible photons from the image must travel down channels at an angle and undergo multiple scattering — reducing visible transmission by ~5000×. This is a fortuitous byproduct of the curved geometry: the same physics that demands distortion correction also kills visible light.

Single-stage gain is ~2 × 10⁴ at 1000 V, consistent with L/D ≈ 40 channels and pulse-height-distribution measurements. Photocathode deposition on the MCP front did not measurably reduce gain. Electrons accelerate through 3500 V to the phosphor, where the visible image forms. A ~200 Å aluminum overcoat over the P-20 phosphor (with an underlying tin-oxide conduction layer) prevents visible light scattered through the MCP channels from reaching the CCD; about 2000 eV is lost per electron in penetrating this overcoat.

KBr and CsI cathodes are hygroscopic. The MCPs were never exposed to air after coating: they were stored in vacuum, transported in N₂-purged sealed containers, transferred via N₂-purged bags, and protected by a continuous N₂ purge once installed. This logistics burden is a recurring theme in UV instrumentation.

**한국어** — 각 광증폭기는 창이 없고 곡면(R = 22.4 mm, ⌀ 24.8 mm)이다. 하우징은 Macor 가공 세라믹, MCP와 형광체는 세라믹 링으로 고정, 평판 원형 전극이 MCP·형광체 가장자리에 접촉. 출력 페이스플레이트(영구 접합)는 광섬유 왜곡 보정기로, MCP 뒷면 곡률에 맞춘 오목면에 P-20 형광체가 증착됐다. MCP 앞 −25 V 스크린과 +25 V MCP 전면이 조합되어 저에너지 하전입자를 약하게 차단, 또한 photoelectron을 MCP로 다시 밀어 넣는다.

Galileo Electro-Optics의 맞춤 MCP는 12 µm 채널을 카메라 광축 방향으로 정렬했다. 구면 초점에서는 광선이 광축과 나란할 수 없으므로 가시광 광자는 채널 안에서 사선으로 진행하며 다중 산란하여 ~5000배 감쇠된다 — 곡면 기하의 행운의 부산물(왜곡 보정 필요와 가시광 억제가 동일 물리에서 비롯). 단단 이득은 1000 V에서 ~2 × 10⁴, L/D ≈ 40 채널과 일치하고 펄스높이 분포로도 검증. 광음극 증착은 이득에 영향이 없었다.

전자가 3500 V로 형광체로 가속되어 가시 영상을 형성. P-20 형광체 위 ~200 Å Al 오버코트(아래는 SnO₂ 도전층)가 MCP 채널을 거쳐온 가시광이 CCD에 도달하는 것을 차단; 전자는 이 오버코트 통과 시 약 2000 eV를 잃는다.

KBr·CsI 광음극은 흡습성이라 한 번도 공기에 노출되지 않는다. 진공 보관, N₂ 퍼지 밀폐 용기 운반, N₂ 퍼지 가방 안에서 전달, 설치 후 지속적 N₂ 퍼지 — UV 기기에서 반복되는 운영 부담이다.

### Part VII: Detectors and electronic despinning / 검출기와 전자식 despinning

**English** — The CCD is an English Electric Valve P8602: three-phase frame-transfer, 288 rows × 385 columns in both image and storage regions. CCDs were procured without windows and bonded directly to the fiber-optic blocks. The aluminum mask blocks the storage region. Effective pixel size projected onto the MCP is 30 µm square, giving 0.076° angular resolution.

Spin compensation works because frame-transfer CCDs already step charge along columns. Normally this happens fast at end-of-exposure; here it happens slowly during the exposure, matched to the angular rate of the spinning satellite. The trick: as the image moves across the focal plane in the spin direction, the CCD steps the accumulated photoelectrons in the same direction at the same rate. Each scene element collects light for 1 s of effective exposure, even though the satellite is rotating. The optical chain must therefore deliver each scene element as a straight line down a column at uniform rate — this is precisely why the fiber-optic distortion corrector exists, and why the CCD columns must be aligned with the spin equator within ±0.2°.

Quantitative photometric chain (paper's preliminary numbers):
- Sensitivity: 0.1–0.4 dn per photoelectron (8-bit dn)
- Detectability: a few photoelectrons per pixel-spin
- Predicted: sub-kilo-rayleigh aurora detectable in 1 s

**한국어** — CCD는 English Electric Valve P8602: 3상 프레임 전송, 영상과 저장 영역 모두 288행 × 385열. 윈도우 없이 광섬유 블록에 직접 본딩, Al 마스크가 저장 영역 차폐. MCP에 투영된 유효 픽셀 크기 30 µm 사각, 각도 분해능 0.076°.

스핀 보정은 프레임 전송 CCD가 본래 컬럼 방향 전하 이송 능력을 가졌다는 점을 이용한다. 평소엔 노출 종료 시 고속 이송이지만, 여기선 노출 중에 위성 회전각속도에 맞춰 천천히 이송한다. 영상이 스핀 방향으로 초점면 위를 움직일 때 CCD도 같은 방향·같은 속도로 누적 전자를 이송하면 각 장면 요소가 1초 유효 노출을 얻는다. 광학계는 각 장면 요소를 한 컬럼을 따라 직선·등속으로 전달해야 하므로 광섬유 왜곡 보정이 필수이며, CCD 컬럼은 스핀 적도면에 ±0.2° 이내로 정렬되어야 한다.

측광 체인 정량(논문 예비치):
- 감도: 0.1–0.4 dn / photoelectron (8-bit dn)
- 검출 한계: 픽셀-스핀당 수 photoelectron
- 예측: 1초 노출에서 sub-kR 오로라 검출 가능

### Part VIII: Electronics, telemetry, and operations / 전자부, 텔레메트리, 운영

**English** — Five electronics blocks: spacecraft interface (uplink/downlink, limb/sun trigger), CPU (bit-slice, 1024 × 32-bit microcoded RISC, uploadable), 48 kB radiation-hardened CMOS RAM (Harris 8 kB modules), CCD analog chain (correlated double sampling, selectable gain, 8-bit ADC), and power supply (low-voltage switching synced to spacecraft 54 kHz clock; 750–950 V eight-step MCP HV; fixed 4500 V phosphor HV).

Telemetry: nominal 5.1 kbps direct downlink; 30.7 kbps when V4 (plasma instrument) lends spare channels. Five imaging-parameter lists are stored, including a default (power-up) list. Each list specifies: ID tag, MCP HV setting (intensifier gain), camera select (0, 1, or both), camera video gain, vertical clock frequency (matched to instantaneous spin rate), windowed image dimensions and offset, viewing-angle rotation delay (relative to limb or sun reference), pixel binning factors, repeat factor. Instrument software cycles the active list, executing each definition repeatedly. Lists can be uploaded during a Kiruna pass, providing near-real-time operational flexibility.

**한국어** — 다섯 전자부 블록: 우주선 인터페이스(업링크/다운링크, 림·태양 트리거), CPU(bit-slice, 1024 × 32-bit 마이크로코드 RISC, 업로드 가능), 48 kB 방사선 강화 CMOS RAM(Harris 8 kB 모듈), CCD 아날로그(correlated double sampling, 가변 이득, 8-bit ADC), 전원(우주선 54 kHz와 동기된 저전압 스위칭; 8단 750–950 V MCP HV; 고정 4500 V 형광체 HV).

텔레메트리: 기본 5.1 kbps 직접 다운링크, V4 채널을 빌릴 때 30.7 kbps. 5개의 영상 파라미터 리스트(전원 기본 1개 포함). 각 리스트는 ID, MCP HV(이득), 카메라 선택(0/1/양쪽), 비디오 이득, 수직 클록(스핀 속도 보정), 윈도우 크기·오프셋, 시야각(림·태양 기준 회전 지연), 비닝, 반복 횟수를 정의한다. 소프트웨어는 활성 리스트를 순환하며 각 정의를 반복 실행. Kiruna 통과 중 리스트를 업로드해 거의 실시간 운영 유연성을 제공.

---

## 3. Key Takeaways / 핵심 시사점

1. **Solar-blind UV is the only way to see global aurora in daylight.** — Below ~1800 Å, atmospheric reflectance is so low and Rayleigh scattering at 1304 Å so self-absorbed that a sunlit Earth becomes a dark background and aurora dominates. V5 is the architecture that exploits this. / **데이라이트 전역 오로라는 solar-blind UV에서만 본다.** ~1800 Å 이하에서 대기 반사율이 낮고 1304 Å 산란은 자기흡수되어 sunlit Earth가 어두운 배경이 된다. V5는 이 물리를 활용한다.

2. **Concentric optics buy aberration correction at the cost of a spherical focal surface.** — The Burch f/1 inverse-Cassegrain corrects spherical and off-axis aberrations almost perfectly, but its focal surface is a sphere. Every downstream subsystem (curved MCP, fiber taper) exists to manage that consequence. / **동심 광학은 수차를 보정하는 대신 곡면 초점을 강요한다.** Burch f/1 역 Cassegrain은 수차를 거의 완벽히 보정하지만 초점면이 구면이다. 곡면 MCP, 테이퍼 광섬유 등 후단 모두가 이 결과를 처리한다.

3. **Distortion-free remapping enables electronic despinning.** — If the spherical-to-planar mapping introduced any column-direction distortion, charge-shift exposures would smear. The tapered fiber bundle is the linchpin that lets the CCD step charges in straight lines at uniform rate. / **무왜곡 사상이 전자식 despin을 가능케 한다.** 사상에 컬럼 방향 왜곡이 있으면 전하 이송 노출이 번진다. 테이퍼드 광섬유가 컬럼 직선·등속을 보장하는 핵심이다.

4. **Visible-light rejection is built in layers, each cheap, the product enormous.** — Filter (~10⁴), photocathode (≤2 × 10⁻⁸ at 3000 Å), MCP scatter (~5 × 10³), Al overcoat on phosphor (extra blocking). Multiplied, the visible-light suppression exceeds ~10¹⁵, enough to stare at a sunlit Earth and still see sub-kR aurora. / **가시광 거부는 층층이, 각 층은 싸지만 곱은 거대하다.** 필터(~10⁴), 광음극(3000 Å에서 ≤ 2 × 10⁻⁸), MCP 다중산란(~5 × 10³), 형광체 위 Al 오버코트가 곱해져 ~10¹⁵ 억제, sunlit Earth에서도 sub-kR 검출.

5. **Stray light is engineered, not optical.** — The 25° FOV with Sun within 90° of the boresight forces a sophisticated knife-edge baffle. Breault's diffuse-reflectance < 3 % rule and the 45° critical sun-angle limit set hard operational constraints (no quality dayside images during noon-midnight orbit phases). / **미광은 광학이 아니라 공학으로 막는다.** 25° 시야 + 광축 90° 안의 태양은 정교한 knife-edge 배플을 강요한다. Breault의 < 3 % 확산반사 규칙과 45° 임계 태양각이 운영 제약(noon-midnight 위상에서는 dayside 영상 품질 제한)을 정한다.

6. **Two passbands provide spectroscopic context, but day/night break the symmetry.** — The LBH/OI ratio gives mean precipitating-electron energy on the nightside but is corrupted by 1304 Å dayglow on the dayside. Future missions add narrowband 1356 Å channels to recover dayside diagnostics. / **두 통과대역이 분광 맥락을 주나 day/night 비대칭이 있다.** LBH/OI 비율은 nightside에서 평균 전자 에너지를 주지만 dayside에서는 1304 Å dayglow에 오염. 후속 임무는 1356 Å 협대역 채널로 dayside 진단을 회복.

7. **Hygroscopic photocathodes dictate logistics.** — KBr and CsI cathodes never see air after deposition. The required vacuum-handling, N₂-purge containers, and on-orbit dust-cap-then-purge-removal sequence are not add-ons — they are mission-critical operations that flow from a single material choice. / **흡습성 광음극이 운영을 정한다.** 증착 후 KBr·CsI는 공기 노출 금지. 진공 운반, N₂ 밀폐 용기, 궤도 진입 후 N₂ 양압 → 먼지 덮개 분리 순서는 부가 절차가 아니라 핵심 운영이다.

8. **Software-defined operations were already needed in 1986.** — Five uploadable parameter lists, near-real-time re-pointing and re-windowing from Kiruna, and limb/sun-triggered exposures anticipate today's commandable, software-defined space instruments. / **소프트웨어 정의 운영은 이미 1986년에 필요했다.** 업로드 가능 파라미터 리스트 5개, Kiruna에서의 거의 실시간 가리킴/비닝 변경, 림·태양 트리거 노출은 오늘날 명령 기반·SW 정의 우주 기기의 선구.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Pixel scale / 픽셀 스케일

For a small pixel of physical size $p$ at focal length $f$:
$$\theta_{\text{pix}} = 2\,\arctan\!\left(\frac{p}{2f}\right) \approx \frac{p}{f}\quad (\text{rad})$$
With $p = 30~\mu\text{m}$ and $f = 22.4~\text{mm}$: $\theta_{\text{pix}} \approx 1.34 \times 10^{-3}~\text{rad} \approx 0.0768^{\circ}$, matching the paper's quoted 0.076°. The detector pixel grid (288 × 385) covers $\sim 22^{\circ} \times 29^{\circ}$ — slightly larger than the 20° × 25° usable FOV after vignetting/baffle masking.
물리 픽셀 $p$, 초점거리 $f$ → 픽셀 시야각. p = 30 µm, f = 22.4 mm로 0.0768°이며 논문 0.076°와 일치. 288 × 385 격자가 약 22° × 29°를 덮어 통상 사용 시야 20° × 25°를 포함.

### 4.2 Optical throughput / 광학 처리량

Effective collecting area for an unobstructed circular aperture is $A_{\text{eff}} = \pi (D/2)^2$ where $D = f/N$. For f/1 and $f = 22.4~\text{mm}$, $D = 22.4~\text{mm}$ and $A = \pi(11.2)^2 \approx 394~\text{mm}^2 = 3.94~\text{cm}^2$. With the secondary central obscuration and filter aperture this reduces to roughly $A_{\text{eff}} \sim 2~\text{cm}^2$, but the f/1 design still maximizes per-pixel signal.
원형 무차폐 개구의 유효 면적 = π(D/2)². f/1, f=22.4 mm → D=22.4 mm, A≈3.94 cm². 부거울 중앙공·필터로 실제로는 약 2 cm²지만 f/1 설계가 픽셀당 신호를 극대화.

### 4.3 Auroral signal rate per pixel / 픽셀당 오로라 신호율

Expressing column emission rate in rayleighs ($1~\text{R} = 10^{6}/(4\pi)~\text{photons cm}^{-2}~\text{s}^{-1}~\text{sr}^{-1}$) the photoelectron rate per pixel is
$$\dot N_{\text{pe}} = I_{\text{R}}\cdot\frac{10^{6}}{4\pi}\cdot \Omega_{\text{pix}}\cdot A_{\text{eff}}\cdot \tau_{\text{filter}}\cdot \eta_{\text{QE}}$$
with $\Omega_{\text{pix}} = \theta_{\text{pix}}^{2}$ (sr), $\tau_{\text{filter}}$ filter transmission, $\eta_{\text{QE}}$ photocathode QE.

For $\theta_{\text{pix}} = 0.076^{\circ} = 1.33 \times 10^{-3}~\text{rad}$, $\Omega_{\text{pix}} = 1.78 \times 10^{-6}~\text{sr}$. Using $A_{\text{eff}} \approx 2~\text{cm}^2$, $\tau_{\text{filter}} \sim 0.3$, $\eta_{\text{QE}} \sim 0.10$ and $I_{\text{R}} = 1~\text{kR}$:
$$\dot N_{\text{pe}} \approx 10^{3}\cdot\frac{10^{6}}{4\pi}\cdot 1.78\times 10^{-6}\cdot 2\cdot 0.3\cdot 0.1 \approx 8.5~\text{pe s}^{-1}$$
giving $\sim 8$ pe per 1-s exposure — comfortably above the "few pe / pixel-spin" detection criterion stated by the authors. Sub-kR aurora at the few-hundred-R level still produces ~2–3 pe / pixel-spin.
오로라 강도 $I_{\text{R}}$ → 픽셀당 광전자율. 위 수치에서 1 kR이면 1초당 ~8 pe로 검출 한계(수 pe/픽셀-스핀)를 여유롭게 넘는다. 수백 R 수준 오로라도 ~2–3 pe로 검출.

### 4.4 MCP gain / MCP 이득

For an L/D ratio $r$ and per-collision secondary-electron yield $\delta$, channel gain is approximately
$$G \approx \delta^{n},\qquad n \approx r\,\sqrt{\frac{V_{\text{collision}}}{V_{\text{MCP}}/r}}$$
Detailed Eberhardt formulae aside, Galileo's 12 µm channels with $r = 40$ at $V_{\text{MCP}} = 1000~\text{V}$ produce $G \sim 2 \times 10^{4}$, consistent with measured pulse-height distributions reported in the paper.
L/D=40의 12 µm 채널, 1000 V에서 단단 이득 ~2 × 10⁴.

### 4.5 Visible-light leakage budget / 가시광 누설 예산

The product of suppression factors at λ ≈ 5500 Å (peak human eye / Earth albedo):
$$T_{\text{vis}} = \tau_{\text{filter}}(λ)\cdot \eta_{\text{cathode}}(λ)\cdot s_{\text{MCP}}^{-1}\cdot s_{\text{Al}}^{-1}$$
With BaF₂/CaF₂ deep-UV filters $\tau_{\text{filter}}(5500) \sim 10^{-4}$ in transmission for visible (the filters are bandpass; long-λ leakage exists), CsI cathode QE $\sim 10^{-12}$ at 5500 Å (extrapolating 10×/290 Å from $2 \times 10^{-8}$ at 3000 Å), MCP scatter rejection $5 \times 10^{3}$ and Al-overcoat blocking $\sim 10^{2}$:
$$T_{\text{vis}} \sim 10^{-4}\cdot 10^{-12}\cdot \frac{1}{5\times 10^{3}}\cdot \frac{1}{10^{2}} \sim 4\times 10^{-22}$$
The dominant term is the photocathode itself; engineering layers add safety margin. The figure justifies "sub-kR aurora detectable on a sunlit Earth."
주요 항은 광음극, 다른 층은 안전 여유. sunlit Earth 배경에서도 sub-kR 검출 가능한 근거.

### 4.6 Electronic despinning kinematics / 전자식 despinning 운동학

Let $\omega$ be the spin angular rate, $f$ the focal length, and $p$ the pixel pitch. The image moves across the focal plane at $v = f\,\omega$ (m/s). The CCD must shift charge along columns at the same rate. The vertical clock frequency is
$$f_{\text{vclk}} = \frac{f\,\omega}{p}\quad [\text{rows/s}]$$
For Viking $\omega \sim 3$ rpm $= 0.314$ rad/s, $f = 22.4~\text{mm}$, $p = 30~\mu\text{m}$:
$$f_{\text{vclk}} = \frac{0.0224\cdot 0.314}{3\times 10^{-5}} \approx 234~\text{rows/s}$$
A 288-row exposure is therefore $\sim 1.23$ s. Note the paper's Table 1 specifies "time resolution 20 seconds" — this is the spin period giving one image per spin, not the CCD vertical-clock period. The exposure misalignment requirement is geometric: column tilt $\beta$ smears each scene element by $\Delta x = N_{\text{rows}}\,p\,\tan\beta$ in one exposure. Setting $\Delta x \le p$ for $N_{\text{rows}} = 288$: $\tan\beta \le 1/288 \approx 0.0035 \approx 0.20^{\circ}$ — exactly the ±0.2° alignment quoted by the paper.
컬럼 정렬 오차 β의 번짐량 $\Delta x = N\,p\,\tan\beta$가 한 픽셀 이하가 되려면 $\tan\beta \le 1/288 \approx 0.0035 \approx 0.20°$로, 논문의 ±0.2° 정렬 요구와 정확히 일치.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1947 ─ Burch — concentric two-mirror f/1 reflecting microscope
   |
1953 ─ Fraser — 0.8 NA reflecting microscope (IR)
   |
1968 ─ Rosin — Inverse Cassegrainian systems
   |
1969 ─ Wynne — two-mirror anastigmat optics
   |
1971 ─ ISIS-II — Anger Scanning Auroral Photometer (visible)
   |
1973 ─ Anger et al. ISIS-II SAP paper (Applied Optics)
   |
1974 ─ DMSP — Rogers visible auroral photography
   |
1977 ─ Breault — knife-edge baffle stray-light theory
   |
1978 ─ Kyokko/EXOS-A — first space UV (1304 Å) imaging
   |
1981 ─ Frank et al. — DE-1 Spin-Scan Auroral Imager
   |
1984 ─ Meng & Huffman — HILAT UV imaging in full sunlight
   |
1986 ─ Viking V5 launch (Feb 22)
   |
─►1987 ─ Anger et al. — V5 instrument paper (THIS WORK)
   |
1996 ─ Polar UVI — 4-filter LBH/OI imager (descendant)
   |
2000 ─ IMAGE FUV (WIC, SI13 + SI12)
   |
2018 ─ NASA TIMED-GUVI / GOLD generations
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Anger et al. 1973 (ISIS-II SAP) | Predecessor instrument by the same lead author. Established UV photometric techniques on a spinning platform. / 동일 주저자의 선행 ISIS-II 기기. 회전 플랫폼 UV 측광 기법 확립. | Direct heritage / 직접 계승 |
| Frank et al. 1981 (DE-1 SAI) | First true global auroral imager (UV/visible spin-scan). V5 inherits the global-imaging philosophy but moves to dedicated UV with curved-MCP intensifiers. / 최초 본격 전역 오로라 영상기. V5는 전역 영상 철학을 계승, 전용 UV·곡면 MCP로 진화. | Conceptual ancestor / 개념적 선조 |
| Meng & Huffman 1984 (HILAT) | First proof-of-concept UV imaging in full sunlight; V5 takes the technique to a high-altitude global scale. / 최초 sunlit UV 영상 시연; V5가 고고도 전역 규모로 확장. | Technique validator / 기법 검증 |
| Burch 1947 / Wynne 1969 / Rosin 1968 | Optical heritage — concentric two-mirror f/1 anastigmat originally for microscopy, repurposed for the V5 camera. / 광학 계보 — 동심 두거울 f/1 비점수차 광학(원래 현미경), V5 카메라로 재사용. | Optical foundation / 광학 토대 |
| Breault 1977 | Stray-light theory guiding the V5 baffle design (3 % diffuse reflectance rule, knife-edge dominance). / V5 배플 설계의 산란 이론(3 % 확산반사 규칙, knife-edge 지배). | Engineering basis / 공학 기반 |
| Polar UVI (1996, Torr et al.) | Direct successor — 4-filter LBH/OI imager on Polar; design lineage clearly traceable to V5. / 직접 후계 — Polar의 4필터 LBH/OI 영상기; V5 설계 계보가 명확. | Direct descendant / 직접 후계 |
| IMAGE FUV (2000, Mende et al.) | Modernized V5 architecture: separate WIC (LBH), SI13 (1356 Å), SI12 (Lyman-α) channels solving the 1304 Å dayglow problem identified in V5. / V5의 1304 Å dayglow 문제를 해결한 현대화된 후계: WIC, SI13, SI12 분리 채널. | Conceptual completion / 개념적 완성 |

---

### Part IX: A worked example — sub-kR aurora detection / 작동 예: sub-kR 오로라 검출

**English** — Consider IBC II nightside aurora at 1356 Å, $I_{\text{R}} = 1$ kR, observed by Camera 1.

Step 1. Convert column emission rate to specific intensity. By definition $1~\text{R} = \frac{10^{6}}{4\pi}~\text{photons cm}^{-2}~\text{s}^{-1}~\text{sr}^{-1}$ for an isotropic emitter.

Step 2. Pixel solid angle. $\Omega_{\text{pix}} = (1.33 \times 10^{-3})^{2} = 1.78\times 10^{-6}~\text{sr}$.

Step 3. Geometric photon rate per pixel. For $A_{\text{eff}} = 2~\text{cm}^{2}$ (after central obscuration):
$$\dot N_{\gamma} = 1000 \cdot \frac{10^{6}}{4\pi}\cdot 1.78\times 10^{-6}\cdot 2 \approx 283~\text{photons s}^{-1}$$

Step 4. Photoelectron rate. With CaF₂ filter transmission $\tau \approx 0.4$ at 1356 Å and KBr photocathode QE $\eta \approx 0.10$:
$$\dot N_{\text{pe}} = 283\cdot 0.4\cdot 0.10 \approx 11~\text{pe s}^{-1}$$

Step 5. Digital signal. With electro-optic gain $g = 0.2$ dn/pe (mid-range of 0.1–0.4):
$$S = 11~\text{pe}\cdot 0.2 \approx 2.2~\text{dn / pixel-spin}$$

This is well above the read-noise floor (a few dn for an 8-bit ADC after CDS). For a 200 R aurora the signal drops to ~0.4 dn; with 2 × 2 binning ($N_{\text{eff}}=4$) per-bin signal recovers to ~1.8 dn — confirming the paper's "sub-kR detectable" claim with binning.

**한국어** — IBC II 야간 오로라 1356 Å, $I_{\text{R}} = 1$ kR, Camera 1.

1. R → 비강도: $1~\text{R} = 10^{6}/(4\pi)$ 광자 cm⁻² s⁻¹ sr⁻¹.
2. 픽셀 입체각 $\Omega_{\text{pix}} = (1.33\times 10^{-3})^{2} = 1.78\times 10^{-6}~\text{sr}$.
3. 픽셀당 광자율 ($A_{\text{eff}} = 2$ cm²): $\dot N_{\gamma} \approx 283$ photons/s.
4. 광전자율 ($\tau = 0.4$, $\eta = 0.10$): $\dot N_{\text{pe}} \approx 11$ pe/s.
5. dn 변환 ($g = 0.2$ dn/pe): $S \approx 2.2$ dn/픽셀-스핀.

8-bit CDS 후 읽기잡음 floor를 충분히 상회. 200 R 오로라는 ~0.4 dn이지만 2 × 2 비닝하면 ~1.8 dn로 회복 — sub-kR 검출 주장 검증.

### Part X: Why curved-MCP + fiber-taper became the standard / 곡면 MCP + 광섬유 테이퍼가 표준이 된 이유

**English** — Three competing approaches existed in 1986: (a) flat MCP at the focus of a slow optical system (low throughput, large optics), (b) flat MCP at the focus of a fast system with field-flattener lenses (lens absorbs UV), (c) curved MCP matched to a Burch-type focal sphere with fiber-taper readout (V5's choice). The third option preserves UV throughput (no lens absorption), keeps the optics fast and compact, and offloads the geometric remap to a passive fiber bundle with no optical loss in the visible-coupling path. This combination — a fast all-reflective UV camera + curved active focal sensor + fiber-coupled CCD — is now the canonical architecture from Polar UVI (1996) through IMAGE FUV (2000) and into modern planetary UV instruments. The V5 paper is the first place all three pieces appear together in a flight design.

**한국어** — 1986년 세 가지 경쟁 안이 있었다. (a) 느린 광학 + 평판 MCP(처리량 낮고 광학 큼), (b) 빠른 광학 + 평판 MCP + 시야평탄 렌즈(렌즈가 UV 흡수), (c) Burch형 초점 구면에 맞춘 곡면 MCP + 광섬유 테이퍼(V5 선택). 세 번째 안은 UV 처리량을 보존(렌즈 흡수 없음), 광학을 빠르고 작게 유지, 기하 사상을 가시광 결합부의 손실이 거의 없는 패시브 광섬유 다발로 떠넘긴다. 이 조합 — 빠른 전반사 UV 카메라 + 곡면 능동 초점 센서 + 광섬유 결합 CCD — 은 Polar UVI(1996), IMAGE FUV(2000), 현대 행성 UV 기기의 표준이 됐다. V5 논문이 이 세 요소가 비행 설계로 함께 등장한 최초의 사례다.

### Part XI′: Block diagram of the photometric chain / 측광 체인 블록도

```
   Sky photon
       │  (auroral or dayglow)
       ▼
[ Baffle vanes ]   ── stray light absorbed (< 3 % diffuse R)
       │
       ▼
[ Primary mirror (Al, Burch concentric) ]
       │
       ▼
[ Filter (BaF₂ / CaF₂, central hole of secondary) ]
       │
       ▼
[ Secondary mirror (Al for Cam 0; Acton selective for Cam 1) ]
       │
       ▼
[ Curved MCP front face ─ photocathode (CsI / KBr) ]
       │  photon → photoelectron
       ▼
[ MCP single-stage channels (12 µm, L/D=40, 1000 V) ]
       │  ×2 × 10⁴ gain; visible photons scattered ×5 × 10³
       ▼
[ Phosphor (P-20) on concave fiber-output face,
   with 200 Å Al overcoat blocking back-scattered visible ]
       │  visible photons toward CCD
       ▼
[ Tapered fiber-optic distortion corrector (curved → planar) ]
       │
       ▼
[ CCD P8602 (288 × 385, frame-transfer, vertical-clocked at f·ω/p) ]
       │  electronic despinning, 1-s effective exposure
       ▼
[ Correlated double sampling → 8-bit ADC ]
       │  0.1–0.4 dn/pe
       ▼
[ Spacecraft interface → 5.1 / 30.7 kbps downlink ]
       │
       ▼
[ Kiruna VAX 11/750 archive + interactive control ]
```

**English** — The block diagram makes the multilayer suppression strategy explicit: each block is responsible for a specific physical filter (geometric, spectral, scattering, charge readout). Removing any one layer would let visible light reach the CCD at IBC II levels — which is why V5's heritage emphasizes the chain rather than any single component.

**한국어** — 블록도는 다층 억제 전략을 명시한다: 각 블록이 특정 물리 필터(기하·분광·산란·전하 읽기)를 담당. 어느 한 층을 빼도 가시광이 IBC II 수준으로 CCD에 도달한다 — V5의 유산이 한 부품이 아니라 사슬(chain) 자체임을 강조하는 이유다.

### Part XI: Limitations acknowledged by the authors / 저자가 인정한 한계

**English** — The paper is unusual in its honesty about limits:
- **Dayglow contamination at 1304 Å** prevents Camera-1-based mean-electron-energy retrieval on the dayside. (Resolved by IMAGE SI13's narrower 1356 Å passband 14 years later.)
- **Baffle scatter near noon-midnight orbit phase** introduces operational viewing-time limits. The 45° critical sun angle is a hard constraint, not a soft one.
- **Photocathode QE in the visible (3000–5500 Å) is poorly known** — the paper relies on manufacturer data ($2 \times 10^{-8}$ at 3000 Å, falling 10×/290 Å) and adds the Acton coating as a margin against this uncertainty.
- **Single-stage MCP gain (~2 × 10⁴) is "more than adequate"** but limits dynamic range at the bright end (saturation in IBC III aurora). The 8-step HV control (750–950 V) compensates by tunable gain.
- **8-bit dynamic range** is tight; 1 dn ~ 5–10 pe means the noise floor is partly quantization, not photon shot noise. Modern instruments use 12–14 bits.

**한국어** — 논문은 한계도 솔직히 적었다.
- **1304 Å dayglow 오염**으로 dayside에서 Camera 1 기반 평균 전자 에너지 추정 불가(14년 뒤 IMAGE SI13의 협대역 1356 Å로 해결).
- **noon-midnight 위상에서 배플 산란**으로 관측 시간 제약. 45° 임계 태양각은 절대 제약.
- **광음극의 가시광(3000–5500 Å) QE 부정확** — 제조사 데이터에 의존하고 Acton 코팅으로 안전 여유.
- **단단 MCP 이득 ~2 × 10⁴**은 충분하나 IBC III 밝은 오로라 포화 → 동적 범위 제한. 8단 HV(750–950 V)로 가변 이득 보정.
- **8-bit 동적 범위**는 좁아 1 dn ~ 5–10 pe로 양자화 잡음이 광자 산탄잡음과 비슷. 현대 기기는 12–14 bit.

---

## 7. References / 참고문헌

- Anger, C. D., S. K. Babey, A. L. Broadfoot, R. G. Brown, L. L. Cogger, R. Gattinger, J. W. Haslett, R. A. King, D. J. McEwen, J. S. Murphree, E. H. Richardson, B. R. Sandel, K. Smith, A. Vallance Jones, "An Ultraviolet Auroral Imager for the Viking Spacecraft", *Geophysical Research Letters*, 14(4), 387–390, 1987. DOI: 10.1029/GL014i004p00387
- Anger, C. D., T. Fancott, J. McNally, H. S. Kerr, "ISIS-II Scanning Auroral Photometer", *Applied Optics*, 12, 1753–1766, 1973.
- Breault, R. P., "Problems and Techniques in Stray Radiation Suppression", *SPIE*, 107, 2–23, 1977.
- Burch, C. R., "Reflecting Microscopes", *Proc. Phys. Soc.*, 59, 41–46, 1947.
- Frank, L. A., J. D. Craven, K. L. Ackerson, M. R. English, R. H. Eather, R. L. Carovillano, "Global Auroral Imaging Instrumentation for the Dynamics Explorer Mission", *Space Sci. Inst.*, 5, 369, 1981.
- Fraser, R. D. B., "A 0.8 N.A. Reflecting Microscope for Infrared Absorption Measurements", *JOSA*, 43, 929–930, 1953.
- Hirao, K. and T. Itoh, "Scientific Satellite Kyokko (EXOS-A)", *Solar Terr. Env. Res. in Japan*, 2, 148–152, 1978.
- Meng, C. I. and R. E. Huffman, "Ultraviolet Imaging From Space of the Aurora Under Full Sunlight", *GRL*, 11, 315–318, 1984.
- Rogers, E. H., D. F. Nelson, R. C. Savage, "Auroral Photography From a Satellite", *Science*, 183, 951–952, 1974.
- Rosin, S., "Inverse Cassegrainian Systems", *Applied Optics*, 7, 1483–1497, 1968.
- Wynne, C. G., "Two-Mirror Anastigmats", *JOSA*, 59, 572–578, 1969 (Errata, *JOSA*, 60, 143, 1970).
- Mende, S. B. et al., "Far-Ultraviolet Imaging from the IMAGE Spacecraft", *Space Sci. Rev.*, 91, 287–318, 2000. (Modern descendant / 현대 후계)
- Torr, M. R. et al., "A Far-Ultraviolet Imager for the International Solar-Terrestrial Physics Mission", *Space Sci. Rev.*, 71, 329–383, 1995. (Polar UVI design)
