# Paper Directory Rename Log / 논문 디렉토리 이름 변경 기록

**Date / 날짜**: 2026-04-15
**Reason / 사유**: Naming convention unified to first-author-only (`{num}_{first_author}_{year}`) / 명명 규칙을 1저자 성만 사용하는 방식으로 통일

---

## Summary / 요약

- **Total renamed papers / 이름 변경 논문 수**: 16
- **Total renamed files / 이름 변경 파일 수**: 64 (4 files per paper: `_paper.pdf`, `_briefing.md`, `_notes.md`, `_implementation.ipynb`)
- **All renamed papers have status `[x]` (completed) / 모든 변경 논문은 읽기 완료 상태**

---

## Artificial Intelligence (5)

| # | Old Directory | New Directory |
|---|---------------|---------------|
| 1 | `01_mcculloch_pitts_1943/` | `01_mcculloch_1943/` |
| 6 | `06_rumelhart_hinton_williams_1986/` | `06_rumelhart_1986/` |
| 8 | `08_cortes_vapnik_1995/` | `08_cortes_1995/` |
| 9 | `09_hochreiter_schmidhuber_1997/` | `09_hochreiter_1997/` |
| 15 | `15_kingma_welling_2013/` | `15_kingma_2013/` |

**File renames (per directory):**
- `01_mcculloch_pitts_1943_*` → `01_mcculloch_1943_*`
- `06_rumelhart_hinton_williams_1986_*` → `06_rumelhart_1986_*`
- `08_cortes_vapnik_1995_*` → `08_cortes_1995_*`
- `09_hochreiter_schmidhuber_1997_*` → `09_hochreiter_1997_*`
- `15_kingma_welling_2013_*` → `15_kingma_2013_*`

---

## Living Reviews in Solar Physics (2)

| # | Old Directory | New Directory |
|---|---------------|---------------|
| 3 | `03_nakariakov_verwichte_2005/` | `03_nakariakov_2005/` |
| 5 | `05_gizon_birch_2005/` | `05_gizon_2005/` |

**File renames:**
- `03_nakariakov_verwichte_2005_*` → `03_nakariakov_2005_*`
- `05_gizon_birch_2005_*` → `05_gizon_2005_*`

---

## Solar Observation (1)

| # | Old Directory | New Directory |
|---|---------------|---------------|
| 4 | `04_goode_cao_2012/` | `04_goode_2012/` |

**File renames:**
- `04_goode_cao_2012_*` → `04_goode_2012_*`

---

## Solar Physics (3)

| # | Old Directory | New Directory |
|---|---------------|---------------|
| 3 | `03_kirchhoff_bunsen_1860/` | `03_kirchhoff_1860/` |
| 7 | `07_hale_nicholson_1925/` | `07_hale_1925/` |
| 13 | `13_leighton_noyes_simon_1962/` | `13_leighton_1962/` |

**File renames:**
- `03_kirchhoff_bunsen_1860_*` → `03_kirchhoff_1860_*`
- `07_hale_nicholson_1925_*` → `07_hale_1925_*`
- `13_leighton_noyes_simon_1962_*` → `13_leighton_1962_*`

---

## Space Weather (5)

| # | Old Directory | New Directory |
|---|---------------|---------------|
| 2 | `02_chapman_ferraro_1931/` | `02_chapman_1931/` |
| 7 | `07_axford_hines_1961/` | `07_axford_1961/` |
| 10 | `10_mcpherron_russell_aubry_1973/` | `10_mcpherron_1973/` |
| 11 | `11_burton_mcpherron_russell_1975/` | `11_burton_1975/` |
| 13 | `13_richmond_kamide_1988/` | `13_richmond_1988/` |

**File renames:**
- `02_chapman_ferraro_1931_*` → `02_chapman_1931_*`
- `07_axford_hines_1961_*` → `07_axford_1961_*`
- `10_mcpherron_russell_aubry_1973_*` → `10_mcpherron_1973_*`
- `11_burton_mcpherron_russell_1975_*` → `11_burton_1975_*`
- `13_richmond_kamide_1988_*` → `13_richmond_1988_*`

---

## Naming Convention / 명명 규칙

**New rule (from 2026-04-15):**
```
{number:02d}_{first_author_surname}_{year}
```

- First author only (never multi-author) / 1저자만 사용
- Suffixes removed: Jr., Sr., III, II, IV / 접미사 제거
- Prefixes preserved: van, von, de, di, le, la / 접두사 보존
- Single source of truth: `scripts/paper_dir.py` / 단일 명명 함수
