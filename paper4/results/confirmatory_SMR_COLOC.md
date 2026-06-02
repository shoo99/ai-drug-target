# 확증분석 종합: SMR+HEIDI & COLOC (BrainMeta 뇌 eQTL × PGC MDD2025)

| 유전자 | SMR p | HEIDI p | COLOC PP4 | COLOC PP3 | 종합 판정 |
|---|---|---|---|---|---|
| **SLC12A5**(KCC2) | 3.6e-14 ✓ | 0.20 ✓ | **0.996** ✓ | 0.004 | ✅ **삼중 확증 인과 (최강)** |
| **FURIN** | 1.8e-12 ✓ | 0.002 ✗ | **1.00** ✓ | ~0 | ✅ COLOC 강력 지지(HEIDI와 불일치하나 coloc 우세) |
| **DCC** | 7.4e-11 ✓ | 0.28 ✓ | 0.72 ◐ | 0.28 | ◐ 인과 시사(suggestive) |
| NEGR1 | 1.0e-8 ✓ | 0.39 ✓ | 8e-7 ✗ | 1.00 | ⚠️ 불일치(SMR/HEIDI 통과나 COLOC는 별개변이) |
| GPX1 | 3.6e-13 ✓ | 0.008 ✗ | 0.19 ✗ | 0.81 | ❌ 비공존(별개 변이 가능) |
| 🔴 **DRD2** | 0.024 ✗ | 0.003 ✗ | **3e-12** ✗ | **1.00** | ❌ **세 방법 모두 인과성 부정** |

## 핵심
- 🔴 **DRD2(논문 헤드라인)**: SMR 미통과 + HEIDI 탈락 + **COLOC PP4≈0, PP3=1.0(별개 인과변이)**. **세 독립 검증 + DeepSeek 리뷰가 모두 "DRD2 뇌발현 매개 인과 아님"으로 수렴** → 도파민/DRD2 서사는 대폭 신중화 필요.
- ✅ **SLC12A5(KCC2, 신경세포 염소수송체)**: SMR+HEIDI+COLOC(0.996) 삼중 확증 → 가장 견고한 인과 후보. **새 헤드라인 후보**.
- ✅ **FURIN**(COLOC 1.0), ◐ **DCC**(0.72): 인과 지지.
- 방법 주의: cis-window 전체(9~18K SNP)로 coloc 수행 → 다중신호 영역은 PP3 과대 가능. tighter window/conditioning으로 정밀화 가능하나, DRD2 비공존·SLC12A5/FURIN 공존은 견고.

## 전체 TWAS 유전자 COLOC (2026-05-22, DeepSeek P1 대응)
- BrainMeta probe 있는 **208개 TWAS 유의 유전자** 전수 coloc.abf.
- **PP4>0.8 (colocalized) = 30개(14%)**, 0.5–0.8 suggestive = 16개. → **대다수 TWAS 신호는 비공존(LD 교란)**, TWAS 문헌과 일치.
- colocalized 30개: FURIN, TRMT61A, GPR27, SLC12A5, SPPL3, KLF11, HACD2, INAFM1, AREL1, DIAPH3, PPP6C, PDS5A, KDELR2, ZDHHC5, PPP1R18, MYRF, CHRNA4, GIGYF2, FADS1, SP4, TMEM165, RERE, OPN3, B4GALNT4, VRK2, DDX27, ZNHIT1, FRAT1, FLJ20021, CTBP1
- 결과: results/coloc/coloc_all_summary.csv. DRD2는 비공존(확인). SLC12A5/FURIN는 최상위 유지.
