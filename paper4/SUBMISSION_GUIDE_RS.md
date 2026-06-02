# Research Square 제출 가이드 — Paper 4 (MDD GWAS Drug Repurposing) v19

> 제출 파일: `paper4_submission_v19_en.pdf` (영문, 제출 본문 — 그림·표 임베드 완결본)
> 그림(개별 첨부용): `fig1_manhattan` · `fig2_qq` · `fig3_magma_top` · `fig4_twas_forest` · `fig5_tissue_dot` · `fig6_directional_drugs` · `figS1_drug_network`
> 포털: https://www.researchsquare.com  (로그인 후 "Post a preprint" / New submission)

---

## 1. Title (그대로 복사)

Colocalization Reprioritizes the Major Depressive Disorder Druggable Genome: Demotion of the Top TWAS Signal (DRD2) and Nomination of SLC12A5/FURIN/DCC by Direction-Aware, Confirmation-Gated Analysis of 635 GWAS Loci

---

## 2. Abstract (plain text, 그대로 복사 — 마크다운 기호 제거됨)

Background: Major depressive disorder (MDD) is highly polygenic, yet translating genome-wide association study (GWAS) findings into therapeutics remains challenging. Using PGC MDD2025, we integrated gene-level mapping, brain TWAS, direction-aware drug repurposing, and summary-based colocalization to prioritize causal genes and pharmacological targets.

Methods: We analyzed the European-ancestry no-23andMe PGC MDD2025 subset (412,305 cases, 1,588,397 controls; effective N≈1,152,650). MAGMA gene-level association (Bonferroni 2.67e-6), S-PrediXcan TWAS across 13 GTEx v8 brain tissues, and ACAT correlation-robust multiple testing were applied. To distinguish causal from LD-confounded TWAS hits we performed genome-wide SMR/HEIDI and Bayesian colocalization (coloc.abf) against BrainMeta brain cis-eQTL (n=2,865). TWAS-prioritized genes were intersected with DGIdb v5 and filtered so that only drugs opposing the predicted risk direction were retained. Heritability was estimated via LDSC (liability scale, K=0.15).

Results: MAGMA identified 358 Bonferroni-significant genes; S-PrediXcan yielded 314 TWAS-significant (275 cross-tissue Bonferroni, 220 by ACAT). DRD2 was the strongest signal (P=4.43e-28, Z=−10.99 in nucleus accumbens), implying agonism rather than blockade. Direction-aware filtering reduced 678 raw matches to 102 drugs (28 approved): dopamine agonists/partial agonists (bromocriptine, pramipexole, aripiprazole), DDB1 inhibitors, and RHOA blockers, with D2 antagonists explicitly excluded. Critically, summary-based confirmation did not support DRD2 as a colocalized causal gene at this cis-eQTL locus (COLOC PP4≈3×10⁻¹² ≈ 0, distinct GWAS/eQTL variants) — i.e., not proven causal via cis-regulation rather than proven non-causal, as trans-regulatory or post-transcriptional contributions are not excluded. Of 208 TWAS genes with BrainMeta probes, only 30 (14%) colocalized. SLC12A5 (KCC2) alone passed all three tests (SMR p=3.6×10⁻¹⁴, HEIDI p=0.20, COLOC PP4=0.996), confirming causality though not the direction of therapeutic effect. FURIN showed high COLOC PP4=1.00 but failed HEIDI — locus complexity precludes confirmation without SuSiE-coloc fine-mapping. DCC was suggestive (PP4=0.72). Liability-scale h²≈0.084 (0.07–0.09 range).

Conclusion: DRD2 does not colocalize with the GWAS signal at this cis-eQTL locus; SLC12A5 (KCC2) emerges as the confirmed causal lead — establishing causality, though the direction of therapeutic modulation (enhancement vs inhibition) remains unresolved, as the TWAS Z-sign and the preclinical literature are discordant — with FURIN (COLOC/HEIDI-discordant) and DCC (suggestive) as secondary, lower-confidence candidates that remain provisional pending fine-mapping. We accordingly demote DRD2-based dopaminergic drugs from the high-confidence shortlist to an exploratory tier — reflecting the absence of cis-colocalized support rather than a claim that DRD2 is non-causal — and re-anchor on SLC12A5. SuSiE-coloc fine-mapping, LiftOver re-harmonization, mtCOJO conditioning on schizophrenia, independent replication, and non-European validation are priority next steps.

---

## 3. Keywords (5–6개 제안)

major depressive disorder; colocalization; transcriptome-wide association study; drug repurposing; SLC12A5 (KCC2); Mendelian randomization (SMR)

---

## 4. Subject Area / Category (RS 선택지)

- Primary: Genetics & Genomics  (또는 Psychiatry)
- Secondary: Pharmacology / Drug Discovery; Neuroscience

---

## 5. Authors & Affiliations  ⚠️ 사용자 입력 필요

RS 계정 정보에서 자동 채워지는 경우가 많음. 기존 Paper 1·2·ARIA와 동일 저자/소속으로 통일 권장.
(제가 저자명·소속 정보를 보유하고 있지 않으니, 기존 프리프린트와 동일하게 입력해 주세요.)

---

## 6. Data & Code Availability (그대로 복사 — 본문에도 포함됨)

- Code: https://github.com/shoo99/KBRI/tree/main/gwas-mdd
- PGC MDD2025 summary statistics: https://figshare.com/articles/dataset/27061255
- GTEx v8 MASHR PredictDB models: Zenodo 3518299
- DGIdb v5: https://dgidb.org/data/latest
- Stratified LDSC reference panel: Zenodo 8367200

---

## 7. 업로드 파일 목록

1. **본문**: `paper4_submission_v19_en.pdf`  (그림·표 임베드되어 있음 → 본문만으로도 완결; 3차 외부리뷰까지 반영한 최신본)
2. (선택) 개별 고해상 그림 — RS가 별도 figure 업로드를 요구할 경우:
   - Figure 1: `figures/fig1_manhattan.png`
   - Figure 2: `figures/fig2_qq.png`
   - Figure 3: `figures/fig3_magma_top.png`
   - Figure 4: `figures/fig4_twas_forest.png`  (coloc 반영 신버전)
   - Figure 5: `figures/fig5_tissue_dot.png`
   - Figure 6: `figures/fig6_directional_drugs.png`  (2패널 신버전)
   - Supplementary Figure S1: `figures/figS1_drug_network.png`  (제목 정정 신버전)

---

## 8. Competing interests / Funding

- Competing interests: None declared (해당 없으면)
- Funding: (해당 시 기재, 없으면 "No external funding")

---

## 9. 제출 순서 (포털)

1. researchsquare.com 로그인 → New preprint submission
2. Title 붙여넣기 (위 1)
3. Abstract 붙여넣기 (위 2, plain text)
4. Authors/affiliations 확인·입력 (위 5)
5. Keywords (위 3), Subject area (위 4)
6. 본문 PDF 업로드 (위 7-1); figure 별도 요구 시 위 7-2
7. Data availability / competing interests / funding (위 6, 8)
8. Review → Submit → DOI 발급 (보통 수일 내 게시)
