# Project Plan — MDD GWAS Drug Repurposing Analysis

> **목적:** PGC MDD2025 GWAS summary statistics (697 loci, 688,808 cases)를 재분석하여 우울증 치료를 위한 약물 재창출(drug repurposing) 후보를 식별하는 논문 작성
>
> **인프라:** KBRI CPU 클러스터 (GPU 불필요)
> **예상 소요:** ~1주
> **이전 프로젝트와 무관 — 독립 프로젝트**

---

## 1. 논문 제목 (가안)

**"Genetic Drug Target Discovery for Major Depressive Disorder: Integrating 697 GWAS Loci with Druggability Assessment and Mendelian Randomization"**

---

## 2. 데이터 소스

### 2.1 Primary — PGC MDD2025 Summary Statistics
- **논문:** Als et al. (2025). "Trans-ancestry genome-wide study of depression identifies 697 associations implicating cell types and pharmacotherapies." *Cell*.
- **다운로드:** https://figshare.com/articles/dataset/GWAS_summary_statistics_for_major_depression_PGC_MDD2025_/27061255
- **규모:** 688,808 MDD cases, 4,364,225 controls, 29개국
- **포맷:** TSV.gz (multi-ancestry + European-only subsets)
- **제한:** 자유 다운로드 가능. 단, 23andMe 데이터 제외 버전 사용

```bash
# 다운로드 명령
wget https://figshare.com/ndownloader/articles/27061255/versions/1 -O pgc_mdd2025.zip
unzip pgc_mdd2025.zip -d data/gwas/mdd2025/
```

### 2.2 Secondary — Cross-disorder (선택)
- **OpenGWAS API:** https://gwas.mrcieu.ac.uk/
- Anxiety, PTSD, Bipolar, Schizophrenia summary stats (LDSC cross-disorder용)

### 2.3 Reference Data
- **LD reference panel:** 1000 Genomes Phase 3 EUR
  - https://ctg.cncr.nl/software/magma (MAGMA 페이지에서 다운로드)
- **Gene locations:** NCBI37.3 gene locations (MAGMA 포맷)
- **DrugBank:** https://go.drugbank.com/releases/latest (약물-표적 매핑)
- **DGIdb:** https://www.dgidb.org/downloads (약물-유전자 상호작용)
- **eQTL data:** GTEx v8 brain tissues (S-PrediXcan용)
  - https://predictdb.org/

---

## 3. 분석 파이프라인

### Phase 1: 데이터 준비 (1시간)

```bash
# 1. GWAS summary stats 다운로드 + 검증
wget [figshare URL]
zcat mdd2025_eur.tsv.gz | head -5  # 컬럼 확인
wc -l mdd2025_eur.tsv.gz           # SNP 수 확인

# 2. LD reference panel
wget https://ctg.cncr.nl/software/magma/ref/g1000_eur.zip
unzip g1000_eur.zip -d data/reference/

# 3. Gene annotation
wget https://ctg.cncr.nl/software/magma/aux/NCBI37.3.gene.loc
```

### Phase 2: Gene-Level Association — MAGMA (2시간)

**목적:** SNP-level p-values를 gene-level p-values로 변환

```bash
# Step 1: Annotation (SNP → Gene 매핑)
magma --annotate \
  --snp-loc data/gwas/mdd2025/mdd2025_eur.tsv.gz \
  --gene-loc NCBI37.3.gene.loc \
  --out results/mdd_annotated

# Step 2: Gene Analysis
magma --bfile data/reference/g1000_eur \
  --gene-annot results/mdd_annotated.genes.annot \
  --pval data/gwas/mdd2025/mdd2025_eur.tsv.gz \
  ncol=N \
  --out results/mdd_gene_analysis

# Step 3: Gene-Set Analysis (pathway enrichment)
magma --gene-results results/mdd_gene_analysis.genes.raw \
  --set-annot data/gene_sets/msigdb_c2_c5.txt \
  --out results/mdd_geneset
```

**출력:** `mdd_gene_analysis.genes.out` — 각 유전자의 p-value, z-score

### Phase 3: Drug Target Mapping (1시간)

```python
# drug_target_mapping.py

import pandas as pd

# 1. MAGMA 유의미 유전자 로드 (Bonferroni p < 0.05/20000)
genes = pd.read_csv("results/mdd_gene_analysis.genes.out", sep="\s+")
sig_genes = genes[genes["P"] < 0.05/20000]
print(f"Significant genes: {len(sig_genes)}")

# 2. DrugBank 약물-표적 매핑
drugbank = pd.read_csv("data/drugbank/drug_targets.csv")
# 또는 DGIdb
dgidb = pd.read_csv("data/dgidb/interactions.tsv", sep="\t")

# 3. 매칭: GWAS sig genes × DrugBank targets
hits = sig_genes.merge(drugbank, left_on="GENE", right_on="gene_name")

# 4. 분류
# - Approved drugs (repurposing 후보)
# - Clinical trial drugs (validation 후보)  
# - Experimental drugs (discovery 후보)

# 5. Output
hits.to_csv("results/drug_repurposing_candidates.csv")
```

### Phase 4: Mendelian Randomization (4시간)

**목적:** 유전적 도구 변수를 이용한 인과 추론 — "이 약물 표적을 조절하면 MDD 위험이 변하는가?"

```r
# mendelian_randomization.R

library(TwoSampleMR)

# 1. Exposure: 약물 표적 유전자의 cis-eQTL (GTEx brain)
# 2. Outcome: MDD GWAS summary stats

# 각 drug target gene에 대해:
for (gene in top_drug_targets) {
  # cis-eQTL을 instrument로 사용
  exposure <- extract_instruments(
    outcomes = gene_eqtl_id,  # OpenGWAS에서
    p1 = 5e-8
  )
  
  # MDD outcome
  outcome <- extract_outcome_data(
    snps = exposure$SNP,
    outcomes = "ieu-b-102"  # MDD GWAS ID in OpenGWAS
  )
  
  # Harmonize + MR
  dat <- harmonise_data(exposure, outcome)
  results <- mr(dat, method_list = c("mr_ivw", "mr_egger", "mr_weighted_median"))
  
  # Sensitivity: MR-Egger intercept, MR-PRESSO
}
```

**또는 Python 대안:**
```python
# pip install mr-miner  또는 수동 구현
# S-PrediXcan (gene-level MR의 대안)

# S-PrediXcan: GWAS summary stats → predicted gene expression → MDD association
python3 SPrediXcan.py \
  --model_db_path data/gtex/Brain_Hippocampus.db \
  --covariance data/gtex/Brain_Hippocampus.txt.gz \
  --gwas_folder data/gwas/mdd2025/ \
  --snp_column SNP \
  --effect_allele_column A1 \
  --non_effect_allele_column A2 \
  --beta_column BETA \
  --pvalue_column P \
  --output_file results/spredixcan_hippocampus.csv
```

### Phase 5: Cell-Type Enrichment — LDSC-SEG (4시간)

```bash
# LD Score Regression - Specifically Expressed Genes
# Brain single-cell RNA-seq reference

# 1. Munge summary stats
python3 munge_sumstats.py \
  --sumstats data/gwas/mdd2025/mdd2025_eur.tsv.gz \
  --out results/mdd_munged \
  --merge-alleles data/reference/w_hm3.snplist

# 2. Run LDSC-SEG with brain cell types
python3 ldsc.py --h2-cts results/mdd_munged.sumstats.gz \
  --ref-ld-chr data/reference/baseline_v1.2/baseline. \
  --w-ld-chr data/reference/weights_hm3_no_hla/weights. \
  --ref-ld-chr-cts data/cell_types/brain_celltypes.ldcts \
  --out results/mdd_celltype
```

### Phase 6: Figure Generation (3시간)

```
Fig 1: Manhattan plot (697 loci highlighted)
Fig 2: MAGMA gene-level results → top genes bar chart
Fig 3: Drug repurposing Venn diagram (approved × clinical × GWAS hits)
Fig 4: Mendelian Randomization forest plot (top drug targets)
Fig 5: Cell-type enrichment dot plot (brain regions × cell types)
Fig 6: Drug-gene network visualization
```

### Phase 7: Manuscript Writing (2일)

---

## 4. 필요 소프트웨어

```bash
# 설치 목록
# 1. MAGMA (gene analysis)
wget https://ctg.cncr.nl/software/magma/magma_v1.10_static.zip
unzip magma_v1.10_static.zip

# 2. LDSC (LD Score Regression)
git clone https://github.com/bulik/ldsc.git
cd ldsc && conda env create --file environment.yml

# 3. S-PrediXcan (TWAS)
git clone https://github.com/hakyimlab/MetaXcan.git

# 4. R packages
Rscript -e 'install.packages(c("TwoSampleMR", "MendelianRandomization", "ggplot2"))'

# 5. Python packages
pip install pandas numpy scipy matplotlib seaborn statsmodels
```

---

## 5. 예상 결과 및 Novelty

### 예상 결과
1. **~50-100개 druggable gene** (697 loci에서 MAGMA → DrugBank 매칭)
2. **~10-20개 MR-validated drug targets** (인과 관계 확인)
3. **Top repurposing 후보:** 기존 승인 약물 중 MDD에 유전적 근거 있는 것
4. **Cell-type specificity:** 어떤 뇌세포가 MDD와 가장 연관되는지

### Novelty
- PGC MDD2025 (697 loci)는 **2025년 1월 출판** → 아직 drug repurposing 분석 논문 없음
- MR로 인과 검증까지 한 MDD drug target 논문은 소수
- 697 loci 전체에 대한 systematic druggability + MR = **최초**

### 기존 유사 연구와 차별점
| 기존 연구 | 우리 | 차이 |
|----------|------|------|
| Gaspar 2019 (44 loci) | **697 loci** | 16배 규모 |
| So 2017 (drug target) | **MR 인과 검증 포함** | 방법론 강화 |
| PGC 2025 원 논문 | **Drug target 초점** | 분석 확장 |

---

## 6. 타깃 저널

| 순위 | 저널 | IF | 적합도 |
|------|------|-----|--------|
| 1 | **Translational Psychiatry** | ~7 | ⭐⭐⭐⭐⭐ |
| 2 | **Molecular Psychiatry** | ~11 | ⭐⭐⭐⭐ |
| 3 | **Biological Psychiatry** | ~10 | ⭐⭐⭐⭐ |
| 4 | **Neuropsychopharmacology** | ~7 | ⭐⭐⭐⭐ |
| 5 | **Frontiers in Psychiatry** | ~4 | ⭐⭐⭐ (safe) |

---

## 7. 작업 일정

| Day | 작업 | 산출물 |
|-----|------|--------|
| 1 | 데이터 다운로드 + MAGMA 설치 + gene analysis | `mdd_gene_analysis.genes.out` |
| 2 | Drug target mapping + DGIdb/DrugBank 매칭 | `drug_repurposing_candidates.csv` |
| 3 | Mendelian Randomization (S-PrediXcan 또는 TwoSampleMR) | `mr_results.csv` |
| 4 | LDSC-SEG cell-type enrichment | `mdd_celltype.results` |
| 5 | Figures (6개) | `figures/*.png` |
| 6-7 | Manuscript 작성 | `manuscript.tex` |

---

## 8. 주의사항

1. **23andMe 데이터 제외 버전 사용** — PGC는 23andMe 포함/제외 버전 모두 제공. 제외 버전 사용 (라이선스 문제 방지)
2. **Summary statistics만 사용** — 개인 수준 데이터 아님. IRB 불필요.
3. **MR 가정 검증 필수** — Instrument strength (F-statistic), horizontal pleiotropy (MR-Egger intercept), MR-PRESSO outlier test
4. **Multiple testing correction** — Bonferroni for gene-level, FDR for drug targets
5. **기존 약물 safety 확인** — repurposing 후보의 기존 부작용 정보 포함

---

## 9. 참고 논문

1. Als TD et al. (2025). Trans-ancestry GWAS of depression. *Cell*. DOI: 10.1016/j.cell.2024.12.025
2. Gaspar HA et al. (2019). Using genetic drug-target networks to develop new drug hypotheses for MDD. *Transl Psychiatry*.
3. So HC et al. (2017). Exploring shared genetic bases and causal relationships of schizophrenia and bipolar disorder with 28 cardiovascular and metabolic traits. *Psychol Med*.
4. de Leeuw CA et al. (2015). MAGMA: Generalized gene-set analysis of GWAS data. *PLoS Comput Biol*.
5. Hemani G et al. (2018). The MR-Base platform supports systematic causal inference across the human phenome. *eLife*.
6. Barbeira AN et al. (2018). Exploring the phenotypic consequences of tissue specific gene expression variation inferred from GWAS summary statistics. *Nature Commun*.

---

*Generated for KBRI cluster execution. All analyses use summary statistics only (no individual-level data). CPU-only compatible.*
