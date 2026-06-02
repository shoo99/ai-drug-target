# Colocalization 기반 MDD 약물성 유전체 재우선순위화: 최강 TWAS 신호(DRD2)의 강등과 SLC12A5/FURIN/DCC의 부상 — 방향성 인지·확증 게이트 분석(635 GWAS 좌위)

## 초록

**배경:** 주요우울장애(MDD)는 매우 다인자성이나 GWAS 발견을 치료법으로 전환하는 것은 여전히 난제이다. PGC MDD2025를 활용해 유전자 매핑, 뇌 TWAS, 방향성 인지 약물 재배치, 요약기반 colocalization 확증을 통합하여 인과 후보 유전자와 약리 표적을 우선순위화했다.

**방법:** 유럽계 no-23andMe PGC MDD2025 하위집단(환자 412,305명, 대조군 1,588,397명; 유효 N≈1,152,650)을 분석했다. MAGMA 유전자 수준 분석(Bonferroni 2.67e-6), GTEx v8 13개 뇌 조직 S-PrediXcan TWAS, ACAT 상관 강건 다중검정을 적용했다. 인과 확증을 위해 BrainMeta 뇌 cis-eQTL(n=2,865) 대상 전유전체 SMR/HEIDI 및 베이지안 colocalization(coloc.abf)을 수행했다. TWAS 우선 유전자를 DGIdb v5와 교차한 후 예측 위험 방향과 반대 작용하는 약물만 우선순위에 두는 방향성 필터를 적용했다. 유전율은 LDSC로 추정하고 liability scale(K=0.15)로 변환했다.

**결과:** MAGMA가 358개 Bonferroni 유의 유전자를 확인했고, S-PrediXcan은 314개 TWAS 유의(275개 조직간 Bonferroni, ACAT 220개)를 도출했다. *DRD2*가 최강 신호였으며(P=4.43e-28, 측좌핵 Z=−10.99), 차단이 아닌 작용을 시사했다. 방향성 필터로 678건 원시 매칭이 **102개 약물(승인 28개)** 로 축소됐다: 도파민 작용제/부분작용제(브로모크립틴·프라미펙솔·아리피프라졸), DDB1 억제제, RHOA 차단제 — D2 길항제는 명시적으로 제외. 결정적으로 요약기반 확증에서 *DRD2*는 *이 cis-eQTL 좌위에서* colocalize되지 **않았다**(PP4≈0, 서로 다른 GWAS/eQTL 변이) — 즉 cis-조절을 통한 인과성이 *증명되지 않았다*는 것이지 인과성이 *없다*는 것은 아니며, trans-조절·전사후 기여는 배제되지 않았다. BrainMeta probe가 있는 208개 TWAS 유전자 중 30개(14%)만 colocalize. 세 검정 모두 통과는 *SLC12A5*(KCC2)뿐(SMR p=3.6×10⁻¹⁴, HEIDI p=0.20, COLOC PP4=0.996)으로, 인과성은 확증되나 치료 효과의 방향까지 확정하는 것은 아니다. *FURIN*은 높은 PP4=1.00이나 HEIDI 미통과 — 좌위 복잡성으로 SuSiE-coloc fine-mapping 없이는 확증 불가. *DCC*는 시사적(PP4=0.72). Liability h²≈0.084(0.07–0.09 범위).

**결론:** *DRD2*는 이 cis-eQTL 좌위에서 GWAS 신호와 colocalize되지 않으며, *SLC12A5*(KCC2)가 확증된 인과 lead로 부상한다 — 인과성은 확립되나, 치료적 조절 방향(강화 대 억제)은 TWAS Z 부호와 전임상 문헌이 불일치하여 미결정으로 남는다. *FURIN*(불일치)·*DCC*(시사적)은 fine-mapping 전까지 잠정적인 저신뢰 부차 후보로 둔다. 이에 따라 *DRD2* 도파민 약물을 고신뢰 shortlist에서 탐색적 등급으로 강등하되 — 이는 cis-colocalization 근거의 부재를 반영하는 것이지 DRD2가 비인과적이라는 주장은 아니다 — *SLC12A5*로 재정렬한다. SuSiE-coloc fine-mapping, LiftOver 재정합, 조현병 mtCOJO 조건부, 독립 재현, 비유럽계 검증이 우선 후속 과제이다.

## 1. 서론

주요우울장애(MDD)는 전 세계적 장애의 주요 원인으로서 심각한 사회경제적 부담을 초래하고 있습니다. 그럼에도 불구하고, 약물 치료는 여전히 보통 수준의 효능과 장기간의 시행착오적 처방으로 인해 제약을 받고 있습니다. 일선 항우울제에 반응하는 환자는 약 50–60%에 불과하며, 대략 30%는 치료 저항성 우울증을 발현시킵니다. 결정적으로, 1987년 선택적 세로토닌 재흡수 억제제의 도입부터 2019년 에스케타민 승인에 이르기까지 MDD에 대한 새로운 기전의 약물 계열이 승인된 바 없어, 수십 년에 걸친 전환 의학적 간극(translational gap)이 남게 되었습니다. 이러한 약물치료의 병목현상은 견고한 질병 병인에 기반한 새로운 생물학적 표적을 규명할 시급한 필요성을 강조합니다.

유전학적 통찰은 신약 개발의 위험을 줄이는 강력한 경로를 제공합니다. Nelson et al. (2015) 및 King et al. (2019)은 유전적으로 뒷받침된 약물 표적이 인간 유전학적 검증이 부족한 표적에 비해 임상 승인 단계로 진행될 확률이 두 배 높음을 입증했습니다. MDD에 대한 전게놈연관분석(GWAS)은 이 장애의 다유전자적 구조를 점진적으로 규명해 왔습니다. So et al. (2019)의 연구와 같은 초기 노력은 다수의 로커스(loci)를 탐지할 통계적 검증력이 부족했으나, Wray et al. (2018)의 획기적인 PGC 메타분석은 44개 위험 로커스를, 이어 Howard et al. (2019)이 102개 로커스를 확인하여 MDD의 유전학적 기반에 대한 최초의 실질적인 통찰을 제공했습니다. 그럼에도 불구하고, 이전 연구들의 제한된 해상도는 체계적인 약물 표적 선정 및 전사체 우선순위 결정을 제약했습니다.

최근 정신의학유전체컨소시엄(PGC) MDD2025 GWAS(Adams MJ et al., Cell 2025)는 다혈통 688,808명 환자·4,364,225명 대조군에서 635개 좌위의 697개 독립 연관을 보고하며 연구 지형을 변화시켰습니다. 본 연구는 그중 유럽계·no-23andMe 하위집단(412,305례/1,588,397 대조군; 유효 N≈1,152,650)을 분석했습니다. 이 전례 없는 규모는 로커스 발견에서 기전적 우선순위 결정으로 전환하는 데 필요한 통계적 해상도를 제공합니다. 즉, 유전자 수준 연관성과 전사체 예측을 통합하여 약물 표적 경로를 선정하는 것입니다. 그러나 급증하는 연관성의 방대한 양은 인과적 생물학적 기질과 비기능적 대리 표지를 구분하기 위해 엄격하고 다중양상의 필터링을 필요로 합니다.

MDD 유전학적 발견과 치료적 혁신 사이의 간극을 메우고자, 우리는 PGC MDD2025의 유럽인 no-23andMe 하위 집단을 이용하여 포괄적인 유전체 삼각측량(triangulation) 연구를 수행했습니다. 우리는 네 가지 구체적인 목표를 추구했습니다: (1) MAGMA 유전자 수준 분석을 사용하여 GWAS 신호를 보니페로니 유의성을 갖는 유전자에 매핑; (2) 13개 GTEx v8 뇌 조직에 걸쳐 S-PrediXcan 전사체 전체 연관분석(TWAS)을 통해 뇌 특이적인 전사체 기전을 추론; (3) TWAS 유의 및 MAGMA 유의 유전자를 Drug Gene Interaction Database(DGIdb)와 상호 참조하여 유전학적으로 뒷받침된 약물 재배치 후보를 확인; (4) LD Score 회귀 분할 유전율을 통해 MDD의 다유전자적 구조 및 기능적 풍부화를 특정. 이러한 목표들은 MDD에 대한 높은 신뢰도의 치료 표적을 체계적으로 우선순위화합니다.

## 2. 방법


**2.1 PGC MDD2025 요약 통계량**
우리는 정신유전학 컨소시엄(Psychiatric Genomics Consortium, PGC) MDD2025 전장게놈연관분석의 요약 통계량을 분석했다(Adams MJ et al., Cell 2025, DOI 10.1016/j.cell.2024.12.002). 잠재적인 선정 편향(ascertainment bias)을 완화하기 위해, 412,305명의 주요우울장애(MDD) 환자와 1,588,397명의 대조군으로 구성된 유럽 no-23andMe 하위 집단(v3-49-24-11, hg19 좌표)을 활용했다. 유효 표본수는 N_eff≈1,152,650으로, 각 코호트별 유효 표본수 N_eff,i = 4/(1/N_case,i + 1/N_ctrl,i)를 합산한 PGC MDD2025 보고값이다(Adams et al. 2025). 이 코호트별 합산값은 전체 표본 단순 공식값(4/(1/412,305 + 1/1,588,397) ≈ 1,308,800)보다 작은데, 이는 개별 코호트의 환자:대조군 비가 통합 전체보다 덜 균형적이기 때문이다. 표준 품질 관리(QC) 후, 7,363,302개의 SNP가 하위 분석을 위해 유지되었다. 우리는 통합을 위해 SNP, CHR, BP, A1, A2, BETA, SE 및 P 열을 추출했다.

**2.2 MAGMA 유전자 수준 분석**
우리는 MAGMA v1.10(de Leeuw et al. 2015)을 사용하여 유전자 수준 연관성을 수행했다. NCBI 37.3 유전자 정의를 사용하여 SNP를 18,734개의 단백질 코딩 유전자에 매핑했다. 1000 Genomes Project 유럽(EUR) 레퍼런스 패널(n=503)을 통해 연쇄 불평형(LD)을 모델링하여 SNP 단위 평균 유전자 연관성을 계산했다. 본페로니 보정 유의성 임계값 2.67e-6(0.05/18,734)을 적용하여 358개의 본페로니 유의 유전자를 확인했다.

**2.3 DGIdb를 통한 약물 재배치**
우리는 DGIdb v5를 사용하여 약물-유전자 상호작용을 조사했다. 해당 데이터베이스는 32,796개의 범주에 걸쳐 98,240개의 약물-유전자 상호작용을 포함하고 있었다. 우리는 358개의 MAGMA 유의 유전자를 DGIdb와 상호 참조하여, 1,425개의 고유한 약물에 해당하는 2,494개의 유전자-약물 행을 도출했다. 우리는 314개의 TWAS 유의 유전자와 교차하여 678개의 원시 유전자-약물 매칭(방향성 미적용)을 얻은 뒤, **방향성 인지(direction-aware) 필터**를 적용했다: 각 TWAS 유전자의 위험 방향을 S-PrediXcan Z 부호로 추론하고(Z<0 = 발현 감소가 위험 → 작용제/활성제/양성 조절제, Z>0 → 억제제/길항제), 그 방향과 반대로 작용하는 약물만 남겨 **102개의 방향성 일치 약물(승인 28개)** 을 도출했다.

**2.4 S-PrediXcan 뇌 전사체 연관분석(TWAS)**
우리는 13개의 뇌 조직에 걸쳐 GTEx v8 MASHR 예측 모델을 사용하여 S-PrediXcan(Barbeira et al. 2018) 기반 요약 통계량 전사체 연관분석(TWAS)를 수행했다. PGC hg19 rsID를 GTEx v8 varID(GRCh38)와 정렬하기 위해, 85,793개의 고유한 모델 SNP의 합집합으로부터 rsID-to-varID 빌드 크로스워크(crosswalk)를 구성했다. 이 크로스워크는 67,209개의 PGC rsID(모델의 78%)를 성공적으로 매칭했다. 우리는 모든 조직에서 최소 P < 5e-6을 갖는 314개의 고유 유전자를 확인했다. 조직 간 본페로니 보정(P < 2.7e-6)을 적용하여 275개의 조직 간 본페로니 유의 유전자를 도출했다. 뇌 조직 간 강한 상관으로 본페로니가 과보정되므로, 유전자별 조직 p값을 조직 상관에 강건한 ACAT(Cauchy 결합; Liu et al. 2019)로 추가 결합하고 ACAT 전장유의 임계 P<2.8e-6을 적용했다(결과 3.3).

**2.5 확증적 colocalization (SMR/HEIDI 및 COLOC)**
TWAS 연관이 LD 교란이 아닌 인과적 cis-조절 효과를 반영하는지 구분하기 위해, BrainMeta v2 뇌 cis-eQTL(n=2,865; Qi et al. 2022)을 대상으로 두 가지 요약기반 확증분석을 수행했다. (i) SMR+HEIDI(SMR v1.3.1; Zhu et al. 2016)를 전장(16,186 probe) 실행하고 LD는 1000 Genomes EUR 패널을 사용했으며, p_SMR < 2.95e-6(본페로니) 및 p_HEIDI > 0.05인 probe를 인과 후보로 채택했다(HEIDI p < 0.05는 GWAS와 eQTL 신호가 LD상 서로 다른 변이에서 기인함을 의미). (ii) 베이지안 colocalization(coloc.abf; Giambartolomei et al. 2014)을 전체 cis-window에 적용했다. 모든 유전자에 일관 적용하는 3단계 PP4 분류 규칙을 사전 정의했다: **PP4 ≥ 0.8 = 공존(colocalized)**, **0.5 ≤ PP4 < 0.8 = 시사적(suggestive)**, **PP4 < 0.5 = 비공존(not colocalized)**(PP3 > 0.8이면 별개 인과변이를 추가로 시사). 이 규칙을 결과·표 2에 예외 없이 적용하여, 동일한 SMR/HEIDI/PP4 패턴을 가진 유전자는 동일한 라벨을 받는다. SMR+HEIDI와 COLOC는 서로 다른 검정임을 강조한다: SMR+HEIDI는 단일 공유 변이가 두 신호를 설명하는지를, COLOC는 cis-window 전반의 공유 인과변이 사후확률을 평가하므로, 한 검정은 통과하고 다른 검정은 통과하지 못할 수 있다. 따라서 두 결과 집합은 병합하지 않고 별도로 보고한다(3.4절, 그림 3 flow).

**2.6 LDSC 분할 유전율**
우리는 LDSC v3.0.1(Bulik-Sullivan 2015)을 사용하여 분할 SNP-유전율을 추정했다. 우리는 HapMap3 SNP 가중치를 사용하여 PGC MDD2025 요약 통계량을 정제(munge)하여 6,239,820개의 SNP를 유지했다. 관측 척도 유전율(h² = 0.0458 ± 0.0016)을 계산하고 baseline_v1.2 모델을 사용하여 53개의 기능적 범주에 이 유전율을 분할했다. HapMap3 가중치와 병합 후, 1,106,748개의 SNP가 분석되었다. 우리는 게놈 인플레이션(Lambda GC = 1.8485; Mean chi² = 2.1914)과 LD Score 회귀 절편(1.0907 ± 0.0158)을 평가했다. 절편 비율은 0.0761이었으며, 이는 인플레이션의 약 92.4%가 진정한 다유전자 신호를, 약 7.6%가 잔여 집단 층화/혼재를 반영함을 시사한다(잘 통제된 GWAS의 통상 <3%보다 높은, 무시할 수 없는 중간 수준). 관측 척도 h²=0.0458±0.0016은 liability scale h²≈0.084로 변환되며, 이때 MDD 평생 유병률 K=0.15(대규모 역학 추정치 ~15–17%와 일치; Kessler et al. 2005)와 표본 환자비율 P≈0.21(412,305/2,000,702)을 가정했다. K=0.10–0.20 민감도 범위에서 h²≈0.07–0.09를 추가로 보고한다. 가장 강한 농축은 코딩 영역, H3K4me3 프로모터 피크, 보존 서열 및 TSS Hoffman 주석에 국소화되었다.

**2.7 통합**
우리는 치료 표적의 우선순위를 정하기 위해 다중 모달 결과를 통합했다. 수렴하는 유전적 및 전사체적 증거를 가진 변이를 분리하기 위해 358개의 MAGMA 유의 유전자와 275개의 조직 간 본페로니 유의 S-PrediXcan 유전자를 교차시켰다. 우리는 DGIdb v5로 이 교차 유전자 집합을 기능적으로 주석 달고, 각 유전자의 TWAS Z 부호 대비 상호작용 방향에 따라 화합물을 분류했다. **S-PrediXcan은 MR이 아니라 TWAS임을 강조한다**: 예측발현-형질 상관을 측정할 뿐이며, 요약통계만으로는 Steiger 방향성 검정이나 수평 다면발현(horizontal pleiotropy) 배제가 불가능하다. Colocalization(COLOC)과 SMR/HEIDI를 필수 확증 분석으로 명시한다(한계점 참조). DGIdb 주석 품질은 출처 데이터베이스마다 다르다.

**2.8 소프트웨어**
우리는 Python 3.9, R v4.2 및 배시 스크립팅(bash scripting)을 사용하여 KBRI CPU 클러스터에서 모든 분석을 실행했다. 맞춤형 파이프라인 및 재현 가능한 코드는 github.com/shoo99/KBRI/tree/main/gwas-mdd에서 공개적으로 이용 가능하다.

## 3. 결과

### 3.1 전장유전체 신호 분포

우리는 412,305명의 주요우울장애(MDD) 환자와 1,588,397명의 대조군으로 구성된 PGC MDD2025 전장유전체 연관 분석(Adams MJ et al., Cell 2025)의 요약 통계량을 분석했습니다. 품질 관리 후, 하위 분석을 위해 7,363,302개의 SNP(유럽인 no-23andMe 하위 집단, v3-49-24-11, hg19)이 남았습니다. 맨해튼 플롯(그림 1)은 매우 다유전적인 구조를 보여주었습니다(모집단 전체연구 기준 635개 좌위의 697개 독립 연관; Adams MJ et al. 2025). 분위수-분위수(QQ) 플롯(그림 2)은 검정 통계량의 상당한 팽창을 보여주었습니다(λGC = 1.85). 그러나 연관불평형 점수 회귀(LDSC)(LDSC; Bulik-Sullivan 2015)는 이러한 팽창을 분해하여 1.0907 ± 0.0158의 절편과 0.0761의 비율을 산출했습니다. 이는 게놈 팽창의 약 92.4%가 진정한 다유전체 신호를 반영하는 한편, 약 7.6%는 잔여 집단 층화 또는 은밀한 친연성에 기인함을 나타냅니다(무시할 수 없는 중간 수준; 한계점 참조). 관측 척도 SNP-유전력은 0.0458 ± 0.0016이었으며, 이는 liability scale h²≈0.084로 변환됩니다(K=0.15, P≈0.21; K=0.10–0.20에서 0.07–0.09 범위) — 매우 다유전적 구조에 부합하는 보통 수준의 유전력입니다.

### 3.2 MAGMA 유전자 수준 연관 분석

NCBI 37.3 유전자 정의 및 1000 Genomes 유럽인 LD 패널(n = 503)을 사용하여 MAGMA(v1.10)로 유전자 수준 연관 분석을 수행했습니다. 분석된 18,734개 유전자 전체에 본페로니 보정 임계값 2.67e-6(0.05/18,734)을 적용했습니다. 본페로니 유의성을 갖는 358개의 유전자를 확인했습니다. 가장 강한 연관성은 염색체 18의 *DCC* 영역(Entrez ID 1630, P = 1.43e-27, Z = 10.82)에 매핑되었으며, 이는 MDD 병태생리에서 넷린-1/축삭유도 신호의 핵심적인 역할을 강조합니다. 특히 *DCC*는 colocalization 시사 인과 후보로도 지목되어(3.4절), 유전자 수준과 전사체 수준의 수렴적 근거를 가집니다. 연관성이 가장 높은 상위 5개 유전자는(표 1; 그림 3) *DCC* 영역(P = 1.43e-27), *SGCZ*(P = 5.47e-23), *ASTN2*(P = 1.52e-20), *SOX5*(P = 1.10e-19) 및 *SORCS3*(P = 1.05e-18)로 구성되었습니다. 이러한 결과는 유전자 수준에서 신경발달, 시냅스 및 소포 수송 경로를 우선순위로 지정합니다.

**표 1. MDD 연관 상위 10개 유전자 (MAGMA, 본페로니 유의).**

| 순위 | 유전자 | 염색체 | Z | P |
|---|---|---|---|---|
| 1 | DCC | 18 | 10.82 | 1.43e-27 |
| 2 | SGCZ | 8 | 9.80 | 5.47e-23 |
| 3 | ASTN2 | 9 | 9.22 | 1.52e-20 |
| 4 | SOX5 | 12 | 9.00 | 1.10e-19 |
| 5 | SORCS3 | 10 | 8.75 | 1.05e-18 |
| 6 | SDK1 | 7 | 8.69 | 1.78e-18 |
| 7 | TCF4 | 18 | 8.17 | 1.50e-16 |
| 8 | EYS | 6 | 8.06 | 3.77e-16 |
| 9 | PCLO | 7 | 7.92 | 1.21e-15 |
| 10 | NICN1 | 3 | 7.90 | 1.39e-15 |

### 3.3 뇌 조직 전사체 연관분석(TWAS)

GWAS 변이를 잠정적 인과 유전자에 매핑하기 위해, 13개 뇌 조직에 걸쳐 GTEx v8 MASHR 모델을 사용하여 S-PrediXcan(요약 통계량 전사체 연관분석(TWAS))을 적용했습니다. 85,793개의 고유한 모델 SNP 합집합으로부터 rsID-to-varID 빌드 크로스워크를 구성하여, 67,209개의 PGC rsID(모델의 78%)를 성공적으로 매칭했습니다. 임의의 조직에서 최소 P < 5e-6을 갖는 314개의 고유 유전자를 확인했으며, 이 중 275개는 조직 간 본페로니 보정 후에도 유의성을 유지했습니다(P < 2.7e-6). 뇌 조직들이 강하게 상관되어 본페로니가 과보정되므로, 유전자별로 조직 p값을 ACAT(Cauchy 결합, 상관에 강건; Liu et al. 2019)로 추가 결합한 결과 220개가 ACAT 전장유의(P < 2.8e-6)에 도달하여, 유전자 집합이 다중검정 방법에 강건함을 확인했습니다.

가장 강력한 TWAS 신호는 측좌핵(nucleus accumbens)의 *DRD2*에 국소화되었으며(P = 4.43e-28, Z = -10.99), 12개의 추가 뇌 조직 중 6개에서 전장 게놈 수준의 유의성을 나타냈습니다(그림 4, colocalization 상태로 색칠한 TWAS Z-score 포레스트). 발견(TWAS) 수준에서는 중변연로의 도파민성 신경전달을 후보 기작으로 지목하나, 3.4절의 확증분석이 보이듯 *DRD2* 신호는 colocalize되지 않아 이후 강등되므로, 이 발견 단계의 두드러짐을 인과적 검증으로 읽어서는 안 됩니다. 여러 유전자가 강건한 범조직 효과를 나타냈습니다(그림 5 조직 히트맵): *GPX1*(최소 P = 3.02e-19, 피각핵에서 Z = +8.97)과 *AMT*(P = 4.38e-17, Z = -8.40)은 모든 13/13 조직에서 유의했습니다. 추가적인 상위 TWAS 유전자로는 *KLHDC8B*(P = 9.68e-19, Z = +8.84), *SLC12A5*(P = 2.38e-17, Z = +8.47), *BSN*(bassoon; P = 5.06e-17, Z = -8.39), *FURIN*(P = 8.39e-16, Z = -8.05), *NEGR1*(P = 9.20e-16, Z = +8.04; 알려진 MDD 위험 유전자), *RHOA*(P = 1.61e-15, Z = +7.97), 그리고 *LRFN5*(P = 1.65e-14, 예측 모델이 존재한 10개 조직 중 9개에서 유의; LRFN5는 13개 중 3개 조직에서 GTEx 모델이 없었음)가 포함되었습니다. TWAS(발견) 수준에서 이들은 알려진 MDD 신호(*DRD2*, *NEGR1* 등)를 재현하고, *GPX1*·*AMT* 같은 범조직 히트는 각각 산화적 스트레스·미토콘드리아 원탄소 대사를 시사합니다. 다만 TWAS의 두드러짐 자체는 인과 근거가 아니며 — *DRD2*·*NEGR1*·*GPX1*은 모두 colocalization에 실패합니다(3.4절) — 따라서 확증이 아닌 후보 기작으로 다룹니다.

### 3.4 확증적 colocalization 및 SMR

TWAS 연관이 LD 교란이 아닌 인과적 cis-조절 효과를 반영하는지 검증하기 위해, BrainMeta 뇌 cis-eQTL(n=2,865)을 대상으로 SMR/HEIDI와 베이지안 colocalization을 수행했다. 여러 분모(314/275/220/208/33/30) 간 혼선을 피하기 위해, 각 필터 단계의 유전자 집합과 그 관계를 아래 flow로 정리한다:

```
  314  TWAS 유의 (13개 뇌조직 중 최소 P < 5e-6)
        ├─ 275  조직간 본페로니 유의 (P < 2.7e-6)
        └─ 220  ACAT 전장유의 (P < 2.8e-6)
   │
   ▼  (BrainMeta cis-eQTL probe 보유 유전자로 제한)
  208  확증분석 가능 TWAS 유전자
        ├─ 집합 A: 33  SMR 유의(p_SMR<2.95e-6) AND HEIDI 통과(p>0.05)
        └─ 집합 B: 30  공존 (COLOC PP4 > 0.8)
              A ∩ B (둘 다 통과) = 최고신뢰 인과 집합
```

집합 A(SMR+HEIDI)와 B(COLOC)는 서로 다른 검정의 산물이라 일치할 필요가 없으며, 한 유전자는 A만/B만/둘다/어느쪽도 아님 중 하나에 속한다. 우선순위 유전자: **SLC12A5 ∈ A ∩ B**(삼중 확증); **FURIN ∈ B만**(COLOC 통과·HEIDI 실패 → COLOC/HEIDI 불일치); **DCC ∈ A만**(SMR+HEIDI 통과, PP4=0.72 *시사적* 구간, 비공존); **NEGR1 ∈ A만**(SMR+HEIDI 통과, PP4≈0 *비공존*); **GPX1·DRD2 ∈ 어느쪽도 아님**. 전장에서 집합 A는 SMR 검정 16,186개 중 33개, 집합 B는 probe 보유 208개 TWAS 유전자 중 30개(14%)였다. 결정적으로, **헤드라인 TWAS 유전자 _DRD2_는 세 확증검정을 모두 통과하지 못했다**: SMR 비유의(p_SMR = 0.024), HEIDI 탈락(p = 0.003), colocalization 강력 부정(PP4 = 3×10⁻¹², PP3 ≈ 1.0 = GWAS와 뇌-eQTL 신호가 *서로 다른* 인과변이에서 기인). 즉 세 독립적 요약기반 방법이 일관되게 *DRD2* 뇌발현 연관이 MDD에 대한 인과적 cis-조절 효과가 아님을 시사한다.

반면 **_SLC12A5_(KCC2)** 는 세 방법 모두에서 견고하게 확증되어(p_SMR = 3.6×10⁻¹⁴, HEIDI p = 0.20, COLOC PP4 = 0.996) 가장 신뢰할 수 있는 인과 유전자로 지목되었다. **_FURIN_** 은 높은 COLOC PP4(1.00; p_SMR = 1.8×10⁻¹²)를 보였으나 HEIDI를 통과하지 못했다(p = 0.002); 유의한 HEIDI는 단일 공유 인과변이를 기각하므로 **좌위의 복잡성으로 인해 PP4=1.00이 공유 인과 eQTL을 반영한다고 확증할 수 없다** — 높은 PP4는 GWAS 신호와 LD 관계의 2차 eQTL에서 비롯되었을 가능성이 높다. 따라서 *FURIN*은 COLOC/HEIDI 불일치 좌위로 fine-mapping(**SuSiE-coloc, 다중 신호 cis-window 해결의 definitive 방법**) 대상으로 분류하며 **colocalize된 lead로 분류하지 않는다**. **_DCC_** 는 SMR+HEIDI를 통과했고(p_SMR = 7.4×10⁻¹¹, HEIDI p = 0.28) PP4 = 0.72로, 사전 정의된 *시사적* 구간(0.5 ≤ PP4 < 0.8)에 속한다 — 공존은 아니나 비공존 기준선 위이다. **_NEGR1_** 도 SMR+HEIDI를 통과했으나(p_SMR = 1.0×10⁻⁸, HEIDI p = 0.39) PP4 ≈ 8.5×10⁻⁷로 *비공존* 구간(PP4 < 0.5)에 속한다. **즉 DCC("시사적")와 NEGR1("비공존")의 라벨 차이는 동일한 PP4 규칙을 각자의 PP4 값에 적용한 결과이지, 사례별 자의적 판단이 아니다.** *GPX1*도 마찬가지로 공존하지 않았다(PP4 = 0.19; 높은 PP3). DCC가 최상위 MAGMA 유전자 신호(3.2절)라는 사실이 그 미달 colocalization을 **보완하지 않음**을 강조한다: MAGMA 순위와 colocalization은 독립적 증거이며, DCC는 (각각 불완전하나) 수렴적 신호에 근거한 *시사적* 후보로 carry-forward하는 것이지 colocalization 확증 lead가 아니다(표 2). colocalization을 BrainMeta probe가 있는 **TWAS 유의 유전자 208개 전체**로 확장하면 **30개(14%)만 공존(PP4>0.8; 표 S3)** 했으며(*SLC12A5*, *FURIN*, *GPR27*, *TRMT61A*, *SPPL3*, *DIAPH3*, *FADS1*, *SP4*, *CHRNA4*, *VRK2* 등), 대다수는 공존하지 않았다 — 이는 "대부분의 TWAS 연관이 공유 인과변이가 아닌 LD를 반영한다"는 문헌과 일치한다. 이 공존 집합이 약물성 평가로 이어지는 고신뢰 인과 유전자다. 30개 공존 유전자 중 사전 정의 알고리즘으로 핵심 표적을 선정했다: (i) PP4>0.8, (ii) SMR+HEIDI 보강, (iii) DGIdb의 방향성 일치·CNS 관련 약물 상호작용, (iv) MDD/신경정신 생물학적 근거. *SLC12A5*는 4가지 모두 충족, *FURIN*은 (i)(iii)(iv) 충족하나 HEIDI 단서. *DCC*는 **예외**로, PP4=0.72(0.8 미만, 30개 집합 비포함)이나 넷린/DCC 중피질 도파민 배선 근거로 시사적 수준임을 명시해 포함한다. 30개 공존 유전자 전체 순위표(표 S3)를 제공해 선택적 강조를 피한다. 이 결과는 유전자 수준 결론을 상당히 재보정한다: 도파민 *DRD2* 가설은 colocalization으로 뒷받침되지 않으며, 염소수송(*SLC12A5*/KCC2)·전구단백질 전환효소(*FURIN*)·축삭유도(*DCC*) 유전자가 공존 또는 시사 근거를 갖는 인과 후보로 부상한다.

**표 2. 우선순위 유전자의 확증 SMR/HEIDI 및 colocalization (BrainMeta 뇌 cis-eQTL).**

판정은 사전 정의 규칙(2.5절)을 적용: SMR+HEIDI "통과" = p_SMR<2.95e-6 & p_HEIDI>0.05; COLOC 구간 = 공존(PP4≥0.8)/시사적(0.5≤PP4<0.8)/비공존(PP4<0.5). PP4는 유효숫자 3자리로 보고(FURIN PP4 = 0.99999…는 1.00으로 반올림).

| 유전자 | SMR p | HEIDI p | SMR+HEIDI | COLOC PP4 | COLOC PP3 | COLOC 구간 | 판정 |
|---|---|---|---|---|---|---|---|
| SLC12A5 | 3.6e-14 | 0.20 | 통과 | 0.996 | 0.004 | 공존 | 삼중 확증 (집합 A ∩ B) |
| FURIN | 1.8e-12 | 0.002 | 실패(HEIDI) | 1.00 | 8.6e-08 | 공존 | COLOC만(집합 B); COLOC/HEIDI 불일치 |
| DCC | 7.4e-11 | 0.28 | 통과 | 0.72 | 0.28 | 시사적 | SMR+HEIDI + 시사적 COLOC (집합 A) |
| NEGR1 | 1.0e-8 | 0.39 | 통과 | 8.5e-07 | 1.00 | 비공존 | SMR+HEIDI만, 비공존 (집합 A) |
| GPX1 | 3.6e-13 | 0.008 | 실패(HEIDI) | 0.194 | 0.806 | 비공존 | 어느 집합도 아님 |
| DRD2 | 0.024 | 0.003 | 실패(둘다) | 2.9e-12 | 1.00 | 비공존 | 어느 집합도 아님(세 방법 모두 기각) |

### 3.5 DGIdb 통합을 통한 약물 재배치

유전적 연관성을 치료적 가설로 전환하기 위해, 우리는 유의미한 유전자를 Drug Gene Interaction Database(DGIdb v5; 98,240개의 약물-유전자 상호작용, 32,796개의 범주)와 교차 참조했습니다. MAGMA 유의 유전자를 표적으로 하는 1,425개 고유 약물에 해당하는 2,494개의 MAGMA × DGIdb 유전자-약물 행을 생성했고, 314개 TWAS 유의 유전자로 제한해 678개의 원시 매칭(방향성 미적용)을 얻었으며 이는 중간 집합으로만 취급합니다.

결정적으로, 우리는 **방향성 인지 필터**를 적용했습니다(그림 6): 각 TWAS 유전자의 위험 방향을 Z 부호로 판정하고(Z<0 = 발현 감소가 위험 → 활성을 *높이는* 작용제/활성제/양성 조절제, Z>0 → 활성을 *낮추는* 억제제/길항제), 위험 방향과 반대로 작용하는 약물만 남겨 678개를 **102개 방향성 일치 약물(승인 28개)** 로 축소했습니다. *DRD2*의 음의 Z(발현 감소가 위험)와 일관되게, **D2 길항제가 아니라 도파민 작용제/부분작용제(브로모크립틴, 프라미펙솔, 로티고틴, 카베르골린, 아포모르핀, 미나프린, 아리피프라졸, 브렉스피프라졸)** 가 우선순위에 올랐습니다. 그 외 방향성 일치 후보로 DDB1 억제제(탈리도마이드, 레날리도마이드, 포말리도마이드; *DDB1* Z=+4.7)와 RHOA 차단제(퀴니딘; Z=+8.0)가 포함되었습니다. 방향이 맞지 않는 D2 길항 항정신병약과 오프타겟 TKI는 필터에서 제외되었습니다.

도파민성 후보 외에도, 리튬은 5개의 유전자(*ADCY2*, *ASIC2*, *CACNG2*, *CREB1* 및 *DRD2*)로부터 지지를 받았으며, 가바펜틴과 프레가발린은 *CACNA1C* 및 *CACNA1E* 표적을 통해 기전적 중복을 공유했고, 시탈로프람은 *CACNA1C*, *CREB1* 및 *METTL21A*와의 연관성을 나타냈습니다(고찰에서 신중히 해석). 중요하게, 확증분석(3.4절)에 따라 **102개 방향성 약물을 colocalization 여부로 층화**한다. *DRD2*는 공존하지 않고 GWAS와 eQTL이 별개 인과변이에서 기인하므로, *DRD2* 표적 도파민 클러스터(브로모크립틴·프라미펙솔·카베르골린·아리피프라졸 등)는 방향은 일치하나 **저신뢰·탐색적 등급으로 강등하고 우선순위 목록에서 제외**한다(확증분석이 드러낸 LD 교란 위양성을 답습하지 않기 위함). **고신뢰 우선순위 집합은 확증 등급이 상이한 우선순위 유전자(확증 *SLC12A5*; COLOC/HEIDI 불일치 *FURIN*; 시사적 *DCC*)** 기준으로 재정렬하며, 이들의 방향성 일치 약물과 약물성을 일차 후속검토 대상으로 둔다.

### 3.6 LDSC 기능적 분할

품질 관리 후, PGC MDD2025 요약 통계에서 유지된 6,239,820개의 SNP이 baseline_v1.2 모델을 사용하여 53개의 기능적 범주에 걸쳐 기능적 분할에 활용되었습니다. 가장 강한 접힘-풍부화(fold-enrichment)는 코딩 영역, H3K4me3 프로모터 피크, 보존 서열(LindbladToh) 및 전사 시작 부위(TSS) 영역(Hoffman)에서 관찰되었습니다. LD 점수 회귀 절편 분석은 관찰된 팽창의 92.4%가 진정한 다유전자 신호에 기인한 반면, 7.6%는 잔여 층화(stratification)로 인한 것으로 추정했습니다(절편 = 1.0907 ± 0.0158). 관측 척도 h²=0.0458±0.0016은 liability scale h²≈0.084로 변환됩니다(K=0.15, P≈0.21).

## 4. 고찰


본 연구는 주요우울장애(MDD)의 유전적 구조와 치료적 환경을 규명하기 위해 유전자 매핑, 뇌 전사체 임퓨테이션 및 약물 재배치 분석을 통합했습니다. 412,305명의 환자와 1,588,397명의 대조군으로 구성된 PGC MDD2025 코호트(Adams MJ et al., Cell 2025)를 활용하여, 358개의 본페로니 유의성을 갖는 MAGMA 유전자와 275개의 교차 조직 본페로니 유의성을 갖는 S-PrediXcan 유전자를 확인했습니다. 본 연구 결과는 시냅스 부착, 축삭유도, GABA성 억제 조절(KCC2/SLC12A5)의 수렴적 생물학과 산화-환원 요소를 함께 밝힙니다. 결정적으로 확증 레이어는 치료적 지형을 재우선순위화합니다: colocalization 확증 인과 lead(*SLC12A5*)를 지목하는 한편, 최강 TWAS 신호(*DRD2*)를 LD 교란·비공존 연관으로서 고신뢰 shortlist에서 강등합니다 — 이는 도파민 신호를 포함한 TWAS 두드러짐 자체를 검증된 약리학적 기반으로 읽어서는 안 됨을 보여줍니다.

본 연구는 PGC MDD2025 발견 논문(Adams et al. 2025)을 다음 세 가지 면에서 확장합니다: 첫째, 원 연구가 포함하지 않았던 방향성 인지(direction-aware) 약물 재배치 필터링을 체계적으로 수행하여, 678건의 원시 유전자-약물 매칭을 102개의 방향성 일치 후보로 축소했습니다. 둘째, 뇌 cis-eQTL 대상 전유전체 SMR/HEIDI 및 베이지안 colocalization을 적용해 TWAS로 지목된 인과 유전자를 확증하거나 배제했습니다 — 이는 colocalization 확증 없이 TWAS 결과만 보고한 원 분석에 없던 단계입니다. 셋째, 가장 강한 TWAS 신호(*DRD2*)를 colocalization 미통과로 명시적으로 강등하고 *SLC12A5*를 확증 lead로 재배치했는데, 이는 확증 레이어 없이는 도출할 수 없었던 결론입니다. 이러한 추가 작업은 유전자 발견 카탈로그를 인과적으로 필터링된, 치료적으로 실행 가능한 표적 shortlist로 변환합니다.

핵심적인 방법론적 논점은 *DRD2* 신호의 **방향**입니다. S-PrediXcan은 *DRD2*를 최강 유전자로 식별했으며(측좌핵 P=4.43e-28, **Z=−10.99**), 이는 *DRD2* 예측발현이 *낮을수록* MDD 위험이 *높음*을 의미합니다. 따라서 약리학적으로 일관된 치료 방향은 **D2 신호의 차단이 아니라 강화(작용)** 입니다. 초기의 방향 미고려 매칭은 D2 길항 항정신병약을 최상위로 올렸으나, 이는 본 연구에서 명시적으로 정정하는 인공산물입니다: 방향성 인지 필터 적용 후 우선순위에 오른 승인 *DRD2* 약물은 **도파민 작용제/부분작용제(브로모크립틴, 프라미펙솔, 로티고틴, 카베르골린, 아포모르핀, 아리피프라졸, 브렉스피프라졸)** 입니다. 이는 프라미펙솔·아리피프라졸 부가요법이 치료저항성 및 무쾌감성 우울증에서 항우울 효과를 보인다는 최신 근거와 생물학적으로 부합하며, 비필터 목록에 D2 길항제가 등장한 것은 양성 대조가 아니라 방향이 틀린 매칭입니다. 우리는 방향을 고려하지 않은 표적 중첩을 기전적 검증으로 인용하는 흔한 함정을 경계합니다.

그러나 결정적으로, 요약기반 확증분석은 이 도파민 해석을 신중하게 만듭니다. 뇌 cis-eQTL 대상에서 *DRD2*는 SMR(p=0.024)·HEIDI(p=0.003)·colocalization(PP4≈0, PP3≈1.0)을 모두 통과하지 못했으며, 이는 GWAS와 *DRD2*-eQTL 연관이 *서로 다른* 인과변이에서 기인함을 의미합니다. 따라서 방향성 인지 *DRD2* 약물 클러스터는 colocalization으로 검증된 표적이 아니라 TWAS 수준의 가설이며, 도파민 축에 대한 기존 강조는 그만큼 가중치를 낮춥니다. colocalization 확증 핵심 및 잠정/시사 후보—**_SLC12A5_**(KCC2, GABA 억제성 신호를 좌우하는 신경세포 K⁺–Cl⁻ 공수송체; PP4=0.996), **_FURIN_**(PP4=1.00이나 HEIDI 불일치, fine-mapping 필요), **_DCC_**(넷린-1 수용체, 축삭유도; PP4=0.72)—가 더 방어 가능한 인과 후보이며 MDD 생물학적으로도 설득력이 큽니다: KCC2 기능저하는 억제성 신경전달을 교란하고 스트레스 취약성·기분 표현형과 연결되며, FURIN은 BDNF/전구단백질 성숙을, DCC는 중피질 도파민 배선을 매개합니다. 이에 따라 우리는 *DRD2*가 아니라 이들 공존/시사 후보 유전자를 일차 기전 단서로 재배치합니다.

우리는 본 GWAS 좌위에서 *DRD2*를 colocalize된 인과 유전자에서 강등한 것이 더 넓은 의미의 MDD 도파민 가설을 **무효화하는 것은 아님**을 강조합니다. DRD2 단백질은 cis-eQTL colocalization으로 포착되지 않는 trans-조절 기전, 번역후 수식, 수용체 수준 약리를 통해 MDD 병태생리에 기여할 수 있습니다. 게다가 GABA-도파민 접면은 잘 확립되어 있습니다: 복측 피개야(VTA)의 GABA 인터뉴런이 도파민 투사 신경세포를 긴장성으로 억제하며, 이 억제 게이트의 교란—KCC2(SLC12A5) 기능부전이 정확히 야기하는 결과—은 *DRD2* 자체에 cis-조절 효과 없이도 하류 도파민 조절 이상을 일으킬 수 있습니다(Bocklisch et al. 2013). 따라서 SLC12A5 발견은 그 기능부전이 부분적으로 변화된 도파민 톤으로 발현되는 상위 조절 노드를 대표할 수 있습니다. 핵심 구분은 "이 좌위에서의 *DRD2* cis-발현이 인과 매개체"(지지되지 않음)와 "도파민 신호전달이 MDD의 하류 효과기"(본 분석에서 다루지 않음, 여전히 유효 가능)입니다.

*SLC12A5*(KCC2)의 MDD 인과 유전자로서의 생물학적 타당성은 상세 논의가 필요합니다. KCC2는 성숙 신경세포에서 낮은 세포내 Cl⁻ 농도를 유지하는 주된 신경세포 칼륨-염소 공수송체이며, 이는 GABA_A 수용체 매개 억제의 전제조건입니다(Rivera et al. 1999). KCC2 기능이 손상되면 세포내 Cl⁻가 상승하고 GABA 신호가 억제에서 흥분 방향으로 전환되어 미성숙 신경세포 표현형, 이른바 "탈분극성 GABA"(depolarizing GABA) 상태를 재현합니다(Ben-Ari et al. 2012). KCC2 하향조절을 스트레스 및 기분 표현형과 연결하는 전임상 증거는 다양합니다: 만성 스트레스는 설치류 해마에서 KCC2 발현을 감소시키고(Hewitt et al. 2009; Sarkar et al. 2011), KCC2 이형접합 결손 마우스는 증가된 불안- 및 우울 유사 행동을 보입니다(Tornberg et al. 2005). 약리학적으로 KCC2 강화제 CLP-290은 동물 모델에서 염소 항상성을 회복시키고 신경병성 통증을 감소시키는 효능을 보였으며(Gagnon et al. 2013), 이는 KCC2 표적 화합물이 MDD를 포함한 CNS 질환의 치료 후보가 될 수 있음을 시사합니다(조절 방향은 미결정이며, 아래 방향성 단서 참조). 우리의 colocalization 확증 결과(PP4 = 0.996)—유전적으로 예측된 *SLC12A5* 발현이 MDD 위험과 인과적으로 연관됨—는 이러한 전임상 연구의 인간 유전적 근거를 제공하며, KCC2를 임상 검증이 필요한 기전적으로 근거 있는 치료 표적으로 지목합니다.

다만 방향성에 관한 중요한 단서를 명시해야 합니다. 우리의 colocalization 분석은 *SLC12A5*가 이 좌위의 *인과* 유전자임을 확립하나, 그 자체로 치료 개입의 *방향*을 결정하지는 못하며, 여기서 두 증거 갈래가 정반대를 가리킵니다. *SLC12A5*의 S-PrediXcan 연관은 **양의** Z(Z=+8.47)를 가져, 유전적으로 예측된 KCC2 발현이 *높을수록* MDD 위험이 *높음*을 의미합니다; 본 연구의 방향성 인지 논리를 액면 그대로 적용하면 이는 KCC2 *억제*를 지목하게 됩니다. 반면 전임상 문헌은 일관되게 KCC2 *기능 상실*을 병인으로(만성 스트레스가 KCC2를 낮추고, KCC2 저발현 마우스가 불안·우울 유사 표현형) KCC2 *강화*를 치료로 지목합니다. 우리는 이 불일치를 해소하지 못하며, 단일 방향을 단정하기보다 명시적으로 표시합니다. 상호 배타적이지 않은 몇 가지 설명이 가능합니다: (i) 예측 발현 Z 부호는 신경세포 아형과 발생 시기를 평균하는 벌크 뇌 cis-eQTL(GTEx/BrainMeta)에서 도출되는데, KCC2의 발현-기능 관계는 세포종·성숙 의존성이 강해 벌크 조직 방향이 질병 관련 맥락에서 역전될 수 있다; (ii) 스트레스 상태에서 KCC2의 항상성/보상적 상향조절이 위험과 양의 상관을 보이더라도 추가 강화가 해롭다는 의미는 아닐 수 있다; (iii) 방향성 약물 필터(3.5절)는 KCC2에 적용된 적이 없다 — 102개 방향 필터 약물 중 KCC2 표적 화합물이 없으므로 — "강화제" 프레이밍은 전적으로 전임상 문헌에 근거하며 우리 Z 부호와는 독립적(현재로선 불일치)이다. 따라서 우리는 KCC2를 colocalization으로 *확증된 인과 표적*이되 *조절 방향은 추가 규명이 필요한* 것으로 규정합니다; 세포종 특이 eQTL/발현 분석, 발생단계 데이터, 직접 기능 검정(예: iPSC 유래 신경세포에서의 양방향 KCC2 조절)이 강화 대 억제 전략 선택 전에 필요합니다. 이러한 방향성 불확실성은 한계점(5번)에서 논의하는 보다 일반적인 방향 가정의 한계와 맞닿아 있다.

중대한 변환적 질문은 SLC12A5/KCC2가 약물 표적성(druggable)이 있는가입니다. DRD2가 수십 개의 승인 리간드를 보유한 것과 달리 KCC2는 현재 승인된 약물이 없습니다. 그러나 저분자 KCC2 강화제 CLP-290(CLP-257의 prodrug)은 KCC2 활성을 선택적으로 강화하며, 신경병성 통증 및 경직 전임상 모델에서 효능을 입증했습니다(Gagnon et al. 2013; Pan et al. 2024). 최근 고처리량 스크리닝은 독자적인 작용 기전을 가진 추가 저분자 KCC2 potentiator 스캐폴드를 식별했으며(Prael et al. 2022), 본 유전자는 NIMH에서 우선 표적으로 지정된 바 있습니다. *SLC12A5*가 가장 신뢰할 만한 colocalization 확증 MDD 인과 유전자라는 우리의 발견은 KCC2 표적 화합물을 기분 장애 적응증으로 진전시킬 인간 유전적 근거를 제공합니다 — 조절 방향(강화 대 억제)은 위에서 언급한 기능·세포종 특이 연구로 규명되어야 합니다. 다만 예측 발현 변화와의 유전적 연관이 KCC2의 치료 농도에서의 약리학적 조절이 동일한 유전적 표지 효과를 재현함을 보장하는 것은 아니며, iPSC 유래 신경세포에서의 용량-반응 검증이 필수적 다음 단계입니다(향후 과제 참조).

두 부차 후보 *DCC*와 *FURIN*은 기전적 논의가 필요하나, *SLC12A5*보다 의도적으로 낮은 신뢰 등급에 둡니다. *DCC*는 여기서 최강 MAGMA 유전자 수준 신호(P=1.43e-27)이면서 colocalization 시사 수준(PP4=0.72, 0.8 역치 미만)을 함께 보여, 엄격한 colocalization 기준을 충족하진 못해도 유전자 수준과 전사체 수준의 수렴적 근거를 가집니다. 생물학적으로 DCC(Deleted in Colorectal Cancer)는 발생기 축삭 투사를 유도하는 정전(canonical) 넷린-1 수용체로, 중피질변연계에서 도파민 및 전전두엽 회로의 성숙을 조형하며, *DCC* 발현은 기분·스트레스 관련 표현형 및 항우울제 반응과 반복적으로 연관되어 — colocalization으로 확증되진 않았으나 — 생물학적으로 일관된 MDD 후보입니다. 반면 *FURIN*은 높은 COLOC PP4(1.00)를 보이나 HEIDI에 실패하며(p=0.002), 이 불일치는 단일 공유 인과변이가 아니라 다중신호 좌위를 시사하므로 **colocalize된 lead로 취급하지 않습니다**. FURIN은 시냅스 가소성·우울증과 관련된 pro-BDNF 처리를 포함해 다수의 신경영양·신호 전구체를 단백질분해로 성숙시키는 전구단백질 전환효소이며, 교차질환(특히 조현병) 공유 위험 유전자로서 — 생물학적 관심을 유발하는 동시에 — 인과 주장 전에 SuSiE-coloc fine-mapping과 교차질환 조건부 분석이 필요함을 강조합니다. 따라서 *DCC*와 *FURIN*은 가설을 강화하나 잠정적이며, 삼중 확증된 *SLC12A5*에 명시적으로 종속됩니다.

도파민 축을 넘어, 방향성 일치 승인 후보로 **DDB1 표적 약물(탈리도마이드, 레날리도마이드, 포말리도마이드; *DDB1* Z=+4.7, 억제제 일치)** 과 **RHOA 차단제(퀴니딘; Z=+8.0)** 가 포함되었습니다. 우리는 이전에 과대 진술했던 **티로신 키나아제 억제제(TKI) 가설을 의도적으로 철회합니다**: 대부분의 TKI는 P-당단백질 유출 기질로 혈액뇌장벽 투과가 낮고, 그 *DRD2* 활성은 길항성이어서 추론된 치료 방향과 반대이므로 방향성 필터에서 제외됩니다. 마찬가지로 리튬이 5개 유전자(*ADCY2*, *ASIC2*, *CACNG2*, *CREB1*, *DRD2*)의 지지를 받은 것은 특정 기전 신호라기보다 DGIdb상의 비특이적 다약리(promiscuous polypharmacology)를 반영할 가능성이 높고(표준적(canonical) 리튬 표적인 GSK-3β·IMPase는 히트에 없었음), 이에 따라 가중치를 낮춥니다. *FURIN*(P=8.39e-16, Z=-8.05)은 높은 COLOC PP4(1.00)를 갖는 주목할 만한 TWAS 우선 노드이나, HEIDI 불일치(p=0.002)가 좌위 복잡성을 시사하여 fine-mapping(SuSiE-coloc) 없이는 확증할 수 없으므로 검증된 표적이 아닌 **fine-mapping 대상 가설**로 보고합니다.

전사체 연관분석(TWAS) 결과는 시냅스 온전성과 산화-환원 균형을 연결하는 수렴성 MDD 생물학을 규명합니다. 글루타티온 퍼옥시데이즈인 *GPX1*(P=3.02e-19)은 13개 뇌 조직 모두에서 TWAS 수준으로 유의하여 산화 스트레스를 후보 기전으로 지목하나, **GPX1은 colocalize되지 않아(PP4=0.19)** 이 연관은 확정된 병인 기전이 아니라 가설 생성 수준으로 다룹니다. 반면, *BSN*(bassoon; P=5.06e-17)과 *AMT*(P=4.38e-17, 13/13 조직에서 유의미)은 각각 시냅스 전 소포 순환과 미토콘드리아 1탄소 대사를 부각합니다. 세포 부착 분자이자 알려진 MDD 위험 유전자인 *NEGR1*(P=9.20e-16)은 *LRFN5*(P=1.65e-14) 및 최상위 MAGMA 히트 유전자인 *DCC*(P=1.43e-27, 넷린-1 축삭유도 수용체)와 함께 시냅스 구조적 연결성 장애와 축삭유도 이상을 추가로 강조합니다. 종합하면, 이러한 데이터는 손상된 항산화 방어(*GPX1*)가 시냅스·축삭유도 구조(*BSN*, *NEGR1*, *DCC*)에 대한 산화적 손상을 악화시켜 모노아민 신호 변화(*DRD2*)를 복합적으로 가중시키는 (현재 미검증, 사후적) 가설을 시사합니다.

우리의 확증분석이 갖는 더 넓은 의의를 강조할 필요가 있습니다. BrainMeta cis-eQTL probe가 가용한 208개의 TWAS 유의 유전자 중 단 30개(14%)만이 colocalize했습니다(PP4 > 0.8). 이 결과는 다수의 TWAS 연관이 공유 인과변이가 아닌 LD 태깅을 반영한다는 최근 문헌과 일치합니다(Wainberg et al. 2019; Mancuso et al. 2019). 중요한 점은, 출판된 대부분의 TWAS 기반 약물 재배치 연구가 체계적 colocalization을 수행하지 않으며, 따라서 그들의 우선순위 약물 표적은 상당 비율의 LD 교란 위양성을 포함할 수 있다는 것입니다. 본 데이터는 TWAS 유전자-약물 지목 중 약 86%가 colocalization 수준에서 인과 근거를 갖지 못할 수 있음을 시사합니다. 따라서 우리는 colocalization(또는 SuSiE-coloc 같은 확률적 fine-mapping과 같은 동등한 확증 방법)이 TWAS-약물표적 파이프라인에서 선택적 민감도 분석이 아니라 필수 단계로 채택되어야 한다고 주장합니다.

몇 가지 한계점은 주의를 요하며, 어떠한 인과적·치료적 주장 전에 추가로 필요한 확증 분석을 정의합니다. **(1) TWAS ≠ 인과; colocalization 확증은 수행했으나 trans-효과·역인과는 배제하지 못함.** S-PrediXcan은 예측 발현-형질 상관을 검정하며, 유의한 유전자가 LD 연계된 인접 유전자에 의해 추동될 수 있습니다(Wainberg et al. 2019). 우리는 BrainMeta 뇌 cis-eQTL 대상 SMR/HEIDI 및 COLOC를 수행했고(3.4절), 16,186 유전자 중 33개만 SMR+HEIDI를 통과했으며 우선순위 유전자 중 *SLC12A5*·*FURIN*은 공존(PP4>0.8), *DCC*는 시사적(PP4=0.72), *DRD2*·*GPX1*·*NEGR1*은 비공존이었습니다. 나머지 TWAS 유전자는 확증 인과가 아닌 후보입니다. 단서: 전체 cis-window coloc는 locus-conditioned 신호를 놓칠 수 있어 경계 *NEGR1*·*FURIN*은 SuSiE-coloc 정밀화 필요; BrainMeta cis-eQTL 패널(n=2,865)은 도구강도가 중간 수준이라 약한 eQTL 유전자는 SMR 위음성 가능; 요약통계 TWAS는 역인과에 대한 Steiger 필터링 불가, 핵심 유전자에 GSMR/SMR 추가 예정. PGC MDD2025 GWAS와 GTEx v8/BrainMeta eQTL 패널은 비중복 독립표본이므로 TWAS/SMR 연관이 표본중복으로 부풀려지지 않습니다. 또한 SuSiE-coloc는 cis-window 내 다중 인과변이를 명시적으로 모델링하여 *FURIN* 같은 COLOC/HEIDI 불일치 좌위를 해결하는 definitive 방법이며, 이의 적용은 향후과제로 둡니다. **(2) 커버리지·혈통·성별 층화.** rsID 매칭(LiftOver hg19→hg38 미적용)으로 S-PrediXcan 모델 SNP 커버리지가 78%로 통상 >90%에 못 미친다. 지목된 유전자의 *확증*은 견고하나(전체 커버리지의 독립 BrainMeta cis-eQTL 대상 SMR/COLOC, 3.4절), *발견* 집합인 314개 TWAS 유전자 자체가 78% 커버리지 모델의 산물이므로 그 집합에 없는 유전자는 colocalization 검증 자체를 받지 못했다. 따라서 우선순위 결과는 **회수된 발견 집합에 조건부**이다: 314개에 누락된 진짜 인과 유전자(커버리지 불완전에 따른 위음성)는 lead로 부상할 수 없었으며, 회수된 lead가 망라적이라고 주장하지 않는다. 커버리지를 90% 이상으로 끌어올릴 LiftOver(hg19→hg38) 재정합을 이 발견 단계 불완전성 해소를 위한 향후과제로 둔다. 전체 분석은 유럽계만이라 비유럽계 일반화 제한. 추가로 성별 층화 분석은 미수행했으며, MDD의 여성:남성 약 2:1 유병률 비를 고려할 때 합산 분석에서 가려진 성별 특이 유전 구조가 존재할 수 있습니다. **(3) 잔여 층화 및 유전율.** LDSC 절편 비율 7.6%는 무시할 수 없으며(잘 통제된 GWAS는 통상 <3%), 중간 수준의 잔여 층화/배우자 선택 혼재를 시사합니다. 관측척도 h²=0.0458±0.0016은 liability-scale h²≈0.084로 변환(K=0.15, P≈0.21; K=0.10–0.20에서 0.07–0.09 범위) — 중간 정도의 유전율. **(4) 재현 및 교차질환 특이성.** 단일 GWAS 분석이라 상위 유전자는 독립 코호트(FinnGen, MVP) 조회와 within-family/sib-GWAS 민감도 검증 필요. *DRD2*는 조현병 상위 좌위이나 colocalization으로 이미 MDD 인과 매개체에서 배제됐고, 공존 lead(*SLC12A5*, *FURIN*, *DCC*)는 전형 SCZ 정의 좌위가 아님. 그럼에도 mtCOJO/MTAG 조건부 SCZ/BD 분석으로 MDD 특이성 정식 검증은 향후과제(GCTA 가용 환경 필요). **(5) 약물성·방향 가정.** 다중검정은 ACAT(Cauchy combination, 상관 강건) 적용으로 보강하여 220개 전유전체 유의 유전자(Bonferroni 275 대비)를 얻고 유전자 집합이 보정 방법에 안정적임을 확인. DGIdb 상호작용 주석은 출처 편차 있어 Open Targets Genetics·PharmGKB 교차참조와 치료 농도에서의 표적 점유 확인이 임상 추론 전에 필요. 방향성 필터는 예측 발현이 수용체 활성/신호 방향에 단조 대응한다고 가정하나 자가수용체·전후 시냅스 위치·보상적 상향조절(*DRD2*가 대표 사례)로 역전 가능 — 표적-방향 일치는 우선순위 휴리스틱일 뿐 치료 방향의 증명은 아님.

본 연구는 전체 TWAS 유의 유전자에 대한 SMR/HEIDI 및 colocalization과 상관에 강건한 ACAT 다중검정을 이미 완료했습니다(3.4·3.3절). 남은 향후 과제(우선순위순): (i) GWAS의 LiftOver(hg19→hg38) 재정합으로 S-PrediXcan SNP 커버리지를 78%에서 >90%로 끌어올려 잠재적 위음성 유전자를 회복; (ii) 조현병/양극성 GWAS에 대한 mtCOJO/MTAG 조건부 분석으로 MDD 특이성을 정식 검증(GCTA 가용 환경 필요); (iii) 독립 코호트 재현(FinnGen, MVP) 및 within-family 민감도 분석; (iv) *DCC* 영역(최상위 MAGMA 좌위이자 colocalization 시사 후보)의 fine-mapping(SuSiE/FINEMAP); (v) **유전자 집합 농축 분석(FUMA)** 및 **SuSiE-coloc fine-mapping** 으로 COLOC/HEIDI 불일치 또는 경계 사례(예: *FURIN*, *NEGR1*) 정밀화; (vi) **우선순위 유전자(확증 *SLC12A5*/KCC2, COLOC/HEIDI 불일치 *FURIN*, 시사적 *DCC*)** 와 그 방향성 일치 약물의 실험적 검증(iPSC 유래 신경세포).

## References

1. Adams MJ, Howard DM, Coleman JRI, et al. Genome-wide meta-analysis of major depressive disorder in 688,808 cases and 4,364,225 controls identifies 697 risk loci [PGC MDD2025]. Cell. 2025. doi:10.1016/j.cell.2024.12.002
2. Barbeira AN, Pividori M, Zheng J, et al. Exploring the phenotypic consequences of tissue specific gene expression variation inferred from GWAS summary statistics. Nat Commun. 2018;9(1):1825. doi:10.1038/s41467-018-03621-1
3. Ben-Ari Y, Khalilov I, Kahle KT, Cherubini E. The GABA excitatory/inhibitory shift in brain maturation and neurological disorders. Neuroscientist. 2012;18(5):467–486. doi:10.1177/1073858412438697
4. Bocklisch C, Pascoli V, Wong JCY, et al. Cocaine disinhibits dopamine neurons by potentiation of GABA transmission in the ventral tegmental area. Science. 2013;341(6153):1521–1525. doi:10.1126/science.1237059
5. Gagnon M, Bergeron MJ, Lavertu G, et al. Chloride extrusion enhancers as novel therapeutics for neurological diseases. Nat Med. 2013;19(11):1524–1528. doi:10.1038/nm.3356
6. Giambartolomei C, Vukcevic D, Schadt EE, et al. Bayesian test for colocalisation between pairs of genetic association studies using summary statistics. PLoS Genet. 2014;10(5):e1004383. doi:10.1371/journal.pgen.1004383
7. Hewitt SA, Wamsteeker JI, Kurz EU, Bains JS. Altered chloride homeostasis removes synaptic inhibitory constraint of the stress axis. Nat Neurosci. 2009;12(4):438–443. doi:10.1038/nn.2274
8. de Leeuw CA, Mooij JM, Heskes T, Posthuma D. MAGMA: generalized gene-set analysis of GWAS data. PLoS Comput Biol. 2015;11(4):e1004219. doi:10.1371/journal.pcbi.1004219
9. Liu Y, Chen S, Li Z, Morrison AC, Boerwinkle E, Lin X. ACAT: A Fast and Powerful p Value Combination Method for Rare-Variant Analysis in Sequencing Studies. Am J Hum Genet. 2019;104(3):410–421. doi:10.1016/j.ajhg.2019.01.002
10. Mancuso N, Freund MK, Johnson R, et al. Probabilistic fine-mapping of transcriptome-wide association studies. Nat Genet. 2019;51(4):675–682. doi:10.1038/s41588-019-0367-1
11. Pan YZ, Talifu Z, Wang XX, et al. Combined use of CLP290 and bumetanide alleviates neuropathic pain and its mechanism after spinal cord injury in rats. CNS Neurosci Ther. 2024;30(9):e70045. doi:10.1111/cns.70045 [PMC11393004]
12. Prael FJ III, Kim K, Du Y, et al. Discovery of Small Molecule KCC2 Potentiators Which Attenuate In Vitro Seizure-Like Activity in Cultured Neurons. Front Cell Dev Biol. 2022;10:912812. doi:10.3389/fcell.2022.912812 [PMC9263442]
13. Qi T, Wu Y, Fang H, et al. Genetic control of RNA splicing and its distinctive role in complex trait variation [BrainMeta]. Nat Genet. 2022;54(9):1355–1363. doi:10.1038/s41588-022-01154-4
14. Rivera C, Voipio J, Payne JA, et al. The K+/Cl- co-transporter KCC2 renders GABA hyperpolarizing during neuronal maturation. Nature. 1999;397(6716):251–255. doi:10.1038/16697
15. Sarkar J, Wakefield S, MacKenzie G, Moss SJ, Maguire J. Neurosteroidogenesis is required for the physiological response to stress: role of neurosteroid-sensitive GABAA receptors. J Neurosci. 2011;31(50):18198–18210. doi:10.1523/JNEUROSCI.2560-11.2011
16. Tornberg J, Voikar V, Savilahti H, Rauvala H, Airaksinen MS. Behavioural phenotypes of hypomorphic KCC2-deficient mice. Eur J Neurosci. 2005;21(5):1327–1337. doi:10.1111/j.1460-9568.2005.03959.x
17. Wainberg M, Sinnott-Armstrong N, Mancuso N, et al. Opportunities and challenges for transcriptome-wide association studies. Nat Genet. 2019;51(4):592–599. doi:10.1038/s41588-019-0385-z
18. Zhu Z, Zhang F, Hu H, et al. Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets [SMR/HEIDI]. Nat Genet. 2016;48(5):481–487. doi:10.1038/ng.3538
19. Nelson MR, Tipney H, Painter JL, et al. The support of human genetic evidence for approved drug indications. Nat Genet. 2015;47(8):856–860. doi:10.1038/ng.3314
20. King EA, Davis JW, Bhagwat JG. Are drug targets with genetic support twice as likely to be approved? Revised estimates of the impact of genetic support for drug mechanisms on the probability of drug approval. PLoS Genet. 2019;15(12):e1008489. doi:10.1371/journal.pgen.1008489
21. Wray NR, Ripke S, Mattheisen M, et al. Genome-wide association analyses identify 44 risk variants and refine the genetic architecture of major depression. Nat Genet. 2018;50(5):668–681. doi:10.1038/s41588-018-0090-3
22. Howard DM, Adams MJ, Clarke TK, et al. Genome-wide meta-analysis of depression identifies 102 independent variants and highlights the importance of the prefrontal brain regions. Nat Neurosci. 2019;22(3):343–352. doi:10.1038/s41593-018-0326-7
23. Bulik-Sullivan BK, Loh PR, Finucane HK, et al. LD Score regression distinguishes confounding from polygenicity in genome-wide association studies. Nat Genet. 2015;47(3):291–295. doi:10.1038/ng.3211
24. So HC, Chau CKL, Lau A, Wong SY, Zhao K. Translating GWAS findings into therapies for depression and anxiety disorders: gene-set analyses reveal enrichment of psychiatric drug classes and implications for drug repositioning. Psychol Med. 2019;49(16):2692–2708. doi:10.1017/S0033291718003641
25. Kessler RC, Berglund P, Demler O, et al. Lifetime prevalence and age-of-onset distributions of DSM-IV disorders in the National Comorbidity Survey Replication. Arch Gen Psychiatry. 2005;62(6):593–602. doi:10.1001/archpsyc.62.6.593

## 데이터 및 코드의 가용성

코드: https://github.com/shoo99/KBRI/tree/main/gwas-mdd
PGC MDD2025 요약 통계: https://figshare.com/articles/dataset/27061255
GTEx v8 MASHR PredictDB 모델: Zenodo 3518299
DGIdb v5: https://dgidb.org/data/latest
계층적 LDSC 참조 패널: Zenodo 8367200
