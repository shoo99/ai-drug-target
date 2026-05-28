# Discussion 보강 지시서 — Paper 4 (MDD GWAS)

> **목적:** 이 파일을 읽는 LLM agent가 `paper_draft_v10_submission_en.md`의 Discussion (Section 4)에 아래 5개 단락을 적절한 위치에 삽입하여 `paper_draft_v11_en.md`를 생성한다.
>
> **입력 파일:** `paper_draft_v10_submission_en.md` (영문 최종 제출본, 같은 디렉토리)
> **출력 파일:** `paper_draft_v11_en.md` (Discussion 보강본)
> **규칙:**
> - Discussion (Section 4) 이외의 섹션은 **일체 수정하지 않는다.**
> - 기존 Discussion 텍스트를 삭제하지 않고, 지정 위치에 새 단락을 **삽입**한다.
> - 학술 영어, 3인칭, 과거시제(결과 서술) / 현재시제(일반 사실) 혼용.
> - 인용은 `(저자 et al. 연도)` 형식으로 하되, References 섹션에 **아직 추가하지 않는다** (별도 작업).

---

## 삽입 1: SLC12A5/KCC2 생물학적 맥락 확장

**삽입 위치:** 현재 3번째 단락 ("Critically, however, summary-based confirmation tempers...") 끝, *SLC12A5* 언급 직후.

**삽입 내용 (영문으로 작성):**

The biological plausibility of *SLC12A5* (KCC2) as an MDD causal gene merits detailed discussion. KCC2 is the principal neuronal potassium-chloride cotransporter responsible for maintaining low intracellular Cl⁻ concentration in mature neurons, which is prerequisite for GABA_A receptor-mediated inhibition (Rivera et al. 1999). When KCC2 function is impaired, intracellular Cl⁻ rises and GABAergic signaling shifts from inhibitory toward excitatory, recapitulating the immature neuronal phenotype — a state termed "depolarizing GABA" (Ben-Ari et al. 2012). Several lines of preclinical evidence link KCC2 downregulation to stress and mood phenotypes: chronic stress reduces KCC2 expression in rodent hippocampus (Hewitt et al. 2009; Sarkar et al. 2011), and KCC2 heterozygous knockout mice exhibit increased anxiety- and depressive-like behaviors (Tornberg et al. 2005). Pharmacologically, the KCC2 enhancer CLP-290 has shown efficacy in restoring chloride homeostasis and reducing neuropathic pain in animal models (Gagnon et al. 2013), suggesting that KCC2-targeted compounds may be viable therapeutic candidates for CNS disorders including MDD. Our colocalization-confirmed finding (PP4 = 0.996) that genetically predicted *SLC12A5* expression is causally associated with MDD risk provides human genetic support for this preclinical body of work and identifies KCC2 enhancement as a mechanistically grounded therapeutic strategy warranting clinical investigation.

---

## 삽입 2: SLC12A5 약물/compound 논의

**삽입 위치:** 삽입 1 바로 뒤.

**삽입 내용:**

A critical translational question is whether SLC12A5/KCC2 is a druggable target. Unlike DRD2, which has dozens of approved ligands, KCC2 currently has no approved drugs. However, the small-molecule KCC2 enhancer CLP-290 (a prodrug of CLP-257) selectively potentiates KCC2 activity and has demonstrated efficacy in preclinical neuropathic pain and spasticity models (Gagnon et al. 2013; Bhatt et al. 2023). More recently, high-throughput screens have identified additional KCC2 activator scaffolds (Tang et al. 2024), and the gene has been designated a priority target by the NIMH. Our finding that *SLC12A5* is the most credible colocalization-confirmed MDD causal gene provides a human-genetics rationale for advancing KCC2 enhancers into mood-disorder indications. We emphasize, however, that genetic association with predicted expression change does not guarantee that pharmacological modulation of KCC2 at therapeutic concentrations will replicate the genetically indexed effect; dose-response validation in iPSC-derived neurons is a necessary next step (see Future Work).

---

## 삽입 3: DRD2 탈락 ≠ 도파민 가설 부정

**삽입 위치:** 현재 3번째 단락 끝 또는 삽입 1 앞. "We accordingly reposition these genes..." 문장 뒤.

**삽입 내용:**

We emphasize that our demotion of *DRD2* as a colocalized causal gene at this GWAS locus does **not** invalidate the broader dopaminergic hypothesis of MDD. DRD2 protein may contribute to MDD pathophysiology through trans-regulatory mechanisms, post-translational modifications, or receptor-level pharmacology that are not captured by cis-eQTL colocalization. Moreover, the GABA-dopamine interface is well established: GABAergic interneurons in the ventral tegmental area tonically inhibit dopaminergic projection neurons, and disruption of this inhibitory gate — precisely the consequence predicted by KCC2 (SLC12A5) dysfunction — could produce downstream dopaminergic dysregulation without requiring a cis-regulatory effect on *DRD2* expression itself (Bocklisch et al. 2013). Thus, the SLC12A5 finding may represent an upstream regulatory node whose dysfunction manifests, in part, as altered dopaminergic tone. The key distinction is between "*DRD2* cis-expression as the causal mediator at this locus" (not supported) and "dopamine signaling as a downstream effector in MDD" (not addressed by our analysis and potentially still valid).

---

## 삽입 4: 14% colocalization의 분야 함의

**삽입 위치:** 현재 5번째 단락 (convergent biology, GPX1 등) 뒤, 한계 단락 전.

**삽입 내용:**

A broader implication of our confirmatory analysis deserves emphasis. Of the 208 TWAS-significant genes with available BrainMeta cis-eQTL probes, only 30 (14%) colocalized (PP4 > 0.8). This finding is consistent with the emerging literature demonstrating that the majority of TWAS associations reflect LD tagging rather than shared causal variants (Wainberg et al. 2019; Mancuso et al. 2019). Critically, most published TWAS-based drug-repurposing studies do not perform systematic colocalization, and therefore their prioritized drug targets may include a substantial fraction of LD-confounded false positives. Our data suggest that approximately 86% of TWAS gene-drug nominations may lack causal support at the colocalization level. We therefore advocate that colocalization (or equivalent confirmatory methods such as probabilistic fine-mapping with SuSiE-coloc) be adopted as a mandatory step in any TWAS-to-drug-target pipeline, rather than treated as an optional sensitivity analysis.

---

## 삽입 5: PGC MDD2025 원 논문과의 차별점

**삽입 위치:** Discussion 첫 단락 끝 (전체 요약 후).

**삽입 내용:**

Our study extends the PGC MDD2025 discovery paper (Adams et al. 2025) in three specific ways that the original publication did not address. First, we performed systematic direction-aware drug-repurposing filtering, which the original study did not include, reducing 678 raw gene-drug matches to 102 directionally consistent candidates. Second, we applied genome-wide SMR/HEIDI and Bayesian colocalization against brain cis-eQTL to confirm or exclude TWAS-nominated causal genes — a step absent from the original analysis, which reported TWAS results without colocalization confirmation. Third, we explicitly demoted the strongest TWAS signal (*DRD2*) based on colocalization failure and repositioned *SLC12A5* as the confirmed lead — a conclusion that could not have been drawn without the confirmatory layer. These additions transform a gene-discovery catalogue into a causally filtered, therapeutically actionable target shortlist.

---

## 추가 수정 사항

### 한계 섹션 축소 (9개 → 5개로 합치기)

현재 한계 (1)~(9)를 다음과 같이 5개로 재구성:

1. **TWAS ≠ causality** — (1)과 (2)를 합침. "TWAS는 인과가 아니며, colocalization으로 보완했으나 trans-effects와 reverse causation은 배제 못함."
2. **Coverage & ancestry** — (7)과 (8)을 합침. "SNP 커버리지 78% (LiftOver 미적용) + European-only → 위음성 및 일반화 제한."
3. **Residual stratification** — (3) 유지. "LDSC intercept 7.6%, liability-scale h²≈0.084."
4. **Replication & cross-disorder** — (4)와 (5)를 합침. "단일 코호트, 독립 재현 미수행. DRD2는 colocalization으로 이미 제외됐으나 mtCOJO 필요."
5. **Druggability & direction assumptions** — (6)과 (9)를 합침. "Multiple testing은 ACAT로 보정. DGIdb 방향 필터는 자가수용체/보상 효과로 역전 가능한 휴리스틱."

### Sex differences 한 줄 추가

한계 섹션 (2)에 추가:
> "Furthermore, sex-stratified analyses were not performed; given the approximately 2:1 female-to-male MDD prevalence ratio, sex-specific genetic architecture may exist that is masked in the combined analysis."

### Abstract 압축 (별도 작업)

Abstract를 300 words 이하로 압축하는 것은 이 지시서의 범위 밖. 별도 지시로 진행.

---

## 실행 절차 요약

```
1. paper_draft_v10_submission_en.md를 읽는다.
2. Discussion (Section 4)에서 지정 위치를 찾는다.
3. 삽입 1~5를 해당 위치에 추가한다.
4. 한계 섹션을 9개→5개로 재구성한다.
5. Sex differences 한 줄을 추가한다.
6. Discussion 이외 섹션은 변경하지 않는다.
7. paper_draft_v11_en.md로 저장한다.
```
