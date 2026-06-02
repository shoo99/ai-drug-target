# Research Square 제출 가이드 — Paper 3 (Chronic Stress × Brain Scoping Review)

> 제출 본문: `paper3/manuscript.pdf` (영문, 15p — 그림·표 임베드 완결본)
> 한국어본(참고용): `paper3/manuscript_ko.md` / `manuscript_ko.pdf`
> 그림(개별 첨부 필요시): `paper3/figures/fig1_yearly_trend` … `fig7_consistency` (.png/.pdf 각 7개)
> 포털: https://www.researchsquare.com  (로그인 후 "Post a preprint")
> ⚠️ `paper_draft_v10_submission_*.md`는 Paper 4 내용이 잘못 복사된 파일 — 제출에 쓰지 말 것.

---

## 1. Title (그대로 복사)

Large-Scale Scoping Review of Chronic Stress-Induced Brain Changes via LLM-Powered Extraction of 9,585 Studies: BDNF Directional Inconsistency, Structural-Functional Dissociation, and the Neuroinflammation-HPA Crossover

---

## 2. Abstract (plain text, 그대로 복사)

Background: The effects of chronic stress on brain structure and function have been extensively studied, producing well-established narratives such as "stress reduces hippocampal BDNF" and "stress increases amygdala volume." However, these narratives are typically derived from qualitative reviews or small-scale meta-analyses. Whether they hold quantitatively across the full body of literature has not been systematically evaluated at scale.

Methods: We extracted structured data from 9,585 PubMed abstracts (2008-2026) using a distributed CPU cluster running Qwen3.5-397B (Q3_K_M quantization, temperature=0) across 9 nodes. For each article, we extracted study type (human/animal), stress type (chronic/PTSD/early-life/acute), gene mentions (HUGO symbols), brain regions with effect direction (decrease/increase/no change), and measurement metric (structural/functional/cellular). We performed metric-separated analysis, temporal trend analysis, cross-study consistency scoring, and research gap identification using observed-versus-expected co-occurrence ratios.

Results: Five findings challenge or refine established narratives: (1) BDNF direction in the hippocampus is not consistently downward—reports of decrease (46%) and increase (46%) are nearly equal across 1,434 BDNF-hippocampus entries, with a temporal reversal from decrease-dominant (56%, 2010-2014) to increase-dominant (57%, 2020-2026); (2) this reversal is stress-type-dependent: early-life stress strongly favors BDNF decrease (55%) while acute stress favors increase (46%); (3) neuroinflammation genes (IL1B, TNF, IL6, NFKB1, NLRP3; combined 1,143 mentions) have overtaken HPA axis genes (NR3C1, CRH, FKBP5; 597 mentions) since approximately 2018 (inflammation/HPA ratio: 0.32 in 2010-2014, 1.03 in 2015-2019, 2.15 in 2020-2026); (4) metric-separated analysis reveals that the hippocampus shows structural decrease (67%) but functional increase (41%), and the amygdala shows structural decrease (50%) but functional increase (45%)—resolving the apparent contradictions in the literature; (5) systematic research gap analysis across 2,157 genes and 1,930 brain regions identified 15 significantly under-studied combinations, with the insula-inflammation axis as the largest gap.

Conclusions: This large-scale scoping review, based on LLM-powered extraction from 9,585 abstracts, reveals that established stress-brain narratives are quantitatively more complex than textbook summaries suggest. The "BDNF always decreases" and "amygdala always increases" simplicities do not survive metric-separated, temporally resolved analysis of reporting patterns. These patterns provide an evidence-based roadmap for future research priorities and demonstrate the utility of LLM-based extraction as a complement to traditional systematic reviews. We emphasize that these are literature reporting patterns, not pooled effect sizes; full-text quantitative meta-analysis is in progress. The complete list of all 9,585 included studies (PMID, title, journal, year, PMC ID) is provided as Supplementary Table S1, enabling full retrieval of the source literature from PubMed/PMC under each article's respective license.

---

## 3. Keywords (본문 그대로)

chronic stress; brain structure; BDNF; neuroinflammation; HPA axis; hippocampus; amygdala; large language model (LLM); natural language processing; scoping review; reporting patterns

---

## 4. Subject area / Category (RS 선택지)

- Primary: Neuroscience  (또는 Psychiatry)
- Secondary: Bioinformatics / Computational Biology; Systematic Reviews & Meta-analyses

---

## 5. Authors & Affiliations

- Byeongsoo Kang¹*  (¹ SYSOFT) — 교신저자 shoo99@gmail.com
  (본문 기재값. 기존 프리프린트와 동일해야 하면 RS 계정값으로 통일.)

---

## 6. Ethics declarations  (Paper 4와 동일 로직)

- "involved human subjects" → **체크 안 함** (출판된 PubMed 초록의 텍스트 마이닝, 신규 피험자 없음)
- "involved non-human vertebrates" → **체크 안 함** (문헌 내 동물연구를 *집계*할 뿐, 신규 동물실험 없음)

## 7. Competing interests → **"No"** (이해관계 없음)

## 8. Data & Code Availability (본문 명시 — A방침: 데이터셋 비공개, 목록만 공개)

- 포함 연구 전체 목록: **Supplementary Table S1** = `supplementary_S1_included_studies.csv` (9,585행; PMID·title·journal·year·study_type·PMC ID) — RS에 Supplementary file로 함께 업로드.
- 원 초록/전문: Supplementary Table S1의 PMID/PMC ID로 PubMed/PMC에서 각 논문 라이선스 하에 조회.
- 구조화 추출 데이터·분석 코드: 교신저자에 합리적 요청 시 제공 (HF 데이터셋·PMC 원문 tarball은 저작권상 비공개 유지).

---

## 9. 제출 순서 (포털)

1. researchsquare.com 로그인 → New preprint / "Post a preprint"
2. Title·Abstract·Keywords·Subject 붙여넣기 (위 1~4)
3. 본문 PDF 업로드: `paper3/manuscript.pdf` (그림 7개 임베드 완결, HF/CC-BY 문구 제거·Table S1 반영본)
   - **Supplementary file로 `supplementary_S1_included_studies.csv` 반드시 함께 업로드** (본문이 S1을 참조)
   - figure 별도 요구 시 `paper3/figures/fig1-7` (.pdf 권장)
4. Authors/affiliations (5), Ethics (6) 둘 다 미체크, Competing interests "No" (7)
5. Data availability (8) 입력
6. Submit → DOI 발급

---

## 부록. 제출 전 데이터 검증 결과 (2026-06-01, 본 세션)
실제 데이터(`data/stress/abstracts.parquet`)와 원고 핵심수치 대조 — 모두 일치:
- 10,001 수집 → 416 파싱실패(study_type NaN 정확히 416) → **9,585 valid** ✓ (산술·데이터 일치)
- study_type: animal 44.3% / human 44.1% ✓ (원고와 정확 일치)
- stress_type: chronic 2,725 / PTSD 1,282 / early-life 937 ✓ (멤버십 집계, 정확 일치)
- inflammation/HPA gene mentions: 전체연도 ratio 2.24, 원고 2.15는 2020–26 시간창 한정값 (측정창 차이, 정상)
- 그림 7개 존재, manuscript.pdf 15p 현행
→ 제출 가능 상태 확인.
