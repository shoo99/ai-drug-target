# References 누락 수정 지시서 — Paper 4 v13

> **목적:** `paper_draft_v13_en.md`의 References 섹션에 누락된 6개 참고문헌을 추가하고 `paper_draft_v14_en.md`로 저장한다.
> **한국어본:** `paper_draft_v13_ko.md`의 References에도 동일하게 추가하고 `paper_draft_v14_ko.md`로 저장한다.
>
> **규칙:**
> - References 섹션 이외의 본문은 **일체 수정하지 않는다.**
> - 기존 18개 참고문헌 뒤에 6개를 번호 순서대로 추가한다 (19~24번).
> - 형식: 기존 참고문헌과 동일한 스타일 (저자, 제목, 저널, 연도, DOI).

---

## 누락된 6개 참고문헌

아래 6개는 본문 Introduction (Section 1) 및 Methods (Section 2)에서 인용되었으나 References에 없는 것이다.

### 19. Nelson et al. 2015
> **인용 위치:** Introduction, "Nelson et al. (2015) and King et al. (2019) demonstrated that genetically supported drug targets are twice as likely..."

```
19. Nelson MR, Tipney H, Painter JL, et al. The support of human genetic evidence for approved drug indications. Nat Genet. 2015;47(8):856–860. doi:10.1038/ng.3314
```

### 20. King et al. 2019
> **인용 위치:** Introduction, 같은 문장.

```
20. King EA, Davis JW, Bhagwat JG. Are drug targets with genetic support twice as likely to be approved? Revised estimates of the impact of genetic support for drug mechanisms on the probability of drug approval. PLoS Genet. 2019;15(12):e1008489. doi:10.1371/journal.pgen.1008489
```

### 21. Wray et al. 2018
> **인용 위치:** Introduction, "the landmark PGC meta-analysis by Wray et al. (2018) identified 44 risk loci"

```
21. Wray NR, Ripke S, Mattheisen M, et al. Genome-wide association analyses identify 44 risk variants and refine the genetic architecture of major depression. Nat Genet. 2018;50(5):668–681. doi:10.1038/s41588-018-0090-3
```

### 22. Howard et al. 2019
> **인용 위치:** Introduction, "later expanded by Howard et al. (2019) to 102 loci"

```
22. Howard DM, Adams MJ, Clarke TK, et al. Genome-wide meta-analysis of depression identifies 102 independent variants and highlights the importance of the prefrontal brain regions. Nat Neurosci. 2019;22(3):343–352. doi:10.1038/s41593-018-0326-7
```

### 23. Bulik-Sullivan et al. 2015
> **인용 위치:** Methods 2.6 및 Results 3.1, "LDSC (Bulik-Sullivan 2015)"

```
23. Bulik-Sullivan BK, Loh PR, Finucane HK, et al. LD Score regression distinguishes confounding from polygenicity in genome-wide association studies. Nat Genet. 2015;47(3):291–295. doi:10.1038/ng.3211
```

### 24. So et al. 2017
> **인용 위치:** Introduction, "Early efforts such as the study by So et al. (2017)" (v10에 있었으나 v13에서도 인용 가능성 확인)

```
24. So HC, Chau CKL, Lau A, Wong SY, Zhao K. Translating GWAS findings into therapies for depression and anxiety disorders: gene-set analyses reveal enrichment of psychiatric drug classes and implications for drug repositioning. Psychol Med. 2019;49(16):2692–2708. doi:10.1017/S0033291718003641
```

> **참고:** So et al.이 v13 본문에서 인용되지 않았다면, #24는 추가하지 않아도 된다. 먼저 본문에서 "So et al." 또는 "So HC"를 검색하고, 인용이 없으면 #24는 건너뛴다.

---

## 실행 절차

```
1. paper_draft_v13_en.md를 읽는다.
2. "## References" 섹션을 찾는다.
3. 기존 18번 (Zhu et al. 2016) 뒤에 위 6개(또는 5개)를 순서대로 추가한다.
4. References 이외 섹션은 변경하지 않는다.
5. paper_draft_v14_en.md로 저장한다.
6. paper_draft_v13_ko.md에 대해서도 동일하게 수행한다:
   - 한국어본의 References 섹션에 같은 6개를 추가 (영문 그대로, 참고문헌은 번역하지 않음).
   - paper_draft_v14_ko.md로 저장한다.
```

---

## 검증 체크리스트

수정 후 아래를 확인:

- [ ] References가 총 23~24개인가? (18 + 5~6)
- [ ] 본문에서 "Nelson", "King", "Wray", "Howard", "Bulik-Sullivan"을 검색했을 때 각각 References에 대응하는 항목이 있는가?
- [ ] 기존 1~18번 참고문헌이 변경되지 않았는가?
- [ ] Abstract, Introduction, Methods, Results, Discussion 본문이 v13과 byte-identical한가?
- [ ] 영문본과 한국어본의 References 수가 동일한가?
