# TASK: Paper 3 v2 — Full-text Effect-Size LLM Mapping (KBRI cluster job)

> **For: KBRI cluster management bot.** Run this job on the cluster (35B node), then push results back to GitHub. The analysis/manuscript side is handled separately once results land.
> **Created:** 2026-06-02 · **Paper 3** (Chronic Stress × Brain scoping review; preprint v1 = RS DOI 10.21203/rs.3.rs-9884522/v1)
> **Goal:** Upgrade Paper 3 from "reporting patterns" (abstract-level) to a quantitative meta-analysis (v2) by mapping regex-extracted full-text effect sizes to (brain region × metric × direction) via the 35B LLM.

---

## 0. Context (why)

Paper 3 v1 reports **literature reporting patterns** from 9,585 abstracts (no pooled effect sizes). v2 needs **actual pooled effect sizes** from full text. Step 1–3 (regex pre-extraction) is already DONE: 14,261 candidate sentences with numeric values (effect sizes, CIs, p-values, sample sizes) + nearby brain-region regex hits. Step 4 (this task) = LLM maps each sentence's numbers to a structured record so they can be pooled.

This is the hybrid pipeline from memory: **regex pre-extract (done) + 35B short-prompt mapping (this task)** — keeps each LLM output small (<512 tokens) to avoid the Q3-quantization JSON-break issue.

---

## 1. Input

- **File:** `data/stress/fulltext_extract/llm_prompts.json` (in `shoo99/ai-drug-target`, **public** repo; also mirror exists under KBRI pipeline dirs)
- **Count:** 14,261 prompts
- **Each item schema:**
  ```json
  {
    "pmid": "40226700",
    "sentence": "<full sentence from PMC full text>",
    "extracted_effect_sizes": [{"type":"r","value":0.08}, ...],
    "extracted_ci": null,
    "extracted_p_values": [{"op":"=","value":0.531}, ...],
    "extracted_sample_sizes": [],
    "regex_brain_regions": ["precuneus","occipital"],
    "llm_prompt": "Given this sentence from a neuroscience paper: ... <numbers> ... Return a JSON array mapping each number to {brain_region, metric, direction, value, measure_type}."
  }
  ```

## 2. Model / endpoint

- **Model:** Qwen3.6-35B-A3B (MoE) — short-input mapping, NOT the 397B (that was abstract NER).
- **Endpoint(s):** llama.cpp `/completion`, `temperature=0`, `n_predict=512`, stop `["\n\n","```"]`.
  - Known node from prior runs: `http://172.20.0.12:11202` (com02). **Use all available 35B-capable nodes** for throughput (the abstract run used 8 nodes com04–com14 at ports 11204–11213; reuse whatever 35B servers are up).
- Prompt wrapping (already in script): `<|im_start|>user\n{llm_prompt}<|im_end|>\n<|im_start|>assistant\n`

## 3. Runner script

`scripts/fulltext_llm_map.py` (already in repo). Run:

```bash
cd ~/ai-drug-target   # or wherever the repo is on the cluster
python scripts/fulltext_llm_map.py \
    --input  data/stress/fulltext_extract/llm_prompts.json \
    --output data/stress/fulltext_extract/llm_mapped_35b.json \
    --server http://172.20.0.12:11202 \
    --workers 8
```

**⚠️ Recommended improvements before/while running (the bot may apply):**
1. **Multi-server round-robin** — the current script takes a single `--server`. To use N nodes, either launch N processes each with a disjoint slice (`--limit`/offset) and a different `--server`, OR edit `SERVERS=[...]` list and round-robin in `process_prompt`. Distributing across 8 nodes → ETA ~1.5–3 h (14,261 × ~3 s / 8).
2. **Checkpointing** — write results incrementally (e.g., append per-100 to a JSONL, or save partial every 500) so a crash doesn't lose progress. Current script only saves at the very end.
3. **Retry on `ERROR:`/unparsed** — re-queue items whose `llm_parsed` is null (target >90% parse rate, like the abstract run's 95.8%).

## 4. Expected output

- **File:** `data/stress/fulltext_extract/llm_mapped_35b.json`
- Each record (script already builds this): `pmid, sentence, regex_* (the inputs), llm_raw, llm_parsed`
  where `llm_parsed` is a JSON array like:
  ```json
  [{"brain_region":"precuneus","metric":"GMV","direction":"no_change","value":0.08,"measure_type":"r"}, ...]
  ```
- **Quality target:** ≥90% of 14,261 with non-null `llm_parsed`. Print final parse-rate (script does).

## 5. Deliverable — push back to GitHub

Commit `llm_mapped_35b.json` to **`shoo99/KBRI`** (private), e.g. under `gwas-mdd/../` or a new `paper3-fulltext/` folder — anywhere, just report the path. Then notify (Discord) "Paper 3 v2 mapping done, results at <path>, parse-rate X%".

> NOTE: do NOT commit the raw PMC full-text tarball or any non-CC-BY full text to a public repo (license-mixed). The mapped JSON (numbers + region/metric labels, no article text beyond single quoted sentences) is fine for the private KBRI repo.

## 6. After results land (assistant will do)

Once `llm_mapped_35b.json` is available, the assistant will: (a) pool effect sizes per (region × metric × stress-type), (b) random-effects meta-analysis + heterogeneity (I²), (c) forest plots, (d) compare v1 reporting-pattern findings vs v2 pooled estimates, (e) write Paper 3 v2 manuscript. A v2-analysis skeleton is being prepared at `scripts/paper3_v2_metaanalysis.py`.
