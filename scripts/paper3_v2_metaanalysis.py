#!/usr/bin/env python3
"""
Paper 3 v2 — Pooled effect-size meta-analysis from full-text LLM mapping.

SKELETON / not-yet-runnable end-to-end: requires the cluster job output
`data/stress/fulltext_extract/llm_mapped_35b.json` (see
paper3/TASK_fulltext_llm_mapping.md). Once that file exists, this script:

  1. loads llm_mapped_35b.json (per-sentence LLM-mapped records)
  2. flattens to one row per (pmid, brain_region, metric, direction, value, measure_type)
  3. harmonizes effect-size types (r, Cohen's d, Hedges g, %change, ...) to a
     common scale where convertible (r<->d), else keeps grouped by measure_type
  4. random-effects meta-analysis (DerSimonian-Laird) per
     (brain_region x metric x stress_type), with I^2 heterogeneity
  5. forest plots for the headline contrasts that v1 raised as "reporting
     patterns" (BDNF-hippocampus direction; amygdala structural vs functional;
     neuroinflammation vs HPA) — now as POOLED estimates
  6. writes results/paper3_v2_pooled.csv + figures/v2_forest_*.png and prints a
     v1-vs-v2 comparison table.

Design notes:
- v1 = reporting patterns (counts of decrease/increase). v2 = pooled magnitude
  + direction with CIs. Keep the honest framing: only sentences with usable
  numbers + a confidently mapped region/metric enter the pool; report coverage
  (how many of 14,261 / how many studies) explicitly — no silent dropping.
- r<->d conversion: d = 2r/sqrt(1-r^2); var(d) from var(r). Pool in d, report
  both. %-change and raw volumes stay in separate strata (not pooled with d).
"""
import json
import math
import os
from collections import defaultdict

IN = "data/stress/fulltext_extract/llm_mapped_35b.json"
OUT_CSV = "paper3/results_v2/paper3_v2_pooled.csv"
FIG_DIR = "paper3/figures_v2"


def load_mapped(path=IN):
    if not os.path.exists(path):
        raise SystemExit(
            f"[paper3_v2] input not found: {path}\n"
            "Run the cluster job first (paper3/TASK_fulltext_llm_mapping.md)."
        )
    return json.load(open(path, encoding="utf-8"))


def flatten(records):
    """records -> list of dicts: pmid, region, metric, direction, value, mtype."""
    rows = []
    for rec in records:
        parsed = rec.get("llm_parsed")
        if not parsed:
            continue
        for e in parsed:
            try:
                rows.append({
                    "pmid": rec["pmid"],
                    "region": (e.get("brain_region") or "").strip().lower(),
                    "metric": (e.get("metric") or "").strip().lower(),
                    "direction": (e.get("direction") or "").strip().lower(),
                    "value": float(e["value"]) if e.get("value") is not None else None,
                    "mtype": (e.get("measure_type") or "").strip().lower(),
                })
            except (KeyError, ValueError, TypeError):
                continue
    return rows


def r_to_d(r):
    r = max(min(r, 0.999), -0.999)
    return 2 * r / math.sqrt(1 - r * r)


def dl_pool(effects, variances):
    """DerSimonian-Laird random-effects pooled estimate. Returns (mu, se, I2, k)."""
    k = len(effects)
    if k == 0:
        return None
    w = [1 / v if v and v > 0 else 0 for v in variances]
    if sum(w) == 0:
        return None
    fixed = sum(e * wi for e, wi in zip(effects, w)) / sum(w)
    Q = sum(wi * (e - fixed) ** 2 for e, wi in zip(effects, w))
    df = k - 1
    C = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr = [1 / (v + tau2) if (v + tau2) > 0 else 0 for v in variances]
    mu = sum(e * wi for e, wi in zip(effects, wr)) / sum(wr)
    se = math.sqrt(1 / sum(wr))
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    return mu, se, I2, k


def main():
    records = load_mapped()
    rows = flatten(records)
    print(f"[paper3_v2] mapped records: {len(records)} | usable effect rows: {len(rows)}")
    # coverage report (honest)
    n_parsed = sum(1 for r in records if r.get("llm_parsed"))
    print(f"[paper3_v2] parse coverage: {n_parsed}/{len(records)} "
          f"({n_parsed/len(records)*100:.1f}%) sentences mapped")

    # TODO once data exists:
    #   - group rows by (region, metric) [optionally x stress_type via pmid->meta]
    #   - convert r->d where mtype=='r'; estimate variance (needs n: join sample sizes)
    #   - dl_pool per group; write CSV; forest plots for headline contrasts
    #   - v1-vs-v2 comparison (does pooled magnitude confirm the count-based pattern?)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print("[paper3_v2] SKELETON ready. Implement grouping/pooling once "
          "llm_mapped_35b.json lands. (see module docstring steps 3-6)")


if __name__ == "__main__":
    main()
