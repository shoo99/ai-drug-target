#!/usr/bin/env python3
"""
Paper 3 v2 — Large-scale effect-size SYNTHESIS from full-text LLM mapping.

NOT a pooled meta-analysis (sample sizes available in only ~4% of records, so
inverse-variance weighting is impossible). Instead, an honest effect-size
synthesis over the LLM-mapped full-text effect sizes:
  - per (brain_region x metric) and per headline contrast:
      n entries, n studies (unique PMID), direction split (dec/inc/null),
      sign test p-value, median |effect| by measure_type
  - r-type effects: median r + sign of correlation
  - compares v1 reporting-pattern direction vs v2 effect-size direction

Input : /tmp/KBRI_sparse/paper3-fulltext-v2/output/llm_mapped_35b.json
Output: paper3/results_v2/*.csv  + console summary
Caveats (printed + for manuscript Limitations): recall ~57% (1,818 effect+region
sentences returned empty by 35B), n missing in 96% -> no inverse-variance pooling.
"""
import json
import os
import re
import math
from collections import defaultdict, Counter

IN = "/tmp/KBRI_sparse/paper3-fulltext-v2/output/llm_mapped_35b.json"
OUTDIR = "/home/sysoft/ai-drug-target/paper3/results_v2"

REGION_KEYS = {
    "hippocampus": ["hippocamp"],
    "amygdala": ["amygdal"],
    "prefrontal cortex": ["prefrontal", "pfc", "mpfc", "dlpfc", "vmpfc"],
    "ACC": ["anterior cingulate", "acc ", " acc", "cingulate"],
    "insula": ["insula", "insular"],
    "thalamus": ["thalam"],
    "striatum": ["striat", "caudate", "putamen", "accumbens"],
}

def norm_region(raw):
    r = (raw or "").lower()
    for canon, keys in REGION_KEYS.items():
        if any(k in r for k in keys):
            return canon
    return None

def norm_dir(raw):
    d = (raw or "").lower()
    if "increase" in d or d == "positive_correlation" or "positive" in d:
        return "increase"
    if "decrease" in d or d == "negative_correlation" or "negative" in d:
        return "decrease"
    if "no" in d:  # no_change / no significant ...
        return "null"
    return "other"

def sign_test_p(inc, dec):
    """Two-sided exact binomial sign test under p=0.5 (ignore nulls)."""
    n = inc + dec
    if n == 0:
        return 1.0
    k = min(inc, dec)
    # P(X<=k) * 2, X~Bin(n,0.5)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

def main():
    d = json.load(open(IN, encoding="utf-8"))
    parsed = [r for r in d if r.get("llm_parsed")]
    os.makedirs(OUTDIR, exist_ok=True)

    # flatten entries
    rows = []
    for r in parsed:
        pmid = r["pmid"]
        p = r["llm_parsed"]
        if not isinstance(p, list):
            continue
        for e in p:
            if not isinstance(e, dict):
                continue
            reg = norm_region(e.get("brain_region"))
            rows.append({
                "pmid": pmid,
                "region": reg,
                "region_raw": (e.get("brain_region") or "").lower(),
                "metric": (e.get("metric") or "").lower(),
                "direction": norm_dir(e.get("direction")),
                "mtype": (e.get("measure_type") or "").lower(),
                "value": e.get("value"),
            })
    print(f"[v2] mapped records {len(parsed)} -> {len(rows)} effect entries")
    print(f"[v2] entries mapped to a core region: {sum(1 for x in rows if x['region'])}")

    # ---- per-region direction synthesis (structural vs functional) ----
    STRUCT = {"volume", "gmv", "thickness", "gray matter", "fa", "density"}
    FUNC = {"connectivity", "activation", "functional connectiv", "rsfc", "alff", "reho", "bold"}
    def metric_class(m):
        if any(s in m for s in STRUCT): return "structural"
        if any(f in m for f in FUNC): return "functional"
        return "other"

    print("\n=== Per-region × metric-class direction (effect-size entries) ===")
    summ = []
    by = defaultdict(lambda: Counter())
    studies = defaultdict(set)
    for x in rows:
        if not x["region"]:
            continue
        mc = metric_class(x["metric"])
        key = (x["region"], mc)
        by[key][x["direction"]] += 1
        studies[key].add(x["pmid"])
    hdr = f"{'region':<18}{'metric':<12}{'n':>5}{'studies':>8}{'dec':>6}{'inc':>6}{'null':>6}{'sign_p':>9}"
    print(hdr); print("-"*len(hdr))
    for key in sorted(by, key=lambda k: -(by[k]['increase']+by[k]['decrease'])):
        reg, mc = key
        c = by[key]
        inc, dec, nul = c["increase"], c["decrease"], c["null"]
        n = inc+dec+nul+c["other"]
        if mc == "other" or (inc+dec) < 10:
            continue
        p = sign_test_p(inc, dec)
        line = f"{reg:<18}{mc:<12}{n:>5}{len(studies[key]):>8}{dec:>6}{inc:>6}{nul:>6}{p:>9.2e}"
        print(line)
        summ.append({"region":reg,"metric_class":mc,"n_entries":n,"n_studies":len(studies[key]),
                     "decrease":dec,"increase":inc,"null":nul,"sign_test_p":p})

    # ---- r-type correlation magnitude per region ----
    print("\n=== r-type effect magnitude per region (median r) ===")
    rmag = defaultdict(list)
    for x in rows:
        if x["region"] and x["mtype"] == "r" and isinstance(x["value"], (int, float)):
            rmag[x["region"]].append(x["value"])
    for reg in sorted(rmag, key=lambda k: -len(rmag[k])):
        vals = sorted(rmag[reg])
        if len(vals) < 5: continue
        med = vals[len(vals)//2]
        absmed = sorted(abs(v) for v in vals)[len(vals)//2]
        print(f"  {reg:<18} n_r={len(vals):>4}  median_r={med:+.3f}  median|r|={absmed:.3f}")

    # write CSV
    import csv
    with open(os.path.join(OUTDIR, "v2_region_metric_direction.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region","metric_class","n_entries","n_studies","decrease","increase","null","sign_test_p"])
        w.writeheader(); w.writerows(summ)
    print(f"\n[v2] wrote {OUTDIR}/v2_region_metric_direction.csv")

    # ---- coverage / caveats ----
    print("\n=== COVERAGE / CAVEATS (for Limitations) ===")
    n_all = len(d); n_mapped = len(parsed)
    print(f"  prompts: {n_all}; mapped (non-empty): {n_mapped} ({n_mapped/n_all*100:.1f}%)")
    print(f"  unique studies represented: {len(set(r['pmid'] for r in parsed))}")
    print(f"  WARNING: recall ~57% (empty-array on many valid sentences); n available in ~4% -> "
          f"sign-test/vote-count synthesis, NOT inverse-variance meta-analysis.")

if __name__ == "__main__":
    main()
