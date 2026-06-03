#!/usr/bin/env python3
"""
Paper 3 v2 — effect-size SYNTHESIS with 2nd-pass noise cleaning (D).

Takes the v3 LLM mapping (llm_mapped_35b_v3.json) and applies a sentence-level
noise filter (drop laterality L-vs-R, ratios, coordinate/cluster stats,
protective-factor, task-performance correlations, covariate-predictor sentences)
BEFORE synthesizing direction/magnitude per (region x metric class).

Honest framing: this is effect-size synthesis (sign test + magnitude
distribution), NOT inverse-variance meta-analysis (sample sizes are in ~5% of
records). Reports clean-set coverage + before/after so the cleaning effect is
transparent.
"""
import json, os, re, csv
from collections import defaultdict, Counter
from math import comb

IN = "/tmp/KBRI_sparse/paper3-fulltext-v2/output/llm_mapped_35b_v3.json"
OUTDIR = "/home/sysoft/ai-drug-target/paper3/results_v2"

NOISE = {
    "laterality": re.compile(r"\b(left|right).{0,25}(vs|versus|than|compared|>|<).{0,25}(left|right)\b|hemispher|laterali", re.I),
    "ratio": re.compile(r"ratio|proportion", re.I),
    "coords": re.compile(r"\b[xyz]\s*=\s*[-−]?\d|MNI|peak voxel|cluster.?(size|k\b)|TFCE|talairach", re.I),
    "protective": re.compile(r"maternal support|social support|resilience|protective|positive parenting|secure attach", re.I),
    "task_corr": re.compile(r"task (score|performance)|cognitive (score|test|performance)|\bERT\b|\bIQ\b|reaction time", re.I),
    "covariate": re.compile(r"\b(age|sex|gender|education|medication)\s*(was|were)?\s*(a |an )?(significant )?(covariate|predictor|associated|factor)", re.I),
}

REGION_KEYS = {
    "hippocampus": ["hippocamp"], "amygdala": ["amygdal"],
    "prefrontal cortex": ["prefrontal", "pfc", "mpfc", "dlpfc", "vmpfc"],
    "ACC": ["anterior cingulate", "cingulate"], "insula": ["insula", "insular"],
    "thalamus": ["thalam"], "striatum": ["striat", "caudate", "putamen", "accumbens"],
}
STRUCT = {"volume", "gmv", "thickness", "gray matter", "fa", "density"}
FUNC = {"connectivity", "activation", "functional", "rsfc", "alff", "reho", "bold"}

def norm_region(raw):
    r = (raw or "").lower()
    for canon, keys in REGION_KEYS.items():
        if any(k in r for k in keys): return canon
    return None

def norm_dir(raw):
    d = (raw or "").lower()
    if "increase" in d or "positive" in d: return "increase"
    if "decrease" in d or "negative" in d: return "decrease"
    if "no" in d: return "null"
    return "other"

def metric_class(m):
    m = (m or "").lower()
    if any(s in m for s in STRUCT): return "structural"
    if any(f in m for f in FUNC): return "functional"
    return "other"

def is_noise(sent):
    return any(p.search(sent) for p in NOISE.values())

def sign_p(inc, dec):
    n = inc + dec
    if n == 0: return 1.0
    k = min(inc, dec)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))

def synth(records, label):
    by = defaultdict(Counter); studies = defaultdict(set); rmag = defaultdict(list)
    n_ent = 0
    for r in records:
        for e in (r.get("llm_parsed") or []):
            if not isinstance(e, dict): continue
            reg = norm_region(e.get("brain_region"))
            if not reg: continue
            n_ent += 1
            mc = metric_class(e.get("metric"))
            by[(reg, mc)][norm_dir(e.get("direction"))] += 1
            studies[(reg, mc)].add(r["pmid"])
            if (e.get("measure_type") or "").lower() == "r" and isinstance(e.get("value"), (int, float)):
                rmag[reg].append(e["value"])
    print(f"\n===== {label}: {len(records)} sentences, {n_ent} region-entries =====")
    print(f"{'region':<18}{'metric':<12}{'dec':>5}{'inc':>5}{'null':>5}{'studies':>8}{'sign_p':>10}")
    rows = []
    for key in sorted(by, key=lambda k: -(by[k]['increase'] + by[k]['decrease'])):
        reg, mc = key; c = by[key]
        if mc == "other" or (c['increase'] + c['decrease']) < 8: continue
        p = sign_p(c['increase'], c['decrease'])
        print(f"{reg:<18}{mc:<12}{c['decrease']:>5}{c['increase']:>5}{c['null']:>5}{len(studies[key]):>8}{p:>10.2e}")
        rows.append({"set": label, "region": reg, "metric_class": mc, "decrease": c['decrease'],
                     "increase": c['increase'], "null": c['null'], "n_studies": len(studies[key]), "sign_p": p})
    print("  r-magnitude:", {reg: f"med={sorted(v)[len(v)//2]:+.2f}(n{len(v)})" for reg, v in
                              sorted(rmag.items(), key=lambda x: -len(x[1])) if len(v) >= 5})
    return rows

def main():
    d = json.load(open(IN, encoding="utf-8"))
    mapped = [r for r in d if r.get("llm_parsed") and len(r["llm_parsed"]) > 0]
    clean = [r for r in mapped if not is_noise(r["sentence"])]
    dirty = [r for r in mapped if is_noise(r["sentence"])]
    print(f"v3 mapped: {len(mapped)} | clean: {len(clean)} | flagged noise: {len(dirty)} "
          f"({len(dirty)/len(mapped)*100:.0f}%)")
    rows_all = synth(mapped, "BEFORE-clean (all v3)")
    rows_clean = synth(clean, "AFTER-clean")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "v2_clean_synthesis.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set","region","metric_class","decrease","increase","null","n_studies","sign_p"])
        w.writeheader(); w.writerows(rows_all + rows_clean)
    print(f"\n[v2] wrote {OUTDIR}/v2_clean_synthesis.csv")
    print(f"[v2] CAVEATS: clean set = {len(clean)} sentences from "
          f"{len(set(r['pmid'] for r in clean))} studies; sample size in ~5% -> "
          f"sign-test/magnitude synthesis, not pooled meta-analysis.")

if __name__ == "__main__":
    main()
