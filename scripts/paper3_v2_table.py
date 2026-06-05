#!/usr/bin/env python3
"""
Paper 3 v2 -- canonical Table 1 builder.

Reads the v3 LLM mapping, applies the SAME sentence-level noise filter as
paper3_v2_synthesis_clean.py, then reports BOTH the study-level majority-vote
sign test (primary) and the entry-level sign test (secondary) for every
region x metric class with >=8 directional entries. Writes the table to CSV so
the manuscript numbers are reproducible from one command.
"""
import json, re, os, csv
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


def nr(raw):
    r = (raw or "").lower()
    for c, ks in REGION_KEYS.items():
        if any(k in r for k in ks):
            return c
    return None


def nd(raw):
    d = (raw or "").lower()
    if "increase" in d or "positive" in d:
        return "increase"
    if "decrease" in d or "negative" in d:
        return "decrease"
    if "no" in d:
        return "null"
    return "other"


def mc(m):
    m = (m or "").lower()
    if any(s in m for s in STRUCT):
        return "structural"
    if any(f in m for f in FUNC):
        return "functional"
    return "other"


def isn(s):
    return any(p.search(s) for p in NOISE.values())


def sp(inc, dec):
    n = inc + dec
    if n == 0:
        return 1.0
    k = min(inc, dec)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    d = json.load(open(IN, encoding="utf-8"))
    clean = [r for r in d if r.get("llm_parsed") and not isn(r["sentence"])]
    n_sent = len({r["sentence"] for r in clean})
    n_stud = len({r["pmid"] for r in clean})
    ent = defaultdict(Counter)
    sd = defaultdict(lambda: defaultdict(Counter))
    n_ent = 0
    for r in clean:
        for e in (r.get("llm_parsed") or []):
            if not isinstance(e, dict):
                continue
            reg = nr(e.get("brain_region"))
            if not reg:
                continue
            n_ent += 1
            key = (reg, mc(e.get("metric")))
            di = nd(e.get("direction"))
            ent[key][di] += 1
            if di in ("increase", "decrease"):
                sd[key][r["pmid"]][di] += 1
    print(f"clean set: {n_sent} sentences, {n_stud} studies, {n_ent} region-entries")
    print(f"{'Region':<18}{'Metric':<11}{'Sdec':>5}{'Sinc':>5}{'Stie':>5}{'study_p':>10}"
          f"{'Edec':>6}{'Einc':>6}{'entry_p':>10}")
    rows = []
    for key in sorted(ent, key=lambda k: -(ent[k]['increase'] + ent[k]['decrease'])):
        reg, m = key
        if m == "other":
            continue
        ed, ei = ent[key]['decrease'], ent[key]['increase']
        if ed + ei < 8:
            continue
        sdec = sinc = stie = 0
        for pmid, c in sd[key].items():
            if c['decrease'] > c['increase']:
                sdec += 1
            elif c['increase'] > c['decrease']:
                sinc += 1
            else:
                stie += 1
        sps, eps = sp(sinc, sdec), sp(ei, ed)
        print(f"{reg:<18}{m:<11}{sdec:>5}{sinc:>5}{stie:>5}{sps:>10.2e}{ed:>6}{ei:>6}{eps:>10.2e}")
        rows.append({"region": reg, "metric": m, "S_dec": sdec, "S_inc": sinc,
                     "S_tie": stie, "study_p": sps, "E_dec": ed, "E_inc": ei, "entry_p": eps})
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "table1_canonical.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "metric", "S_dec", "S_inc", "S_tie",
                                          "study_p", "E_dec", "E_inc", "entry_p"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUTDIR}/table1_canonical.csv")

    # r-magnitude medians (standard median) + Cohen d, for §3.5/§3.7
    import statistics
    import math
    rmag = defaultdict(list)
    for r in clean:
        for e in (r.get("llm_parsed") or []):
            if not isinstance(e, dict):
                continue
            reg = nr(e.get("brain_region"))
            if reg and (e.get("measure_type") or "").lower() == "r" \
                    and isinstance(e.get("value"), (int, float)):
                rmag[reg].append(e["value"])

    def r2d(r):
        r = max(min(r, 0.999), -0.999)
        return 2 * r / math.sqrt(1 - r * r)
    print("\nr-magnitude (standard median) -> Cohen d:")
    for reg in ["amygdala", "hippocampus", "ACC", "prefrontal cortex", "insula", "striatum", "thalamus"]:
        v = rmag[reg]
        if len(v) < 5:
            continue
        med = statistics.median(v)
        print(f"  {reg:<17} n={len(v):<4} median={med:+.3f}  d={r2d(med):+.2f}")


if __name__ == "__main__":
    main()
