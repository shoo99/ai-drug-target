#!/usr/bin/env python3
"""
crosscheck_validation.py
========================
Build a BLINDED, hard-case-oversampled validation set for the LLM *direction*
extraction, and score it against the model's labels.

WHY THIS EXISTS
---------------
The manuscript's "silver" validation derived its reference label from the same
surface lexical cues (reduced/lower/...) the model keys on, and *excluded*
every sentence without an unambiguous cue. So it (a) only measured the easy
subset and (b) was near-tautological there. The dangerous failure modes for a
directional synthesis -- negation, "reduced deactivation", and
correlation-VALENCE mapping -- were never tested.

This tool oversamples exactly those strata and asks an INDEPENDENT RE-LABELLER
for a reference label, keeping the model's prediction hidden until scoring.

IMPORTANT -- who the re-labeller was in this study:
  In Paper 3 v2.2 the 50-sentence cross-check reference labels were produced by
  an INDEPENDENT LLM (a second model, blind to the original mapping), NOT by a
  human. This is therefore an AI-vs-AI cross-check and is reported as inter-model
  *agreement*, not as a human gold standard. The ONLY human-labelled step was the
  author re-reading the five model-vs-cross-check disagreements against their
  source sentences (see manuscript section 3.6 / 2.6.1). The column names below
  read "human_*" only because this harness was written generically to accept any
  independent rater; here that rater was a second LLM. Read "human_*" as
  "independent re-labeller".

INPUT
-----
A file of mapped entries (one row = one extracted number), JSONL or CSV.
Expected fields (rename in CONFIG if yours differ):
  sentence, brain_region, metric, direction, value, measure_type, pmid
`direction` may be: decrease | increase | positive_correlation |
negative_correlation | null  (the model's raw output).

WORKFLOW
--------
  # 1) sample  -> writes annotation_sheet.csv (blind) + .answer_key.csv (hidden)
  python crosscheck_validation.py sample --records mapped_entries.jsonl --n 50

  # 2) an independent re-labeller (a second LLM in this study; could be a human)
  #    fills the `human_direction` column of annotation_sheet.csv with one of:
  #    decrease | increase | null | unclear  (read "human_*" as "re-labeller_*")
  #    (optional: a 2nd rater fills human_direction_2 for a kappa)

  # 3) score
  python crosscheck_validation.py score --sheet annotation_sheet.csv \
                                        --key .answer_key.csv

  # self-test on synthetic data (no real input needed)
  python crosscheck_validation.py demo
"""

import argparse, csv, json, os, random, re, sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------- CONFIG
COL = {  # rename the RHS to match your records file
    "sentence":     "sentence",
    "region":       "brain_region",
    "metric":       "metric",
    "direction":    "direction",
    "value":        "value",
    "measure_type": "measure_type",
    "study":        "pmid",
}

# how many items per stratum (priority order; totals are capped at --n)
STRATA_TARGET = {
    "negation":      14,   # most dangerous
    "deactivation":   8,
    "correlation":   16,   # valence mapping
    "no_cue":         8,   # silver-excluded, non-correlation
    "easy":           4,   # anchors
}

SEED = 20260603

# how the model's raw direction reduces to a sign for the sign test.
# negative_correlation with a SEVERITY/STRESS correlate -> metric DECREASES as
# severity rises -> "decrease". This is the assumption the re-labeller is checking.
REDUCE = {
    "decrease": "decrease", "increase": "increase", "null": "null",
    "negative_correlation": "decrease", "positive_correlation": "increase",
}
VALID_HUMAN = {"decrease", "increase", "null"}   # "unclear" rows are dropped at scoring

# ----------------------------------------------------------------------------- regexes
RE_DEC_CUE = re.compile(r"\b(reduc|decreas|smaller|lower|diminish|atroph|shrink|"
                        r"attenuat|loss|deficit|lesser|declin)\w*", re.I)
RE_INC_CUE = re.compile(r"\b(increas|greater|higher|larger|elevat|enhanc|expand|"
                        r"enlarg|augment|rise|risen|hyperactiv)\w*", re.I)
RE_NEG     = re.compile(r"\b(no|not|nor|never|fail(?:ed|s|ure)?|absen\w+|without|"
                        r"unchanged|did ?n[o']t|was ?n[o']t|were ?n[o']t|"
                        r"non-?signific\w+|no significant|no difference|"
                        r"no association|did not differ|no longer)\b", re.I)
RE_DEACT   = re.compile(r"\b(de-?activat\w+|hypo-?activat\w+|deactivation|"
                        r"reduced (?:de-?activation|suppression))\b", re.I)


def strata_of(rec):
    """Assign one PRIMARY stratum (hardest first) + return all flags."""
    s = rec["_sentence"]
    is_corr = (rec["_measure"] == "r") or rec["_direction"] in (
        "positive_correlation", "negative_correlation")
    has_neg   = bool(RE_NEG.search(s))
    has_deact = bool(RE_DEACT.search(s))
    has_cue   = bool(RE_DEC_CUE.search(s) or RE_INC_CUE.search(s))
    flags = {"negation": has_neg, "deactivation": has_deact,
             "correlation": is_corr, "no_cue": not has_cue, "has_cue": has_cue}
    if has_neg:          primary = "negation"
    elif has_deact:      primary = "deactivation"
    elif is_corr:        primary = "correlation"
    elif not has_cue:    primary = "no_cue"
    else:                primary = "easy"
    return primary, flags

# ----------------------------------------------------------------------------- io
def load_records(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl") or path.endswith(".json"):
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            rows = list(csv.DictReader(fh))
    out = []
    for i, r in enumerate(rows):
        sent = (r.get(COL["sentence"]) or "").strip()
        if not sent:
            continue
        out.append({
            "_id":        r.get("entry_id") or f"E{i:05d}",
            "_sentence":  sent,
            "_region":    (r.get(COL["region"])    or "").strip(),
            "_metric":    (r.get(COL["metric"])    or "").strip(),
            "_direction": (r.get(COL["direction"]) or "").strip().lower(),
            "_value":     r.get(COL["value"], ""),
            "_measure":   (r.get(COL["measure_type"]) or "").strip().lower(),
            "_study":     r.get(COL["study"], ""),
        })
    return out

# ----------------------------------------------------------------------------- sample
def do_sample(records, n, sheet_path, key_path):
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for rec in records:
        primary, flags = strata_of(rec)
        rec["_strata"] = primary
        rec["_flags"] = flags
        buckets[primary].append(rec)

    print("available by stratum:",
          {k: len(v) for k, v in sorted(buckets.items())})

    chosen, seen = [], set()
    # 1) fill each stratum up to its target (scaled so the sum ~= n)
    scale = n / max(1, sum(STRATA_TARGET.values()))
    for stratum, target in STRATA_TARGET.items():
        pool = buckets.get(stratum, [])
        rng.shuffle(pool)
        take = min(len(pool), max(1, round(target * scale)))
        for rec in pool[:take]:
            if rec["_id"] not in seen:
                chosen.append(rec); seen.add(rec["_id"])
    # 2) top up to n from the hardest remaining pools
    if len(chosen) < n:
        rest = [r for st in STRATA_TARGET for r in buckets.get(st, [])
                if r["_id"] not in seen]
        rng.shuffle(rest)
        for rec in rest[: n - len(chosen)]:
            chosen.append(rec); seen.add(rec["_id"])
    # 3) trim if over
    chosen = chosen[:n]
    rng.shuffle(chosen)   # randomize sheet order (blinding)

    # write blind annotation sheet (NO model fields shown)
    with open(sheet_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "sentence", "detected_value", "measure_type",
                    "human_region", "human_metric", "human_direction",
                    "human_direction_2", "note"])
        for r in chosen:
            w.writerow([r["_id"], r["_sentence"], r["_value"], r["_measure"],
                        "", "", "", "", ""])

    # write hidden answer key (model predictions + strata flags)
    with open(key_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "model_region", "model_metric",
                    "model_direction_raw", "model_direction_reduced",
                    "model_value", "strata", "negation", "deactivation",
                    "correlation", "no_cue", "study"])
        for r in chosen:
            f = r["_flags"]
            w.writerow([r["_id"], r["_region"], r["_metric"], r["_direction"],
                        REDUCE.get(r["_direction"], r["_direction"]),
                        r["_value"], r["_strata"],
                        int(f["negation"]), int(f["deactivation"]),
                        int(f["correlation"]), int(f["no_cue"]), r["_study"]])

    counts = Counter(r["_strata"] for r in chosen)
    print(f"\nwrote {len(chosen)} items")
    print("  blind sheet :", sheet_path)
    print("  answer key  :", key_path, "(keep hidden from the annotator)")
    print("  strata mix  :", dict(counts))

# ----------------------------------------------------------------------------- score
def _kappa(a, b):
    """Cohen's kappa on two equal-length label lists."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def do_score(sheet_path, key_path):
    key = {row["item_id"]: row for row in csv.DictReader(open(key_path, encoding="utf-8"))}
    sheet = list(csv.DictReader(open(sheet_path, encoding="utf-8")))

    rows = []          # scored (human label present & valid)
    skipped = 0
    for s in sheet:
        hid = s["item_id"]
        h = (s.get("human_direction") or "").strip().lower()
        if h not in VALID_HUMAN:        # blank / "unclear" -> drop
            skipped += 1
            continue
        k = key.get(hid)
        if not k:
            continue
        rows.append({
            "id": hid,
            "model": (k["model_direction_reduced"] or "").strip().lower(),
            "human": h,
            "strata": k["strata"],
            "correlation": k["correlation"] == "1",
            "human2": (s.get("human_direction_2") or "").strip().lower(),
            "model_region": (k["model_region"] or "").strip().lower(),
            "human_region": (s.get("human_region") or "").strip().lower(),
        })

    if not rows:
        print("No scorable rows. Did the annotator fill `human_direction`?")
        return

    def acc(subset):
        if not subset:
            return (0, 0, float("nan"))
        c = sum(r["model"] == r["human"] for r in subset)
        return (c, len(subset), c / len(subset))

    print(f"\n=== DIRECTION ACCURACY (n scored = {len(rows)}, "
          f"dropped blank/unclear = {skipped}) ===")
    c, n, a = acc(rows)
    print(f"  OVERALL                : {c}/{n} = {a:6.1%}")
    cc, cn, ca = acc([r for r in rows if r["correlation"]])
    print(f"  correlation-only       : {cc}/{cn} = {ca:6.1%}   <- valence test")
    nc, nn, na = acc([r for r in rows if not r["correlation"]])
    print(f"  non-correlation        : {nc}/{nn} = {na:6.1%}")

    print("\n  by stratum:")
    for st in list(STRATA_TARGET) :
        sc, sn, sa = acc([r for r in rows if r["strata"] == st])
        if sn:
            print(f"    {st:13s}: {sc}/{sn} = {sa:6.1%}")

    # confusion matrix model(rows) x human(cols)
    cats = ["decrease", "increase", "null"]
    cm = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cm[r["model"]][r["human"]] += 1
    print("\n  confusion  model\\human :", "  ".join(f"{c:>9s}" for c in cats))
    for m in cats:
        print(f"    {m:21s}:", "  ".join(f"{cm[m][h]:9d}" for h in cats))

    # directional bias: are errors one-sided?
    fd = sum(1 for r in rows if r["model"] == "decrease" and r["human"] == "increase")
    fi = sum(1 for r in rows if r["model"] == "increase" and r["human"] == "decrease")
    print(f"\n  false 'decrease' (model dec, human inc): {fd}")
    print(f"  false 'increase' (model inc, human dec): {fi}")
    print("  -> errors look", "BALANCED" if abs(fd - fi) <= 2 else "SKEWED (check!)")

    # optional region accuracy
    reg = [r for r in rows if r["human_region"]]
    if reg:
        rc = sum(r["model_region"] == r["human_region"] for r in reg)
        print(f"\n  region accuracy        : {rc}/{len(reg)} = {rc/len(reg):.1%}")

    # optional inter-annotator kappa
    both = [r for r in rows if r["human2"] in VALID_HUMAN]
    if both:
        k = _kappa([r["human"] for r in both], [r["human2"] for r in both])
        print(f"\n  inter-annotator kappa  : {k:.3f}  (n={len(both)})")

    out = "scored_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "strata", "correlation",
                    "model_reduced", "human", "correct"])
        for r in rows:
            w.writerow([r["id"], r["strata"], int(r["correlation"]),
                        r["model"], r["human"], int(r["model"] == r["human"])])
    print(f"\nwrote per-item results -> {out}")
    print("\nPaste into the manuscript (§3.6 / Abstract):")
    print(f"  hard-subset direction accuracy = {a:.0%} ({c}/{n}); "
          f"correlation-valence accuracy = {ca:.0%}; errors "
          f"{'symmetric' if abs(fd-fi)<=2 else 'skewed'}.")

# ----------------------------------------------------------------------------- demo
def do_demo():
    """Synthetic self-test. NOT real data -- only proves the pipeline runs."""
    print(">>> SYNTHETIC DEMO (fabricated sentences; not study data) <<<\n")
    synth = [
        # sentence, model_direction, measure, region, metric, true_reduced
        ("Patients showed reduced hippocampal volume vs controls (d=-0.5).",
         "decrease", "d", "hippocampus", "volume", "decrease"),
        ("Amygdala volume was increased in the high-stress group (d=0.4).",
         "increase", "d", "amygdala", "volume", "increase"),
        # negation traps
        ("There was no significant reduction in hippocampal volume (p=.30).",
         "decrease", "d", "hippocampus", "volume", "null"),     # model WRONG
        ("Patients did not show smaller amygdala volume (p=.6).",
         "null", "d", "amygdala", "volume", "null"),
        ("Volume was not increased in controls relative to patients.",
         "increase", "d", "hippocampus", "volume", "increase"), # ambiguous-ish
        # deactivation traps
        ("PTSD patients showed reduced deactivation of the vmPFC (d=0.3).",
         "decrease", "d", "prefrontal cortex", "activation", "increase"), # WRONG
        ("Hypoactivation of the insula was observed in patients (d=-0.2).",
         "decrease", "d", "insula", "activation", "decrease"),
        # correlation valence traps
        ("Amygdala activation correlated positively with cortisol (r=0.38).",
         "positive_correlation", "r", "amygdala", "activation", "increase"),
        ("Hippocampal volume correlated negatively with symptom severity (r=-0.31).",
         "negative_correlation", "r", "hippocampus", "volume", "decrease"),
        ("Hippocampal volume correlated positively with resilience score (r=0.29).",
         "positive_correlation", "r", "hippocampus", "volume", "decrease"), # valence flip -> WRONG
        ("ACC thickness correlated negatively with time-since-trauma (r=-0.25).",
         "negative_correlation", "r", "ACC", "thickness", "increase"),      # valence flip -> WRONG
        ("Striatal volume showed a positive correlation with PTSD severity (r=0.2).",
         "positive_correlation", "r", "striatum", "volume", "increase"),
        # no-cue
        ("The between-group difference in thalamic volume reached d=-0.18.",
         "decrease", "d", "thalamus", "volume", "decrease"),
    ] * 4  # replicate to give the sampler enough to choose from

    recs = "demo_records.jsonl"
    with open(recs, "w", encoding="utf-8") as fh:
        for i, (sent, d, m, reg, met, truth) in enumerate(synth):
            fh.write(json.dumps({
                "entry_id": f"E{i:05d}", "sentence": sent, "direction": d,
                "measure_type": m, "brain_region": reg, "metric": met,
                "value": 0.0, "pmid": f"PMID{i%7}", "_truth": truth,
            }) + "\n")

    records = load_records(recs)
    sheet, key = "annotation_sheet.csv", ".answer_key.csv"
    do_sample(records, n=50, sheet_path=sheet, key_path=key)

    # simulate a PERFECT human annotator by reading _truth back from demo file
    truth = {}
    for line in open(recs, encoding="utf-8"):
        o = json.loads(line); truth[o["entry_id"]] = o["_truth"]
    srows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    for s in srows:
        s["human_direction"] = truth.get(s["item_id"], "unclear")
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=srows[0].keys()); w.writeheader()
        w.writerows(srows)

    do_score(sheet, key)
    print("\n(demo files: demo_records.jsonl, annotation_sheet.csv, "
          ".answer_key.csv, scored_results.csv)")

# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="build blind sheet + hidden key")
    p.add_argument("--records", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--sheet", default="annotation_sheet.csv")
    p.add_argument("--key", default=".answer_key.csv")

    q = sub.add_parser("score", help="score a filled sheet against the key")
    q.add_argument("--sheet", default="annotation_sheet.csv")
    q.add_argument("--key", default=".answer_key.csv")

    sub.add_parser("demo", help="synthetic self-test")

    a = ap.parse_args()
    if a.cmd == "sample":
        do_sample(load_records(a.records), a.n, a.sheet, a.key)
    elif a.cmd == "score":
        do_score(a.sheet, a.key)
    elif a.cmd == "demo":
        do_demo()


if __name__ == "__main__":
    main()
