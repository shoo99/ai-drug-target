#!/usr/bin/env python3
"""
Paper 3 v2 -- figures, regenerated from the SAME canonical v3 clean set used by
paper3_v2_table.py so figures, Table 1 and the text are guaranteed consistent.

Fig 1: directional balance per region x metric class (entry-level diverging
       bars; decrease left, increase right; k = contributing studies).
Fig 2: r-magnitude distribution per region (signed: negative = decrease-with-
       stress / inverse correlation).
"""
import json, re, os
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "/tmp/KBRI_sparse/paper3-fulltext-v2/output/llm_mapped_35b_v3.json"
FIG = "/home/sysoft/ai-drug-target/paper3/figures_v2"

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


def load_clean():
    d = json.load(open(IN, encoding="utf-8"))
    return [r for r in d if r.get("llm_parsed") and not isn(r["sentence"])]


def fig1(clean):
    ent = defaultdict(Counter)
    studies = defaultdict(set)
    for r in clean:
        for e in (r.get("llm_parsed") or []):
            if not isinstance(e, dict):
                continue
            reg = nr(e.get("brain_region"))
            if not reg:
                continue
            key = (reg, mc(e.get("metric")))
            ent[key][nd(e.get("direction"))] += 1
            studies[key].add(r["pmid"])
    # keep region x metric with >=8 directional entries
    keys = [k for k in ent if k[1] != "other" and (ent[k]['decrease'] + ent[k]['increase']) >= 8]
    # order: structural first (by total desc), then functional (by total desc)
    keys.sort(key=lambda k: (0 if k[1] == "structural" else 1,
                             -(ent[k]['decrease'] + ent[k]['increase'])))
    labels, dec_pct, inc_pct, colors, kstud = [], [], [], [], []
    for k in keys:
        reg, m = k
        d, i = ent[k]['decrease'], ent[k]['increase']
        tot = d + i
        labels.append(f"{reg} ({m[:6]}.)")
        dec_pct.append(-100 * d / tot)
        inc_pct.append(100 * i / tot)
        colors.append("#2C6FBB" if m == "structural" else "#E8902A")
        kstud.append(len(studies[k]))
    y = range(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(y, dec_pct, color=colors, alpha=0.95, edgecolor="white")
    ax.barh(y, inc_pct, color=colors, alpha=0.45, edgecolor="white")
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-100, 100)
    ax.set_xlabel("← decrease-dominant      % of effect-size entries      increase-dominant →",
                  fontsize=9)
    for yi, kk in zip(y, kstud):
        ax.text(96, yi, f"k={kk}", va="center", ha="right", fontsize=7.5, color="#444")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2C6FBB", label="structural"),
                       Patch(color="#E8902A", label="functional")],
              loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("Full-text effect-size directional balance (entry-level)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_fig1_directional_balance.png"), dpi=300)
    plt.close(fig)
    print("wrote v2_fig1_directional_balance.png")


def fig2(clean):
    rmag = defaultdict(list)
    for r in clean:
        for e in (r.get("llm_parsed") or []):
            if not isinstance(e, dict):
                continue
            reg = nr(e.get("brain_region"))
            if not reg:
                continue
            if (e.get("measure_type") or "").lower() == "r" and isinstance(e.get("value"), (int, float)):
                rmag[reg].append(e["value"])
    order = [reg for reg in ["amygdala", "hippocampus", "ACC", "prefrontal cortex",
                             "insula", "striatum", "thalamus"] if len(rmag.get(reg, [])) >= 5]
    data = [rmag[reg] for reg in order]
    import statistics
    labels = [f"{reg}\n(n={len(rmag[reg])}, med={statistics.median(rmag[reg]):+.2f})" for reg in order]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bp = ax.boxplot(data, vert=True, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.8)
    for xi, vals in enumerate(data, start=1):
        import random  # not seeded; only horizontal jitter for display
        xs = [xi + (hash((xi, j)) % 1000 / 1000 - 0.5) * 0.3 for j in range(len(vals))]
        ax.scatter(xs, vals, s=6, color="#08519c", alpha=0.35, zorder=3)
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("reported correlation r (signed)", fontsize=9)
    ax.set_title("Stress/severity × region correlations (as reported)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_fig2_r_magnitude.png"), dpi=300)
    plt.close(fig)
    print("wrote v2_fig2_r_magnitude.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    clean = load_clean()
    fig1(clean)
    fig2(clean)


if __name__ == "__main__":
    main()
