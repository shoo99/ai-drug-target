#!/usr/bin/env python3
"""
Paper 3 — Generate all figures from 10,001 abstractions.

Fig 1: Publication year trend + study type stacked bar
Fig 2: Top 30 gene frequency bar chart (colored by pathway)
Fig 3: Brain region × effect heatmap (metric-separated: structural vs functional)
Fig 4: BDNF temporal reversal + stress type comparison
Fig 5: Inflammation vs HPA crossover timeline
Fig 6: Research gap matrix (gene × region)
Fig 7: Consistency score dot plot (most consistent vs controversial)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict

FIG_DIR = Path(__file__).parent.parent / "paper3" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

def load_data():
    df = pd.read_parquet("data/stress/abstracts.parquet")
    def pj(v):
        if pd.isna(v): return []
        if isinstance(v, list): return v
        try: return json.loads(v)
        except: return []
    df["_genes"] = df["gene_mentions"].apply(pj)
    df["_regions"] = df["brain_regions"].apply(pj)
    df["_stress"] = df["stress_type"].apply(pj)
    df["_year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    valid = df[df["parse_error"].isna()].copy()
    return valid


def fig1_yearly_trend(df):
    """Fig 1: Publication year trend with study type stacking."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Yearly count
    year_counts = df[df["_year"] >= 2005]["_year"].value_counts().sort_index()
    ax1.bar(year_counts.index, year_counts.values, color="#1565c0", alpha=0.8, width=0.8)
    ax1.set_xlabel("Publication Year")
    ax1.set_ylabel("Number of Articles")
    ax1.set_title("A. Articles per Year (N = {:,})".format(len(df)))
    ax1.set_xlim(2004.5, 2026.5)

    # Panel B: Study type stacked
    study_year = pd.crosstab(df[df["_year"] >= 2008]["_year"], df["study_type"])
    colors = {"human": "#2196F3", "animal": "#4CAF50", "review": "#FF9800",
              "meta_analysis": "#9C27B0", "in_vitro": "#795548"}
    study_year.plot(kind="bar", stacked=True, ax=ax2, width=0.8,
                    color=[colors.get(c, "#9E9E9E") for c in study_year.columns])
    ax2.set_xlabel("Publication Year")
    ax2.set_ylabel("Number of Articles")
    ax2.set_title("B. Study Type Distribution")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_yearly_trend.png")
    fig.savefig(FIG_DIR / "fig1_yearly_trend.pdf")
    print("  Fig 1 saved")
    plt.close()


def fig2_gene_frequency(df):
    """Fig 2: Top 30 gene frequency."""
    genes = Counter()
    for gl in df["_genes"]:
        for g in gl:
            genes[g.upper().strip()] += 1

    top30 = genes.most_common(30)
    names = [g for g, _ in top30]
    counts = [c for _, c in top30]

    # Color by category
    inflam = {"IL1B", "TNF", "IL6", "NFKB1", "NLRP3", "TLR4", "IL10"}
    hpa = {"NR3C1", "CRH", "CRHR1", "FKBP5", "NR3C2", "POMC"}
    neuro = {"BDNF", "NTRK2", "CREB1", "DCX", "DLG4", "GFAP", "SYP", "GAP43"}
    glutamate = {"GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRM5"}
    apoptosis = {"BCL2", "BAX", "CASP3"}

    colors = []
    for g in names:
        if g in inflam: colors.append("#F44336")
        elif g in hpa: colors.append("#FF9800")
        elif g in neuro: colors.append("#2196F3")
        elif g in glutamate: colors.append("#9C27B0")
        elif g in apoptosis: colors.append("#795548")
        else: colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(names)), counts[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Number of Articles")
    ax.set_title("Figure 2. Top 30 Genes in Chronic Stress × Brain Literature\n"
                 f"(N = {len(df):,} articles, {len(genes):,} unique genes)")

    patches = [
        mpatches.Patch(color="#2196F3", label="Neurotrophic/Synaptic"),
        mpatches.Patch(color="#F44336", label="Inflammatory"),
        mpatches.Patch(color="#FF9800", label="HPA Axis"),
        mpatches.Patch(color="#9C27B0", label="Glutamate"),
        mpatches.Patch(color="#795548", label="Apoptosis"),
        mpatches.Patch(color="#9E9E9E", label="Other"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_gene_frequency.png")
    fig.savefig(FIG_DIR / "fig2_gene_frequency.pdf")
    print("  Fig 2 saved")
    plt.close()


def fig3_metric_separated(df):
    """Fig 3: Brain region effect by metric type (structural vs functional)."""
    struct_kw = ["volume", "gray matter", "grey matter", "cortical thickness", "surface area"]
    func_kw = ["functional connectivity", "activation", "activity", "connectivity", "resting"]

    regions_order = ["hippocampus", "amygdala", "prefrontal cortex",
                     "anterior cingulate cortex", "dentate gyrus",
                     "medial prefrontal cortex", "insula", "thalamus",
                     "orbitofrontal cortex", "basolateral amygdala",
                     "cerebellum", "nucleus accumbens"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, cat, kws, title in [
        (ax1, "Structural", struct_kw, "A. Structural (Volume/GM)"),
        (ax2, "Functional", func_kw, "B. Functional (Connectivity/Activity)")
    ]:
        data_dec = []
        data_inc = []
        labels = []
        for region in regions_order:
            dec = inc = 0
            for rl in df["_regions"]:
                for br in rl:
                    r = (br.get("region") or "").lower()
                    m = (br.get("metric") or "").lower()
                    e = (br.get("effect") or "").lower()
                    if region in r and any(k in m for k in kws):
                        if e == "decrease": dec += 1
                        elif e == "increase": inc += 1
            if dec + inc > 0:
                data_dec.append(dec)
                data_inc.append(inc)
                labels.append(region)

        y = range(len(labels))
        ax.barh(y, [-d for d in data_dec], color="#F44336", alpha=0.8, label="Decrease")
        ax.barh(y, data_inc, color="#2196F3", alpha=0.8, label="Increase")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("← Decrease | Increase →")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Figure 3. Brain Region Effects Separated by Metric Type",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_metric_separated.png")
    fig.savefig(FIG_DIR / "fig3_metric_separated.pdf")
    print("  Fig 3 saved")
    plt.close()


def fig4_bdnf_reversal(df):
    """Fig 4: BDNF temporal reversal + stress type comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Collect BDNF × hippocampus entries
    entries = []
    for _, row in df.iterrows():
        if "BDNF" not in [g.upper() for g in row["_genes"]]:
            continue
        for br in row["_regions"]:
            if "hippocampus" in (br.get("region") or "").lower():
                entries.append({
                    "year": row["_year"],
                    "effect": (br.get("effect") or "").lower(),
                    "stress": row["_stress"][0].lower() if row["_stress"] else "unknown",
                    "study": (row["study_type"] or "").lower(),
                })

    edf = pd.DataFrame(entries)

    # Panel A: Temporal trend
    years = range(2010, 2027)
    dec_pct = []
    inc_pct = []
    for y in years:
        sub = edf[(edf["year"] >= y) & (edf["year"] < y + 2)]
        total = len(sub)
        if total < 5:
            dec_pct.append(np.nan)
            inc_pct.append(np.nan)
            continue
        dec_pct.append(sum(sub["effect"] == "decrease") / total * 100)
        inc_pct.append(sum(sub["effect"] == "increase") / total * 100)

    ax1.plot(list(years), dec_pct, "o-", color="#F44336", linewidth=2, markersize=5, label="↓ Decrease")
    ax1.plot(list(years), inc_pct, "s-", color="#2196F3", linewidth=2, markersize=5, label="↑ Increase")
    ax1.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax1.axvspan(2019.5, 2026.5, alpha=0.1, color="#2196F3")
    ax1.annotate("Reversal\nzone", xy=(2022, 55), fontsize=9, ha="center", color="#1565c0")
    ax1.set_xlabel("Publication Year (2-year window)")
    ax1.set_ylabel("% of Reports")
    ax1.set_title("A. BDNF × Hippocampus: Temporal Reversal")
    ax1.legend(fontsize=9)
    ax1.set_ylim(20, 70)

    # Panel B: By stress type
    stress_data = []
    for st in ["chronic", "ptsd", "early_life", "acute"]:
        sub = edf[edf["stress"] == st]
        if len(sub) < 10: continue
        dec = sum(sub["effect"] == "decrease") / len(sub) * 100
        inc = sum(sub["effect"] == "increase") / len(sub) * 100
        stress_data.append({"stress": st, "dec": dec, "inc": inc, "n": len(sub)})

    sdf = pd.DataFrame(stress_data)
    x = range(len(sdf))
    w = 0.35
    ax2.bar([i - w/2 for i in x], sdf["dec"], w, color="#F44336", alpha=0.8, label="↓ Decrease")
    ax2.bar([i + w/2 for i in x], sdf["inc"], w, color="#2196F3", alpha=0.8, label="↑ Increase")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['stress']}\n(n={r['n']})" for _, r in sdf.iterrows()], fontsize=9)
    ax2.set_ylabel("% of Reports")
    ax2.set_title("B. BDNF × Hippocampus: By Stress Type")
    ax2.legend(fontsize=9)
    ax2.axhline(50, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Figure 4. BDNF Direction Is Not Universally Downregulated by Stress",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_bdnf_reversal.png")
    fig.savefig(FIG_DIR / "fig4_bdnf_reversal.pdf")
    print("  Fig 4 saved")
    plt.close()


def fig5_inflammation_crossover(df):
    """Fig 5: Inflammation vs HPA axis yearly crossover."""
    inflam_set = {"IL1B", "TNF", "IL6", "NFKB1", "NLRP3", "TLR4", "IL10", "CCL2"}
    hpa_set = {"NR3C1", "CRH", "CRHR1", "FKBP5", "NR3C2", "POMC"}

    years = range(2010, 2027)
    inflam_counts = []
    hpa_counts = []

    for y in years:
        sub = df[(df["_year"] >= y) & (df["_year"] < y + 1)]
        inf_n = sum(1 for gl in sub["_genes"] if set(g.upper() for g in gl) & inflam_set)
        hpa_n = sum(1 for gl in sub["_genes"] if set(g.upper() for g in gl) & hpa_set)
        inflam_counts.append(inf_n)
        hpa_counts.append(hpa_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Absolute counts
    ax1.plot(list(years), inflam_counts, "o-", color="#F44336", linewidth=2, label="Inflammation genes")
    ax1.plot(list(years), hpa_counts, "s-", color="#FF9800", linewidth=2, label="HPA axis genes")
    ax1.fill_between(list(years), inflam_counts, hpa_counts, alpha=0.1,
                     where=[i > h for i, h in zip(inflam_counts, hpa_counts)], color="#F44336")
    ax1.fill_between(list(years), inflam_counts, hpa_counts, alpha=0.1,
                     where=[i <= h for i, h in zip(inflam_counts, hpa_counts)], color="#FF9800")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Articles mentioning gene set")
    ax1.set_title("A. Absolute Counts")
    ax1.legend(fontsize=9)

    # Panel B: Ratio
    ratios = [i / h if h > 0 else 0 for i, h in zip(inflam_counts, hpa_counts)]
    colors = ["#F44336" if r > 1 else "#FF9800" for r in ratios]
    ax2.bar(list(years), ratios, color=colors, alpha=0.8)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Inflammation / HPA Ratio")
    ax2.set_title("B. Crossover Ratio (>1 = Inflammation dominant)")
    ax2.annotate("Crossover\n~2018", xy=(2018, 1.1), fontsize=10, ha="center",
                 fontweight="bold", color="#c62828")

    fig.suptitle("Figure 5. Neuroinflammation Overtakes HPA Axis in Stress-Brain Research",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_inflammation_crossover.png")
    fig.savefig(FIG_DIR / "fig5_inflammation_crossover.pdf")
    print("  Fig 5 saved")
    plt.close()


def fig6_research_gap(df):
    """Fig 6: Gene × Region research gap heatmap."""
    genes_all = Counter()
    regions_all = Counter()
    co = defaultdict(Counter)
    for _, row in df.iterrows():
        gl = [g.upper() for g in row["_genes"]]
        rl = [(br.get("region") or "").lower().strip() for br in row["_regions"]]
        for g in gl: genes_all[g] += 1
        for r in rl:
            if r: regions_all[r] += 1
        for g in gl:
            for r in rl:
                if r: co[g][r] += 1

    top_g = [g for g, _ in genes_all.most_common(12)]
    top_r = [r for r, _ in regions_all.most_common(10)]

    # Build matrix: log2(observed/expected)
    matrix = np.zeros((len(top_g), len(top_r)))
    for i, g in enumerate(top_g):
        for j, r in enumerate(top_r):
            actual = co[g].get(r, 0)
            expected = (genes_all[g] / len(df)) * (regions_all[r] / len(df)) * len(df)
            if expected > 0 and actual > 0:
                matrix[i, j] = np.log2(actual / expected)
            elif expected > 0:
                matrix[i, j] = -4  # strong gap
            else:
                matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-4, vmax=4)
    ax.set_xticks(range(len(top_r)))
    ax.set_yticks(range(len(top_g)))
    ax.set_xticklabels([r[:20] for r in top_r], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(top_g, fontsize=9)

    for i in range(len(top_g)):
        for j in range(len(top_r)):
            actual = co[top_g[i]].get(top_r[j], 0)
            ax.text(j, i, str(actual), ha="center", va="center", fontsize=7,
                    color="white" if abs(matrix[i, j]) > 2 else "black")

    plt.colorbar(im, ax=ax, label="log₂(Observed/Expected)", shrink=0.8)
    ax.set_title("Figure 6. Gene × Brain Region Co-occurrence\n"
                 "Blue = Over-studied, Red = Under-studied (Research Gap)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_research_gap.png")
    fig.savefig(FIG_DIR / "fig6_research_gap.pdf")
    print("  Fig 6 saved")
    plt.close()


def fig7_consistency(df):
    """Fig 7: Consistency score — most consistent vs controversial gene-region pairs."""
    pair_eff = defaultdict(Counter)
    for _, row in df.iterrows():
        gl = [g.upper() for g in row["_genes"]]
        for br in row["_regions"]:
            r = (br.get("region") or "").lower().strip()
            e = (br.get("effect") or "").lower().strip()
            if not r or e not in ("decrease", "increase"): continue
            for g in gl:
                pair_eff[(g, r)][e] += 1

    pairs = []
    for (g, r), eff in pair_eff.items():
        total = sum(eff.values())
        if total < 15: continue
        dec = eff.get("decrease", 0)
        inc = eff.get("increase", 0)
        dom = max(dec, inc)
        pairs.append({"gene": g, "region": r, "total": total,
                       "dec": dec, "inc": inc, "consistency": dom / total * 100,
                       "direction": "↓" if dec > inc else "↑"})

    pairs.sort(key=lambda x: x["consistency"])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all pairs
    for p in pairs:
        color = "#2196F3" if p["consistency"] > 65 else ("#F44336" if p["consistency"] < 55 else "#9E9E9E")
        ax.scatter(p["consistency"], p["total"], s=40, c=color, alpha=0.6, edgecolors="white", linewidth=0.3)

    # Annotate extremes
    for p in sorted(pairs, key=lambda x: x["consistency"])[:5]:
        ax.annotate(f"{p['gene']}×{p['region'][:8]}", xy=(p["consistency"], p["total"]),
                    fontsize=7, color="#c62828", ha="center", va="bottom")
    for p in sorted(pairs, key=lambda x: -x["consistency"])[:5]:
        ax.annotate(f"{p['gene']}×{p['region'][:8]}", xy=(p["consistency"], p["total"]),
                    fontsize=7, color="#0d47a1", ha="center", va="bottom")

    ax.axvline(50, color="red", linestyle="--", alpha=0.3, label="Random (50%)")
    ax.axvline(65, color="blue", linestyle=":", alpha=0.3, label="Consistent (65%)")
    ax.set_xlabel("Consistency (% dominant direction)")
    ax.set_ylabel("Number of Reports (N)")
    ax.set_title("Figure 7. Cross-Study Consistency of Gene × Brain Region Effects\n"
                 "(Blue = consistent, Red = controversial, N ≥ 15)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_consistency.png")
    fig.savefig(FIG_DIR / "fig7_consistency.pdf")
    print("  Fig 7 saved")
    plt.close()


def main():
    print("=" * 70)
    print("  PAPER 3 — FIGURE GENERATION (10,001 articles)")
    print("=" * 70)

    df = load_data()
    print(f"  Loaded: {len(df)} valid extractions\n")

    fig1_yearly_trend(df)
    fig2_gene_frequency(df)
    fig3_metric_separated(df)
    fig4_bdnf_reversal(df)
    fig5_inflammation_crossover(df)
    fig6_research_gap(df)
    fig7_consistency(df)

    print(f"\n  All figures saved to: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
