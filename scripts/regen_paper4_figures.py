#!/usr/bin/env python3
"""Regenerate Paper 4 figures 4, 6, S1 to reflect colocalization results (v18).

Fixes review #1/#10/#11: figures previously showed pre-demotion state (DRD2 as
top/validated). New versions overlay colocalization status from result files so
the figures match the manuscript's DRD2-demotion / SLC12A5-nomination conclusion.

Data sources (all in paper4/results/):
  - mr_top_genes.tsv           : TWAS gene, zscore, min_pvalue (discovery)
  - coloc/coloc_all_summary.csv : gene, PP3, PP4 (208 genes with BrainMeta probe)
  - directional_drug_candidates.tsv : gene, zscore, drug, interaction_type, approved
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

RES = Path("/home/sysoft/ai-drug-target/paper4/results")
FIG = Path("/home/sysoft/ai-drug-target/paper4/figures")

# ---- colocalization status, pre-specified PP4 bins (review #3) ----
# >=0.8 colocalized ; 0.5-0.8 suggestive ; <0.5 not colocalized
COL_COLOC = "#2ca02c"      # green
COL_SUGG  = "#ff7f0e"      # orange
COL_NOT   = "#9e9e9e"      # grey
COL_NOPROBE = "#cfd8dc"    # light grey (no BrainMeta probe → untested)

def load_coloc():
    pp4 = {}
    with open(RES / "coloc/coloc_all_summary.csv") as f:
        for r in csv.DictReader(f):
            pp4[r["gene"]] = float(r["PP4"])
    return pp4

def coloc_status(gene, pp4):
    if gene not in pp4:
        return "untested", COL_NOPROBE
    v = pp4[gene]
    if v >= 0.8:
        return "colocalized", COL_COLOC
    if v >= 0.5:
        return "suggestive", COL_SUGG
    return "not colocalized", COL_NOT

def load_twas():
    rows = []
    with open(RES / "mr_top_genes.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                rows.append((r["gene"], float(r["zscore"]), float(r["min_pvalue"])))
            except (ValueError, KeyError):
                continue
    return rows

# =====================================================================
# Figure 4 — TWAS Z-score forest (discovery), colored by colocalization
# =====================================================================
def fig4():
    pp4 = load_coloc()
    rows = load_twas()
    rows.sort(key=lambda x: x[2])           # by TWAS p-value, strongest first
    top = rows[:25]
    top = top[::-1]                          # smallest p at top of plot

    genes = [g for g, _, _ in top]
    zs    = [z for _, z, _ in top]
    ps    = [p for _, _, p in top]
    colors = [coloc_status(g, pp4)[1] for g in genes]

    fig, ax = plt.subplots(figsize=(11, 9))
    y = range(len(genes))
    ax.axvline(0, color="#555", lw=1)
    ax.scatter(zs, y, c=colors, s=130, edgecolor="black", linewidth=0.6, zorder=3)

    for i, (g, z, p) in enumerate(zip(genes, zs, ps)):
        # Place labels toward the zero line (inward) so they never collide with
        # the y-axis tick labels at the plot edges. Negative-Z markers sit on the
        # left, so their label goes to the right; positive-Z markers vice-versa.
        if z >= 0:
            side, ha = -8, "right"
        else:
            side, ha = 8, "left"
        lbl = f"{g}  P={p:.1e}"
        if g == "DRD2":
            lbl = f"{g}  P={p:.1e} (coloc-rejected)"
        ax.annotate(lbl, (z, i), xytext=(side, 0), textcoords="offset points",
                    va="center", ha=ha, fontsize=8.5,
                    fontweight=("bold" if g in ("DRD2", "SLC12A5") else "normal"))

    ax.set_yticks(list(y)); ax.set_yticklabels(genes, fontsize=9)
    ax.set_xlabel("S-PrediXcan TWAS Z-score  (negative = lower-expression risk)", fontsize=11)
    ax.set_title("Brain-tissue TWAS Z-scores (discovery, pre-confirmation)\n"
                 "color = colocalization status against BrainMeta cis-eQTL", fontsize=12)
    xmax = max(abs(min(zs)), abs(max(zs))) * 1.45
    ax.set_xlim(-xmax, xmax)
    ax.margins(y=0.02)

    legend = [
        mpatches.Patch(color=COL_COLOC, label="Colocalized (PP4 ≥ 0.8)"),
        mpatches.Patch(color=COL_SUGG,  label="Suggestive (0.5 ≤ PP4 < 0.8)"),
        mpatches.Patch(color=COL_NOT,   label="Not colocalized (PP4 < 0.5)"),
        mpatches.Patch(color=COL_NOPROBE, label="No BrainMeta probe (untested)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    out = FIG / "fig4_twas_forest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Fig4 ->", out)

# =====================================================================
# Figure 6 — two panels: (A) raw direction-aware, (B) coloc-stratified
# =====================================================================
def load_drugs():
    rows = []
    with open(RES / "directional_drug_candidates.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows

def fig6():
    pp4 = load_coloc()
    drows = load_drugs()
    approved = [r for r in drows if str(r["approved"]).strip().lower() == "true"]

    # Panel A: approved drugs, colored red if target is DRD2 (pre-demotion view)
    seenA, A = set(), []
    for r in approved:
        d = r["drug"]
        if d in seenA:
            continue
        seenA.add(d)
        A.append((d, r["gene"], float(r["zscore"]), r["interaction_type"]))
    A = A[:20][::-1]

    # Panel B: same drugs but tiered by target colocalization status
    def tier(gene):
        st, _ = coloc_status(gene, pp4)
        if st == "colocalized":
            return "Prioritized (colocalized target)", COL_COLOC
        if st == "suggestive":
            return "Provisional (suggestive target)", COL_SUGG
        return "Exploratory (non-colocalized target, incl. DRD2)", COL_NOT

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 8))

    # ---- Panel A ----
    namesA = [f"{d}" for d, *_ in A]
    valsA  = [1] * len(A)
    colsA  = ["#d62728" if g == "DRD2" else "#1f77b4" for _, g, _, _ in A]
    axA.barh(range(len(A)), valsA, color=colsA, edgecolor="black", linewidth=0.4)
    for i, (d, g, z, it) in enumerate(A):
        axA.text(1.02, i, f"{g}({it},Z={z:+.1f})", va="center", fontsize=7.5)
    axA.set_yticks(range(len(A))); axA.set_yticklabels(namesA, fontsize=8)
    axA.set_xlim(0, 3.8); axA.set_xlabel("direction-consistent target (count)", fontsize=10)
    axA.set_title("(A) Direction-aware approved candidates\n(raw, pre-confirmation)", fontsize=11)
    axA.legend(handles=[mpatches.Patch(color="#d62728", label="DRD2-targeting (dopaminergic)"),
                        mpatches.Patch(color="#1f77b4", label="other direction-consistent")],
               loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8, frameon=False)

    # ---- Panel B: same drugs, recolored by colocalization tier ----
    B = A
    namesB = [d for d, *_ in B]
    colsB  = [tier(g)[1] for _, g, _, _ in B]
    axB.barh(range(len(B)), [1]*len(B), color=colsB, edgecolor="black", linewidth=0.4)
    for i, (d, g, z, it) in enumerate(B):
        tag = ""
        if g == "DRD2":
            tag = "  ← demoted"
        axB.text(1.02, i, f"{g} [{coloc_status(g, pp4)[0]}]{tag}", va="center", fontsize=7.5)
    axB.set_yticks(range(len(B))); axB.set_yticklabels(namesB, fontsize=8)
    axB.set_xlim(0, 3.8); axB.set_xlabel("direction-consistent target (count)", fontsize=10)
    axB.set_title("(B) Colocalization-stratified prioritization\n(DRD2 cluster demoted to exploratory tier)", fontsize=11)
    axB.legend(handles=[mpatches.Patch(color=COL_COLOC, label="Prioritized (colocalized)"),
                        mpatches.Patch(color=COL_SUGG, label="Provisional (suggestive)"),
                        mpatches.Patch(color=COL_NOT, label="Exploratory (non-coloc, incl. DRD2)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8, frameon=False)

    fig.suptitle("Direction-aware MDD drug repurposing: raw candidates vs colocalization-stratified prioritization",
                 fontsize=12.5, y=1.00)
    fig.tight_layout()
    out = FIG / "fig6_directional_drugs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Fig6 ->", out)

# =====================================================================
# Figure S1 — drug–gene interaction network (title fixed, DRD2 labeled)
# =====================================================================
def figS1():
    import networkx as nx
    pp4 = load_coloc()
    drows = load_drugs()
    approved = [r for r in drows if str(r["approved"]).strip().lower() == "true"]

    # build bipartite: drug -- gene
    G = nx.Graph()
    genes_used = {}
    drug_seen = []
    for r in approved:
        d, g = r["drug"], r["gene"]
        if d not in drug_seen:
            drug_seen.append(d)
        G.add_node(d, kind="drug")
        G.add_node(g, kind="gene")
        G.add_edge(d, g)
        genes_used[g] = coloc_status(g, pp4)

    # keep top-20 drugs by degree for readability
    drug_nodes = [n for n, a in G.nodes(data=True) if a["kind"] == "drug"]
    gene_nodes = [n for n, a in G.nodes(data=True) if a["kind"] == "gene"]

    pos = {}
    for i, g in enumerate(sorted(gene_nodes)):
        pos[g] = (1.0, i * (len(drug_nodes) / max(len(gene_nodes), 1)))
    for i, d in enumerate(sorted(drug_nodes)):
        pos[d] = (0.0, i)

    fig, ax = plt.subplots(figsize=(15, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=0.6)

    # drugs
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color="#d62728",
                           node_size=120, ax=ax, label="approved drug")
    # genes colored by coloc status
    gcolors = [coloc_status(g, pp4)[1] for g in gene_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color=gcolors,
                           node_size=320, edgecolors="black", linewidths=0.6, ax=ax)

    labels = {}
    for g in gene_nodes:
        st = coloc_status(g, pp4)[0]
        labels[g] = f"{g}\n[{st}]" if g in ("DRD2", "SLC12A5", "FURIN", "DCC", "GPX1", "NEGR1") else g
    nx.draw_networkx_labels(G, pos, labels={g: labels[g] for g in gene_nodes},
                            font_size=7.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={d: d for d in drug_nodes}, font_size=6, ax=ax,
                            horizontalalignment="right")

    ax.set_title("DGIdb gene–drug interactions for TWAS-prioritized genes\n"
                 "(gene color = colocalization status; DRD2 = coloc-rejected, drugs exploratory)",
                 fontsize=12)
    # widen left margin so right-aligned drug labels (anchored at x=0) are not clipped
    ax.set_xlim(-1.05, 1.35)
    ax.axis("off")
    legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#d62728', markersize=9, label='approved drug'),
        mpatches.Patch(color=COL_COLOC, label="gene: colocalized"),
        mpatches.Patch(color=COL_SUGG,  label="gene: suggestive"),
        mpatches.Patch(color=COL_NOT,   label="gene: not colocalized (incl. DRD2)"),
        mpatches.Patch(color=COL_NOPROBE, label="gene: no probe / untested"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    out = FIG / "figS1_drug_network.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("FigS1 ->", out)

if __name__ == "__main__":
    fig4()
    fig6()
    figS1()
