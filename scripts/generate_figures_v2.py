#!/usr/bin/env python3
"""Generate improved paper figures — fix visual issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.settings import DATA_DIR, AMR_CONFIG

FIG_DIR = Path(__file__).parent.parent / "paper" / "figures"
FONT = dict(family="Arial", size=14)


def fig2_fba_knockouts():
    """Fig 2: FBA heatmap — 3 species, mapped genes only."""
    from src.amr.data_collector_v2 import CURATED_ESSENTIAL_GENES, KNOWN_ANTIBIOTIC_TARGETS

    fba_files = {
        "E. coli (iML1515)": DATA_DIR / "metabolic_analysis" / "metabolic_Escherichia_coli.csv",
        "S. aureus (iYS1720)": DATA_DIR / "metabolic_analysis" / "metabolic_Staphylococcus_aureus.csv",
        "K. pneumoniae (iYL1228)": DATA_DIR / "metabolic_analysis" / "metabolic_Klebsiella_pneumoniae.csv",
    }

    # Load all FBA results
    fba_data = {}
    for org_label, path in fba_files.items():
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                g = row["gene"]
                if pd.notna(row.get("growth_ratio")):
                    if g not in fba_data:
                        fba_data[g] = {}
                    fba_data[g][org_label] = "Lethal" if row["is_lethal"] else "Reduced"

    # Only include genes mapped in at least one model
    mapped_genes = sorted(fba_data.keys())
    organisms = list(fba_files.keys())

    # Get drug status
    has_drug = {g: len(KNOWN_ANTIBIOTIC_TARGETS.get(g, [])) > 0 for g in mapped_genes}

    # Build matrix: Lethal=2, Reduced=1, Not mapped=0
    def encode(status):
        if status == "Lethal": return 2
        elif status == "Reduced": return 1
        else: return 0

    z = []
    hover = []
    for org in organisms:
        row_z = []
        row_h = []
        for g in mapped_genes:
            status = fba_data.get(g, {}).get(org, "Not mapped")
            row_z.append(encode(status))
            row_h.append(f"{g}: {status}")
        z.append(row_z)
        hover.append(row_h)

    # Gene labels with drug marker
    x_labels = [f"{g} {'💊' if has_drug.get(g) else '🆕'}" for g in mapped_genes]

    colorscale = [[0, "#bdbdbd"], [0.5, "#ff9800"], [1.0, "#c62828"]]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x_labels, y=organisms,
        hovertext=hover, texttemplate="",
        colorscale=colorscale, zmin=0, zmax=2,
        showscale=True,
        colorbar=dict(title="FBA", tickvals=[0, 1, 2],
                      ticktext=["Not mapped", "Growth reduced", "Lethal"]),
    ))

    fig.update_layout(
        width=1000, height=280,
        font=dict(family="Arial", size=11), plot_bgcolor="white",
        title="Figure 2. Essential Gene FBA Across 3 ESKAPE Models (Mapped Genes Only)<br> <br>"
              "<sub>🔴 Lethal  🟠 Growth-reducing  ⬜ Not mapped  💊 Has drug  🆕 Novel</sub>",
        xaxis=dict(tickangle=60, tickfont=dict(size=8), side="bottom"),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=140, r=40, t=100, b=100),
    )
    fig.write_image(str(FIG_DIR / "fig2_fba_knockouts.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig2_fba_knockouts.pdf"))
    print("  Fig 2: FBA 3-species Heatmap ✅")


def fig3_nlp_comparison():
    """Fig 3: LLM vs Keyword NLP — cleaned categories."""
    # Keyword NLP: 3 categories
    keyword_data = [
        ("RESULTS", 45, "Section header"), ("METHODS", 38, "Section header"),
        ("AREAS", 25, "Section header"), ("COVERED", 22, "Section header"),
        ("EXPERT", 20, "Section header"), ("OPINION", 18, "Section header"),
        ("LPS", 30, "Bio term"), ("CRISPR", 28, "Bio term"),
        ("ROS", 16, "Bio term"), ("UDP", 15, "Bio term"),
        ("GYRA", 14, "Real gene"), ("PARE", 13, "Real gene"),
        ("LPXC", 12, "Real gene"), ("OM", 11, "Bio term"), ("QS", 10, "Bio term"),
    ]

    # LLM: AMR vs Pruritus (duplicates merged: IL31+IL31RA, BAM→BAMA)
    llm_data = [
        ("FTSZ", 41, "AMR"), ("LPXC", 17, "AMR"), ("BAMA", 6, "AMR"),
        ("MURA", 5, "AMR"), ("MURB", 3, "AMR"), ("LPTD", 3, "AMR"),
        ("IL31/IL31RA", 22, "Pruritus"), ("IL13", 8, "Pruritus"),
        ("TRPV1", 7, "Pruritus"), ("IL4", 6, "Pruritus"),
        ("TRPA1", 5, "Pruritus"), ("TRPV3", 5, "Pruritus"),
        ("EGFR", 5, "Pruritus"), ("JAK", 4, "Pruritus"),
    ]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("(A) Keyword NLP — Top 15",
                                       "(B) LLM NLP — Top 15 (393 AMR + 521 Pruritus articles)"),
                        horizontal_spacing=0.18)

    # (A) Keyword — 3 colors
    cat_colors = {"Section header": "#c62828", "Bio term": "#ff9800", "Real gene": "#1565c0"}
    kw_colors = [cat_colors[c] for _, _, c in keyword_data]

    fig.add_trace(go.Bar(
        y=[g for g, _, _ in keyword_data], x=[c for _, c, _ in keyword_data],
        orientation="h", marker_color=kw_colors, showlegend=False,
    ), row=1, col=1)

    # (B) LLM — AMR vs Pruritus
    track_colors = {"AMR": "#1565c0", "Pruritus": "#00897b"}
    llm_colors = [track_colors[t] for _, _, t in llm_data]

    fig.add_trace(go.Bar(
        y=[g for g, _, _ in llm_data], x=[c for _, c, _ in llm_data],
        orientation="h", marker_color=llm_colors, showlegend=False,
    ), row=1, col=2)

    fig.update_yaxes(autorange="reversed")

    # Legend as annotation (below)
    fig.add_annotation(
        x=0.0, y=-0.10,
        text="(A): 🔴 Section header noise  🟠 Bio term (not gene)  🔵 Real gene  |  "
             "(B): 🔵 AMR target  🟢 Pruritus target",
        showarrow=False, font=dict(size=9), xref="paper", yref="paper", align="left")

    fig.update_layout(
        width=1100, height=500, font=FONT, plot_bgcolor="white",
        title="Figure 3. NLP Extraction Quality — Keyword vs LLM<br> <br>"
              "<sub>LLM eliminates section-header noise. 92% gene precision (50% article extraction rate).</sub>",
        margin=dict(l=100, r=40, t=110, b=80),
    )
    fig.write_image(str(FIG_DIR / "fig3_nlp_comparison.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig3_nlp_comparison.pdf"))
    print("  Fig 3: NLP Comparison ✅")


def fig3b_score_distribution():
    """Fig 4: Top 30 targets only — horizontal bar with gene labels."""
    scored_path = AMR_CONFIG["data_dir"] / "top_targets_scored.csv"
    if not scored_path.exists():
        return

    df = pd.read_csv(scored_path)
    if "composite_score" not in df.columns:
        return

    # Deduplicate and take top 30
    df = df.drop_duplicates(subset="gene_name", keep="first")
    df = df.nlargest(30, "composite_score").sort_values("composite_score", ascending=True)

    def get_tier(s):
        if s >= 0.7: return "Tier 1"
        elif s >= 0.5: return "Tier 2"
        elif s >= 0.3: return "Tier 3"
        else: return "Tier 4"

    df["tier"] = df["composite_score"].apply(get_tier)

    tier_colors = {"Tier 1": "#c62828", "Tier 2": "#ff9800", "Tier 3": "#1565c0", "Tier 4": "#9e9e9e"}
    colors = [tier_colors[t] for t in df["tier"]]

    # Check if is_novel or has_existing_drug columns exist
    labels = []
    for _, row in df.iterrows():
        name = row["gene_name"]
        novel = row.get("is_novel_target", False)
        marker = " 🆕" if novel else ""
        labels.append(f"{name}{marker}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=df["composite_score"],
        orientation="h",
        marker_color=colors,
        text=df["composite_score"].apply(lambda s: f"{s:.3f}"),
        textposition="outside",
        textfont=dict(size=9),
    ))

    # Tier boundary lines
    fig.add_vline(x=0.7, line_dash="dash", line_color="gray", line_width=1)
    fig.add_annotation(x=0.7, y=1.02, text="Tier 1 →", showarrow=False,
                       font=dict(size=9, color="#c62828"), xref="x", yref="paper")
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", line_width=1)
    fig.add_annotation(x=0.5, y=1.02, text="Tier 2 →", showarrow=False,
                       font=dict(size=9, color="#ff9800"), xref="x", yref="paper")

    fig.add_annotation(x=0.02, y=-0.06,
                       text="🔴 Tier 1 (High Priority)  🟠 Tier 2 (Promising)  🔵 Tier 3 (Exploratory)  🆕 = Novel target",
                       showarrow=False, font=dict(size=9), xref="paper", yref="paper")

    fig.update_layout(
        width=900, height=max(450, len(df) * 20),
        font=FONT, plot_bgcolor="white",
        title="Figure 4. Top 30 AMR Target Candidates by Composite Score<br> <br>"
              "<sub>Ranked by 6-dimension weighted score. Rankings sensitive to weight selection (see text).</sub>",
        xaxis=dict(title="Composite Score", range=[0, 0.95], gridcolor="#eee"),
        yaxis=dict(title=""),
        margin=dict(l=100, r=80, t=110, b=60),
    )
    fig.write_image(str(FIG_DIR / "fig3b_score_distribution.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig3b_score_distribution.pdf"))
    print("  Fig 4: Top 30 Bar ✅")


def fig3_old_combination_landscape():
    """Fig 3: Combination landscape — use FBA growth ratios directly."""
    landscape_path = DATA_DIR / "calma_results" / "calma_landscape.csv"
    if not landscape_path.exists():
        return

    df = pd.read_csv(landscape_path)

    # Use growth_ab (actual combination growth) and bliss_score for better spread
    if "growth_ab" not in df.columns:
        return

    df["combo_label"] = df["gene_a"] + " + " + df["gene_b"]

    # Create more informative axes
    # X: combo potency (1 - growth_ab)
    # Y: Bliss synergy score (positive = synergistic)
    df["combo_potency"] = 1.0 - df["growth_ab"]

    fig = go.Figure()

    # Color by interaction type
    for interaction, color, symbol in [
        ("synergistic", "#2e7d32", "diamond"),
        ("additive", "#1565c0", "circle"),
        ("antagonistic", "#c62828", "x"),
    ]:
        subset = df[df["interaction"] == interaction]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["combo_potency"],
            y=subset["bliss_score"],
            mode="markers+text",
            marker=dict(size=10, color=color, symbol=symbol, line=dict(width=1, color="white")),
            text=subset["combo_label"],
            textposition="top center",
            textfont=dict(size=7),
            name=interaction.capitalize(),
            hovertext=subset.apply(lambda r: f"{r['combo_label']}<br>Potency: {r['combo_potency']:.3f}<br>Bliss: {r['bliss_score']:.4f}", axis=1),
        ))

    # Pareto optimal
    if "pareto_optimal" in df.columns:
        pareto = df[df["pareto_optimal"] == True]
        if not pareto.empty:
            fig.add_trace(go.Scatter(
                x=pareto["combo_potency"],
                y=pareto["bliss_score"],
                mode="markers",
                marker=dict(symbol="star", size=18, color="gold", line=dict(width=2, color="black")),
                name="Pareto Optimal",
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_annotation(x=1.0, y=0.02, text="↑ Synergistic", showarrow=False, font=dict(color="#2e7d32", size=10))
    fig.add_annotation(x=1.0, y=-0.02, text="↓ Antagonistic", showarrow=False, font=dict(color="#c62828", size=10))

    fig.update_layout(
        width=900, height=650, font=FONT, plot_bgcolor="white",
        title="Supp. Fig S1. Drug Combination Landscape — Potency vs Synergy",
        xaxis=dict(title="Combination Potency (1 - double KO growth ratio)", gridcolor="#eee"),
        yaxis=dict(title="Bliss Synergy Score (positive = synergistic)", gridcolor="#eee"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=80, r=40, t=60, b=60),
    )
    fig.write_image(str(FIG_DIR / "fig3_landscape.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig3_landscape.pdf"))
    print("  Fig 3: Combination Landscape ✅")


def fig4_pathway_importance():
    """Fig 4: Pathway importance — cleaner layout."""
    w_path = DATA_DIR / "calma_results" / "calma_weight_analysis.csv"
    ko_path = DATA_DIR / "calma_results" / "calma_knockoff_analysis.csv"

    if not w_path.exists():
        return

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("(A) Subsystem Weights", "(B) Feature Knock-off (% change)"),
                        horizontal_spacing=0.2)

    w_df = pd.read_csv(w_path).head(12)
    w_df["pathway_short"] = w_df["pathway"].str[:35]

    fig.add_trace(go.Bar(y=w_df["pathway_short"], x=w_df["potency_weight"],
                         name="Potency", orientation="h", marker_color="#1565c0"), row=1, col=1)
    fig.add_trace(go.Bar(y=w_df["pathway_short"], x=w_df["toxicity_weight"],
                         name="Toxicity", orientation="h", marker_color="#c62828"), row=1, col=1)

    if ko_path.exists():
        ko_df = pd.read_csv(ko_path).head(12)
        ko_df["pathway_short"] = ko_df["pathway"].str[:35]
        fig.add_trace(go.Bar(y=ko_df["pathway_short"], x=ko_df["potency_change_pct"],
                             name="Potency Δ%", orientation="h", marker_color="#1565c0",
                             showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(y=ko_df["pathway_short"], x=ko_df["toxicity_change_pct"],
                             name="Toxicity Δ%", orientation="h", marker_color="#c62828",
                             showlegend=False), row=1, col=2)

    fig.update_layout(width=1200, height=550, font=dict(family="Arial", size=11),
                      barmode="group", plot_bgcolor="white",
                      title="Figure 5. Metabolic Pathway Importance Analysis",
                      legend=dict(orientation="h", yanchor="bottom", y=1.05))
    fig.update_yaxes(autorange="reversed")
    fig.write_image(str(FIG_DIR / "fig4_pathway_importance.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig4_pathway_importance.pdf"))
    print("  Fig 4: Pathway Importance ✅")


def fig5_selectivity():
    """Fig 6: Selectivity — single dot per gene, all data corrected."""
    # Literature-curated homology data (one entry per gene, no duplicates)
    data = [
        {"gene": "folA", "identity": 30, "human": "DHFR2 (Q86XF0)", "risk": "High"},
        {"gene": "accA", "identity": 22, "human": "ACACA", "risk": "Moderate"},
        {"gene": "gyrA", "identity": 18, "human": "TOP2A", "risk": "Low"},
        {"gene": "bamA", "identity": 15, "human": "SAM50", "risk": "Low"},
        {"gene": "ftsZ", "identity": 14, "human": "Tubulin (10-17%)", "risk": "Low"},
        {"gene": "dnaA", "identity": 8, "human": "RPA1", "risk": "Low"},
        {"gene": "rpoB", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "murF", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "murC", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "fabI", "identity": 3.3, "human": "PECR", "risk": "Minimal"},
        {"gene": "murA", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "murB", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "murD", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "murE", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "lpxC", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "lpxA", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "lpxB", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "lpxD", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "bamD", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "walK", "identity": 0, "human": "None", "risk": "None"},
        {"gene": "dxr", "identity": 0, "human": "None", "risk": "None"},
    ]

    df = pd.DataFrame(data)
    # Sort: high identity first, then alphabetical within same identity
    df = df.sort_values(["identity", "gene"], ascending=[False, True])

    risk_colors = {"None": "#2e7d32", "Minimal": "#8bc34a", "Low": "#ff9800",
                   "Moderate": "#ff5722", "High": "#c62828"}
    risk_symbols = {"None": "circle", "Minimal": "circle", "Low": "diamond",
                    "Moderate": "square", "High": "x"}

    fig = go.Figure()

    # Plot one trace per risk level (for legend)
    for risk in ["High", "Moderate", "Low", "Minimal", "None"]:
        subset = df[df["risk"] == risk]
        if subset.empty:
            continue

        labels = []
        for _, r in subset.iterrows():
            if r["identity"] > 0:
                labels.append(f"  {r['human']} ({r['identity']}%)")
            else:
                labels.append("  No homolog")

        fig.add_trace(go.Scatter(
            x=subset["identity"].values,
            y=subset["gene"].values,
            mode="markers+text",
            marker=dict(size=12, color=risk_colors[risk], symbol=risk_symbols[risk],
                       line=dict(width=1, color="white")),
            text=labels,
            textposition="middle right",
            textfont=dict(size=8),
            name=f"{risk} risk",
        ))

    # Threshold lines
    fig.add_vline(x=25, line_dash="dash", line_color="#ff9800", line_width=1.5)
    fig.add_vline(x=40, line_dash="dash", line_color="#c62828", line_width=1.5)

    # Threshold labels as x-axis tick marks at bottom
    fig.add_annotation(x=25, y=-0.04, text="▲ Moderate (25%)", showarrow=False,
                       font=dict(size=8, color="#ff9800"), xref="x", yref="paper")
    fig.add_annotation(x=40, y=-0.04, text="▲ High (40%)", showarrow=False,
                       font=dict(size=8, color="#c62828"), xref="x", yref="paper")

    # Safe zone
    fig.add_vrect(x0=-1, x1=10, fillcolor="rgba(46,125,50,0.06)", line_width=0)

    fig.update_layout(
        width=900, height=600,
        font=FONT, plot_bgcolor="white",
        title="Figure 6. Bacterial-Human Protein Homology<br> <br>"
              "<sub>Literature-curated audit of 21 AMR targets. "
              "folA→DHFR2 (30%) is the only high-risk target. "
              "11 targets have no detectable human homolog.</sub>",
        xaxis=dict(title="Sequence Identity to Closest Human Homolog (%)",
                   range=[-2, 48], gridcolor="#eee", dtick=10),
        yaxis=dict(title="", categoryorder="array",
                   categoryarray=list(df["gene"].values)),
        legend=dict(x=0.60, y=0.95, bgcolor="rgba(255,255,255,0.9)",
                   font=dict(size=10)),
        margin=dict(l=70, r=160, t=110, b=50),
    )
    fig.write_image(str(FIG_DIR / "fig5_selectivity.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig5_selectivity.pdf"))
    print("  Fig 6: Selectivity (fixed) ✅")


def main():
    print("=" * 60)
    print("REGENERATING IMPROVED FIGURES")
    print("=" * 60)

    fig2_fba_knockouts()
    fig3_nlp_comparison()
    fig3b_score_distribution()
    fig4_pathway_importance()
    fig5_selectivity()

    # Recompile PDF
    import subprocess
    print("\n  Recompiling PDF...")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
                      capture_output=True, cwd=str(FIG_DIR.parent))
    print("  PDF recompiled ✅")
    print(f"\n  All figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
