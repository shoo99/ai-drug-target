#!/usr/bin/env python3
"""Generate all paper figures as high-resolution PNG/PDF."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.settings import DATA_DIR, AMR_CONFIG, PRURITUS_CONFIG

FIG_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Common styling
FONT = dict(family="Arial", size=14)
COLORS = {"primary": "#0d47a1", "accent": "#00897b", "red": "#c62828",
           "green": "#2e7d32", "amber": "#f57f17", "gray": "#757575"}


def fig1_architecture():
    """Figure 1: Platform architecture diagram (simplified as flowchart)."""
    fig = go.Figure()

    # Boxes — wider with smaller font to keep text inside
    boxes = [
        (0.5, 0.95, "Public Databases<br><sub>ChEMBL · UniProt · OpenTargets<br>PubMed · AlphaFold · FAERS</sub>", "#e3f2fd", 0.22, 0.06),
        (0.15, 0.7, "Knowledge Graph<br><sub>Neo4j</sub>", "#bbdefb", 0.13, 0.055),
        (0.5, 0.7, "GEM + FBA<br><sub>COBRApy</sub>", "#c8e6c9", 0.13, 0.055),
        (0.85, 0.7, "LLM NLP<br><sub>Ollama</sub>", "#fff3e0", 0.13, 0.055),
        (0.5, 0.45, "CALMA Engine<br><sub>Sigma/Delta · Subsystem ANN · Pareto</sub>", "#e1bee7", 0.22, 0.055),
        (0.15, 0.2, "Sequence<br>Homology<br><sub>Toxicity</sub>", "#ffcdd2", 0.13, 0.06),
        (0.5, 0.2, "Multi-dim<br>Scoring<br><sub>6 axes</sub>", "#b2dfdb", 0.13, 0.06),
        (0.85, 0.2, "Clinical Trials<br>+ Patents", "#d7ccc8", 0.13, 0.055),
        (0.5, 0.0, "Dashboard (10 tabs) · PDF Reports · Target Rankings", "#e8eaf6", 0.28, 0.045),
    ]

    for item in boxes:
        x, y, text, color = item[0], item[1], item[2], item[3]
        hw = item[4] if len(item) > 4 else 0.14
        hh = item[5] if len(item) > 5 else 0.06
        fig.add_shape(type="rect",
                      x0=x-hw, y0=y-hh, x1=x+hw, y1=y+hh,
                      fillcolor=color, line=dict(color="#455a64", width=1.5))
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                          font=dict(size=8), align="center")

    # Arrows
    arrows = [(0.5, 0.89, 0.15, 0.755), (0.5, 0.89, 0.5, 0.755), (0.5, 0.89, 0.85, 0.755),
              (0.15, 0.645, 0.5, 0.505), (0.5, 0.645, 0.5, 0.505), (0.85, 0.645, 0.5, 0.505),
              (0.5, 0.395, 0.15, 0.26), (0.5, 0.395, 0.5, 0.26), (0.5, 0.395, 0.85, 0.255),
              (0.15, 0.14, 0.5, 0.045), (0.5, 0.14, 0.5, 0.045), (0.85, 0.145, 0.5, 0.045)]

    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                          axref="x", ayref="y", showarrow=True,
                          arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#455a64")

    fig.update_layout(xaxis=dict(range=[-0.05, 1.05], visible=False),
                      yaxis=dict(range=[-0.12, 1.08], visible=False),
                      width=800, height=900, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text="Figure 1. Platform Architecture", font=FONT))
    fig.write_image(str(FIG_DIR / "fig1_architecture.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig1_architecture.pdf"))
    print("  Fig 1: Architecture ✅")


def fig2_fba_knockouts():
    """Figure 2: FBA gene knockout results."""
    fba_path = DATA_DIR / "metabolic_analysis" / "metabolic_Escherichia_coli.csv"
    if not fba_path.exists():
        print("  Fig 2: No FBA data ❌")
        return

    df = pd.read_csv(fba_path)
    valid = df[df["growth_ratio"].notna()].sort_values("growth_ratio")

    fig = px.bar(valid, y="gene", x="growth_ratio", orientation="h",
                 color="toxicity_risk",
                 color_discrete_map={"low": COLORS["green"], "moderate": COLORS["amber"], "high": COLORS["red"]},
                 labels={"growth_ratio": "Growth Ratio (KO/WT)", "gene": "", "toxicity_risk": "Toxicity Risk"})

    fig.add_vline(x=0.01, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="Lethal", annotation_position="top right")

    fig.update_layout(width=800, height=500, font=FONT, plot_bgcolor="white",
                      title="Figure 2. FBA Gene Knockout Simulation (E. coli iML1515)",
                      xaxis_title="Growth Ratio (0 = lethal)",
                      yaxis=dict(categoryorder="total ascending"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.write_image(str(FIG_DIR / "fig2_fba_knockouts.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig2_fba_knockouts.pdf"))
    print("  Fig 2: FBA Knockouts ✅")


def fig3_combination_landscape():
    """Figure 3: Drug combination potency-toxicity landscape."""
    landscape_path = DATA_DIR / "calma_results" / "calma_landscape.csv"
    if not landscape_path.exists():
        print("  Fig 3: No landscape data ❌")
        return

    df = pd.read_csv(landscape_path)

    fig = px.scatter(df, x="calma_potency", y="calma_toxicity",
                     color="quadrant" if "quadrant" in df.columns else None,
                     size="calma_quality", size_max=15,
                     hover_name=df.apply(lambda r: f"{r.get('gene_a','')}-{r.get('gene_b','')}", axis=1),
                     color_discrete_sequence=["#2e7d32", "#c62828", "#1565c0", "#757575"])

    # Pareto stars
    if "pareto_optimal" in df.columns:
        pareto = df[df["pareto_optimal"] == True]
        if not pareto.empty:
            fig.add_trace(go.Scatter(
                x=pareto["calma_potency"], y=pareto["calma_toxicity"],
                mode="markers", marker=dict(symbol="star", size=20, color="gold",
                                            line=dict(width=2, color="black")),
                name="Pareto Optimal",
                text=pareto.apply(lambda r: f"{r['gene_a']}+{r['gene_b']}", axis=1)))

    # Quadrant lines
    pot_med = df["calma_potency"].median()
    tox_med = df["calma_toxicity"].median()
    fig.add_hline(y=tox_med, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=pot_med, line_dash="dot", line_color="gray", opacity=0.5)

    # Ideal zone
    fig.add_shape(type="rect", x0=pot_med, y0=df["calma_toxicity"].min()-0.01,
                  x1=df["calma_potency"].max()+0.01, y1=tox_med,
                  fillcolor="rgba(76,175,80,0.1)", line=dict(color="green", dash="dot"))
    fig.add_annotation(x=df["calma_potency"].max(), y=df["calma_toxicity"].min(),
                       text="IDEAL ZONE", showarrow=False, font=dict(color="green", size=12))

    fig.update_layout(width=900, height=700, font=FONT, plot_bgcolor="white",
                      title="Figure 3. Drug Combination Potency-Toxicity Landscape",
                      xaxis_title="Predicted Potency (higher = better)",
                      yaxis_title="Predicted Toxicity (lower = better)",
                      legend=dict(font=dict(size=10)))
    fig.write_image(str(FIG_DIR / "fig3_landscape.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig3_landscape.pdf"))
    print("  Fig 3: Combination Landscape ✅")


def fig4_pathway_importance():
    """Figure 4: Pathway importance (weight + knockoff)."""
    w_path = DATA_DIR / "calma_results" / "calma_weight_analysis.csv"
    ko_path = DATA_DIR / "calma_results" / "calma_knockoff_analysis.csv"

    if not w_path.exists():
        print("  Fig 4: No weight data ❌")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=("A. Subsystem Weights", "B. Feature Knock-off"),
                        horizontal_spacing=0.15)

    # A: Weights
    w_df = pd.read_csv(w_path).head(12)
    fig.add_trace(go.Bar(y=w_df["pathway"], x=w_df["potency_weight"], name="Potency",
                         orientation="h", marker_color=COLORS["primary"]), row=1, col=1)
    fig.add_trace(go.Bar(y=w_df["pathway"], x=w_df["toxicity_weight"], name="Toxicity",
                         orientation="h", marker_color=COLORS["red"]), row=1, col=1)

    # B: Knockoff
    if ko_path.exists():
        ko_df = pd.read_csv(ko_path).head(12)
        fig.add_trace(go.Bar(y=ko_df["pathway"], x=ko_df["potency_change_pct"], name="Potency Δ%",
                             orientation="h", marker_color=COLORS["primary"], showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(y=ko_df["pathway"], x=ko_df["toxicity_change_pct"], name="Toxicity Δ%",
                             orientation="h", marker_color=COLORS["red"], showlegend=False), row=1, col=2)

    fig.update_layout(width=1200, height=600, font=FONT, barmode="group",
                      title="Figure 4. Metabolic Pathway Importance Analysis",
                      legend=dict(orientation="h", yanchor="bottom", y=1.08))
    fig.update_yaxes(autorange="reversed")
    fig.write_image(str(FIG_DIR / "fig4_pathway_importance.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig4_pathway_importance.pdf"))
    print("  Fig 4: Pathway Importance ✅")


def fig5_sequence_selectivity():
    """Figure 5: Sequence homology heatmap."""
    seq_path = DATA_DIR / "sequence_toxicity" / "sequence_selectivity.csv"
    if not seq_path.exists():
        print("  Fig 5: No sequence data ❌")
        return

    df = pd.read_csv(seq_path)
    df = df[df["sequence_identity"] >= 0].sort_values("sequence_identity", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["gene"], x=df["sequence_identity"] * 100,
        orientation="h",
        marker=dict(
            color=df["sequence_identity"] * 100,
            colorscale=[[0, "#2e7d32"], [0.25, "#8bc34a"], [0.5, "#ffc107"], [1.0, "#c62828"]],
            cmin=0, cmax=50,
            colorbar=dict(title="Identity %"),
        ),
        text=df.apply(lambda r: f"{r['human_gene'] if r.get('human_gene') else 'None'} ({r['sequence_identity']*100:.1f}%)", axis=1),
        textposition="outside",
    ))

    fig.add_vline(x=25, line_dash="dash", line_color="orange", line_width=2,
                  annotation_text="Moderate risk (25%)", annotation_position="top right")
    fig.add_vline(x=40, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="High risk (40%)", annotation_position="top right")

    fig.update_layout(width=900, height=600, font=FONT, plot_bgcolor="white",
                      title="Figure 5. Bacterial-Human Sequence Identity for Top AMR Targets",
                      xaxis_title="Sequence Identity to Closest Human Homolog (%)",
                      yaxis=dict(categoryorder="total ascending"))
    fig.write_image(str(FIG_DIR / "fig5_selectivity.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig5_selectivity.pdf"))
    print("  Fig 5: Sequence Selectivity ✅")


def main():
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    fig1_architecture()
    fig2_fba_knockouts()
    fig3_combination_landscape()
    fig4_pathway_importance()
    fig5_sequence_selectivity()

    print(f"\n  All figures saved to: {FIG_DIR}")
    print(f"  Files: {list(FIG_DIR.glob('*.png'))}")


if __name__ == "__main__":
    main()
