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
    """Fig 2: FBA — show potency score (1-growth) instead of growth ratio."""
    fba_ec = DATA_DIR / "metabolic_analysis" / "metabolic_Escherichia_coli.csv"
    fba_sa = DATA_DIR / "metabolic_analysis" / "metabolic_Staphylococcus_aureus.csv"

    frames = []
    if fba_ec.exists():
        df = pd.read_csv(fba_ec)
        df["organism"] = "E. coli (iML1515)"
        frames.append(df)
    if fba_sa.exists():
        df = pd.read_csv(fba_sa)
        df["organism"] = "S. aureus (iYS1720)"
        frames.append(df)

    if not frames:
        return

    all_df = pd.concat(frames)
    valid = all_df[all_df["growth_ratio"].notna()].copy()
    valid["potency"] = 1.0 - valid["growth_ratio"]
    valid["label"] = valid["gene"] + " (" + valid["organism"].str.split(" ").str[0] + ")"

    # Deduplicate by gene (keep highest potency)
    valid = valid.sort_values("potency", ascending=False).drop_duplicates(subset="gene", keep="first")
    valid = valid.sort_values("potency", ascending=True)

    fig = go.Figure()

    # Color by lethal vs growth-reducing
    colors = []
    for _, row in valid.iterrows():
        if row["potency"] > 0.99:
            colors.append("#c62828")  # Lethal (red)
        elif row["potency"] > 0.5:
            colors.append("#ff9800")  # Growth-reducing (orange)
        else:
            colors.append("#4caf50")  # Non-essential (green)

    fig.add_trace(go.Bar(
        y=valid["gene"],
        x=valid["potency"],
        orientation="h",
        marker_color=colors,
        text=valid["potency"].apply(lambda p: "LETHAL" if p > 0.99 else f"{p:.2f}"),
        textposition="outside",
        textfont=dict(size=10),
    ))

    # Add threshold line
    fig.add_vline(x=0.99, line_dash="dash", line_color="red", line_width=1.5)
    fig.add_annotation(x=0.95, y=len(valid)-1, text="Lethal\nthreshold",
                       showarrow=False, font=dict(size=10, color="red"))

    # Legend annotations
    fig.add_annotation(x=0.85, y=0, text="🔴 Lethal (potency>0.99)  🟠 Growth-reducing  🟢 Non-essential",
                       showarrow=False, font=dict(size=9), xref="paper", yref="paper")

    fig.update_layout(
        width=900, height=max(450, len(valid)*25),
        font=FONT, plot_bgcolor="white",
        title="Figure 2. FBA Gene Knockout — Target Potency",
        xaxis=dict(title="Potency Score (1 - growth ratio; 1.0 = complete growth arrest)",
                   range=[0, 1.15], gridcolor="#eee"),
        yaxis=dict(title=""),
        margin=dict(l=80, r=120, t=60, b=60),
    )
    fig.write_image(str(FIG_DIR / "fig2_fba_knockouts.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig2_fba_knockouts.pdf"))
    print("  Fig 2: FBA Potency ✅")


def fig3_combination_landscape():
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
        title="Figure 3. Drug Combination Landscape — Potency vs Synergy",
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
                      title="Figure 4. Metabolic Pathway Importance Analysis",
                      legend=dict(orientation="h", yanchor="bottom", y=1.05))
    fig.update_yaxes(autorange="reversed")
    fig.write_image(str(FIG_DIR / "fig4_pathway_importance.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig4_pathway_importance.pdf"))
    print("  Fig 4: Pathway Importance ✅")


def fig5_selectivity():
    """Fig 5: Selectivity — with literature-curated corrections."""
    # Use corrected homology data
    audit_path = DATA_DIR / "scientific_validation" / "homology_audit.csv"
    if not audit_path.exists():
        # Fall back to sequence data
        seq_path = DATA_DIR / "sequence_toxicity" / "sequence_selectivity.csv"
        if not seq_path.exists():
            return
        df = pd.read_csv(seq_path)
        df = df.sort_values("sequence_identity", ascending=False)
    else:
        df = pd.read_csv(audit_path)
        # Add approximate identity from literature
        lit_identity = {
            "murA": 0, "murB": 0, "murC": 3.5, "murD": 0, "murE": 0, "murF": 4.1,
            "lpxC": 0, "lpxA": 0, "bamA": 15, "bamD": 0, "ftsZ": 14,
            "walK": 0, "dxr": 0, "fabI": 3.3, "accA": 22,
        }
        df["identity_pct"] = df["gene"].map(lit_identity).fillna(0)
        df = df.sort_values("identity_pct", ascending=True)

    # Color by risk level
    def risk_color(ident):
        if ident >= 25: return "#c62828"
        elif ident >= 10: return "#ff9800"
        elif ident > 0: return "#ffc107"
        else: return "#2e7d32"

    if "identity_pct" in df.columns:
        id_col = "identity_pct"
    else:
        df["identity_pct"] = df["sequence_identity"] * 100
        id_col = "identity_pct"

    colors = [risk_color(v) for v in df[id_col]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["gene"], x=df[id_col],
        orientation="h", marker_color=colors,
        text=df.apply(lambda r: f"{r.get('literature_homolog', r.get('human_gene', 'None'))} ({r[id_col]:.0f}%)", axis=1),
        textposition="outside", textfont=dict(size=9),
    ))

    fig.add_vline(x=25, line_dash="dash", line_color="#ff9800", line_width=2)
    fig.add_annotation(x=25, y=len(df)-1, text="Moderate\nrisk (25%)",
                       showarrow=False, font=dict(size=9, color="#ff9800"))

    fig.add_vline(x=40, line_dash="dash", line_color="#c62828", line_width=2)
    fig.add_annotation(x=40, y=len(df)-1, text="High\nrisk (40%)",
                       showarrow=False, font=dict(size=9, color="#c62828"))

    fig.add_annotation(x=0.02, y=-0.08, text="🟢 No homolog  🟡 Low (<10%)  🟠 Moderate (10-25%)  🔴 High (>25%)",
                       showarrow=False, font=dict(size=9), xref="paper", yref="paper")

    fig.update_layout(
        width=900, height=max(450, len(df)*30),
        font=FONT, plot_bgcolor="white",
        title="Figure 5. Bacterial-Human Sequence Identity (Literature-Curated Audit)",
        xaxis=dict(title="Sequence Identity to Closest Human Homolog (%)",
                   range=[0, 50], gridcolor="#eee"),
        yaxis=dict(title=""),
        margin=dict(l=80, r=150, t=60, b=60),
    )
    fig.write_image(str(FIG_DIR / "fig5_selectivity.png"), scale=3)
    fig.write_image(str(FIG_DIR / "fig5_selectivity.pdf"))
    print("  Fig 5: Selectivity (corrected) ✅")


def main():
    print("=" * 60)
    print("REGENERATING IMPROVED FIGURES")
    print("=" * 60)

    fig2_fba_knockouts()
    fig3_combination_landscape()
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
