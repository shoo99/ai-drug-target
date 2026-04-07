"""
AI Drug Target Discovery Platform — Streamlit Dashboard
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json

from config.settings import AMR_CONFIG, PRURITUS_CONFIG, DATA_DIR, REPORTS_DIR
from src.common.knowledge_graph import KnowledgeGraph
from src.common.report_generator import ReportGenerator

st.set_page_config(
    page_title="AI Drug Target Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
st.sidebar.title("🧬 Drug Target Discovery")
st.sidebar.markdown("AI-powered platform for novel drug target identification")

page = st.sidebar.radio("Navigation", ["🎯 Target Analysis", "📋 Sales CRM"])

if page == "📋 Sales CRM":
    from src.dashboard.crm import render_crm
    render_crm()
    st.stop()

track = st.sidebar.radio("Select Track", ["AMR (항생제 내성)", "Pruritus (만성 가려움증)"])
track_key = "amr" if "AMR" in track else "pruritus"
config = AMR_CONFIG if track_key == "amr" else PRURITUS_CONFIG

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Status")
st.sidebar.markdown("✅ Data Collection")
st.sidebar.markdown("✅ Knowledge Graph")
st.sidebar.markdown("✅ NLP Mining")
st.sidebar.markdown("✅ AI Prediction")
st.sidebar.markdown("✅ Target Scoring")


# --- Helper functions ---
@st.cache_data
def load_scored_targets(track: str) -> pd.DataFrame:
    path = (AMR_CONFIG if track == "amr" else PRURITUS_CONFIG)["data_dir"] / "top_targets_scored.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_nlp_genes(track: str) -> pd.DataFrame:
    path = (AMR_CONFIG if track == "amr" else PRURITUS_CONFIG)["data_dir"] / "nlp_gene_counts.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_trending(track: str) -> pd.DataFrame:
    path = (AMR_CONFIG if track == "amr" else PRURITUS_CONFIG)["data_dir"] / "trending_genes.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_all_targets() -> pd.DataFrame:
    path = DATA_DIR / "common" / "all_scored_targets.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_clinical_trials(track: str) -> pd.DataFrame:
    name = "antimicrobial" if track == "amr" else "pruritus"
    path = DATA_DIR / "common" / "clinical_trials" / f"trials_{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_clinical_trials_full(track: str) -> list:
    name = "antimicrobial" if track == "amr" else "pruritus"
    path = DATA_DIR / "common" / "clinical_trials" / f"trials_{name}_full.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data
def load_docking_results() -> list:
    path = DATA_DIR / "docking" / "results" / "docking_analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data
def load_alphafold_data(track: str) -> pd.DataFrame:
    path = (AMR_CONFIG if track == "amr" else PRURITUS_CONFIG)["data_dir"] / "alphafold_druggability.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def get_graph_stats():
    try:
        kg = KnowledgeGraph()
        stats = kg.get_graph_stats()
        kg.close()
        return stats[0] if stats else {"nodes": 0, "relationships": 0}
    except Exception:
        return {"nodes": "N/A", "relationships": "N/A"}


# --- Main Content ---
track_name = "Antimicrobial Resistance" if track_key == "amr" else "Chronic Pruritus"
st.title(f"🎯 {track_name} — Target Discovery")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
stats = get_graph_stats()
scored = load_scored_targets(track_key)
nlp_genes = load_nlp_genes(track_key)

col1.metric("Knowledge Graph Nodes", stats.get("nodes", "N/A"))
col2.metric("Graph Relationships", stats.get("relationships", "N/A"))
col3.metric("Target Candidates", len(scored))
col4.metric("Genes from NLP", len(nlp_genes))

st.markdown("---")

# --- Tab Layout ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🏆 Top Targets", "📊 Score Analysis", "📈 Trending Genes",
    "🕸️ Network View", "📄 Reports",
    "🧪 Clinical Trials", "🔬 Docking / Structure", "📜 Patents",
    "🧬 Metabolic Analysis", "🧬 CALMA Analysis"
])

# Tab 1: Top Targets
with tab1:
    st.subheader("Top Ranked Target Candidates")

    if not scored.empty:
        # Filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            min_score = st.slider("Minimum Score", 0.0, 1.0, 0.3, 0.05)
        with fcol2:
            show_known = st.checkbox("Show known targets", value=True)

        filtered = scored[scored["composite_score"] >= min_score]
        if not show_known and "is_known_target" in filtered.columns:
            filtered = filtered[~filtered["is_known_target"].astype(bool)]

        # Display table
        display_cols = ["gene_name", "composite_score", "tier",
                        "genetic_evidence", "druggability", "novelty",
                        "competition", "is_known_target"]
        available_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[available_cols].head(30),
            use_container_width=True,
            hide_index=True,
            column_config={
                "gene_name": st.column_config.TextColumn("Gene", width="medium"),
                "composite_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="%.3f"
                ),
                "tier": st.column_config.TextColumn("Tier"),
                "genetic_evidence": st.column_config.ProgressColumn(
                    "Genetic Ev.", min_value=0, max_value=1
                ),
                "druggability": st.column_config.ProgressColumn(
                    "Druggability", min_value=0, max_value=1
                ),
                "novelty": st.column_config.ProgressColumn(
                    "Novelty", min_value=0, max_value=1
                ),
                "competition": st.column_config.ProgressColumn(
                    "Competition", min_value=0, max_value=1
                ),
                "is_known_target": st.column_config.CheckboxColumn("Known"),
            }
        )

        # Target count by tier
        if "tier" in filtered.columns:
            tier_counts = filtered["tier"].value_counts()
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                title="Target Distribution by Tier",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No scored targets found. Run the analysis pipeline first.")

# Tab 2: Score Analysis
with tab2:
    st.subheader("Multi-dimensional Score Analysis")

    if not scored.empty:
        # Radar chart for top 5
        top5 = scored.head(5)
        dimensions = ["genetic_evidence", "expression_specificity", "druggability",
                       "novelty", "competition", "literature_trend"]
        available_dims = [d for d in dimensions if d in top5.columns]

        if available_dims:
            fig = go.Figure()
            for _, row in top5.iterrows():
                values = [row.get(d, 0) for d in available_dims]
                values.append(values[0])  # Close the polygon
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=available_dims + [available_dims[0]],
                    fill='toself',
                    name=row.get("gene_name", "?"),
                    opacity=0.6,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Top 5 Targets — Score Radar",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Score distribution
        fig2 = px.histogram(
            scored, x="composite_score", nbins=20,
            title="Composite Score Distribution",
            labels={"composite_score": "Score"},
            color_discrete_sequence=["#1a237e"],
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Scatter: Novelty vs Druggability
        if "novelty" in scored.columns and "druggability" in scored.columns:
            fig3 = px.scatter(
                scored, x="druggability", y="novelty",
                size="composite_score", color="composite_score",
                hover_name="gene_name",
                title="Druggability vs Novelty (bubble size = composite score)",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig3, use_container_width=True)

# Tab 3: Trending Genes
with tab3:
    st.subheader("Literature Trend Analysis")

    trending = load_trending(track_key)
    if not trending.empty:
        # Bar chart
        top_trending = trending.head(20)
        fig = px.bar(
            top_trending, x="gene", y="trend_score",
            color="total_mentions",
            title="Top Trending Genes (Recent vs Older Publications)",
            labels={"gene": "Gene", "trend_score": "Trend Score",
                    "total_mentions": "Total Mentions"},
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.dataframe(trending.head(30), use_container_width=True, hide_index=True)
    else:
        st.info("No trending data available.")

    # NLP gene frequency
    st.subheader("Most Mentioned Genes in Literature")
    if not nlp_genes.empty:
        fig = px.bar(
            nlp_genes.head(25), x="gene", y="count",
            title="Gene Mention Frequency in PubMed Abstracts",
            color="count", color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Network View
with tab4:
    st.subheader("Knowledge Graph Network Visualization")

    try:
        kg = KnowledgeGraph()
        # Get a subgraph of top connected nodes
        query = """
        MATCH (g:Gene)-[r]-(connected)
        WITH g, count(r) as rels
        ORDER BY rels DESC LIMIT 30
        MATCH (g)-[r]-(connected)
        RETURN g.name as source, type(r) as rel, connected.name as target,
               labels(connected) as target_labels
        LIMIT 200
        """
        results = kg.run_query(query)
        kg.close()

        if results:
            G = nx.Graph()
            for r in results:
                source = r.get("source", "")
                target = r.get("target", "")
                if source and target and source != "None" and target != "None":
                    G.add_node(source, type="Gene")
                    labels = r.get("target_labels", [])
                    G.add_node(target, type=labels[0] if labels else "Other")
                    G.add_edge(source, target, relationship=r.get("rel", ""))

            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                # Create plotly figure
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color="#888"),
                    hoverinfo="none", mode="lines"
                )

                node_x, node_y, node_text, node_color = [], [], [], []
                color_map = {"Gene": "#1a237e", "Protein": "#00897b",
                             "Drug": "#c62828", "Disease": "#f57f17",
                             "Paper": "#757575", "Bacterium": "#2e7d32",
                             "Other": "#9e9e9e"}

                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    ntype = G.nodes[node].get("type", "Other")
                    node_color.append(color_map.get(ntype, "#9e9e9e"))

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers+text",
                    text=node_text,
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(size=10, color=node_color, line=dict(width=1, color="white")),
                    hoverinfo="text"
                )

                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    title="Knowledge Graph (Top Connected Genes)",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "🔵 Gene | 🟢 Protein/Bacterium | 🔴 Drug | 🟡 Disease | ⚪ Other"
                )
            else:
                st.info("No network data to display.")
        else:
            st.info("No graph data available.")
    except Exception as e:
        st.error(f"Could not connect to Neo4j: {e}")

# Tab 5: Reports
with tab5:
    st.subheader("Generate & Download Reports")

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        if st.button(f"📄 Generate {track_key.upper()} Summary Report"):
            with st.spinner("Generating report..."):
                gen = ReportGenerator()
                path = gen.generate_summary_report(track_key)
                if path and path.exists():
                    with open(path, "rb") as f:
                        st.download_button(
                            f"📥 Download {track_key.upper()} Summary PDF",
                            f.read(),
                            file_name=path.name,
                            mime="application/pdf",
                        )
                    st.success(f"Report generated: {path.name}")

    with rcol2:
        if st.button("📄 Generate All Reports (Both Tracks)"):
            with st.spinner("Generating all reports..."):
                gen = ReportGenerator()
                paths = gen.generate_all_reports()
                st.success(f"Generated {len(paths)} reports in {REPORTS_DIR}")

    # List existing reports
    st.markdown("### Existing Reports")
    report_files = sorted(REPORTS_DIR.glob("*.pdf"))
    if report_files:
        for rf in report_files:
            col_a, col_b = st.columns([3, 1])
            col_a.text(rf.name)
            with open(rf, "rb") as f:
                col_b.download_button(
                    "📥", f.read(), file_name=rf.name,
                    mime="application/pdf", key=rf.name,
                )
    else:
        st.info("No reports generated yet. Click above to generate.")

# Tab 6: Clinical Trials
with tab6:
    st.subheader("Clinical Trial Landscape")

    trials_df = load_clinical_trials(track_key)
    trials_full = load_clinical_trials_full(track_key)

    if not trials_df.empty:
        # Competition overview
        st.markdown("### Competition Overview")
        comp_counts = trials_df["competition_level"].value_counts()
        fig = px.pie(
            values=comp_counts.values, names=comp_counts.index,
            title="Target Competition Distribution",
            color_discrete_map={
                "none": "#4caf50", "low": "#8bc34a",
                "moderate": "#ffc107", "high": "#ff9800",
                "very_high": "#f44336"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.markdown("### Per-Target Clinical Trial Summary")
        display_cols = [c for c in ["gene", "total_trials", "active_trials",
                        "completed_trials", "most_advanced_phase",
                        "competition_level", "n_unique_sponsors"] if c in trials_df.columns]
        st.dataframe(
            trials_df[display_cols].sort_values("total_trials"),
            use_container_width=True, hide_index=True,
            column_config={
                "gene": "Gene",
                "total_trials": st.column_config.NumberColumn("Total Trials"),
                "active_trials": st.column_config.NumberColumn("Active"),
                "completed_trials": st.column_config.NumberColumn("Completed"),
                "most_advanced_phase": "Most Advanced",
                "competition_level": "Competition",
                "n_unique_sponsors": st.column_config.NumberColumn("Sponsors"),
            }
        )

        # Blue ocean targets
        blue_ocean = trials_df[trials_df["competition_level"] == "none"]
        if not blue_ocean.empty:
            st.markdown("### 🟢 Blue Ocean Targets (0 Clinical Trials)")
            st.success(f"**{len(blue_ocean)} targets** have NO existing clinical trials — "
                       f"first-mover opportunity!")
            for _, row in blue_ocean.iterrows():
                st.markdown(f"- **{row['gene']}** — No competition detected")

        # Trial details
        if trials_full:
            st.markdown("### Trial Details")
            selected_gene = st.selectbox("Select target for details",
                                         [t["gene"] for t in trials_full])
            gene_data = next((t for t in trials_full if t["gene"] == selected_gene), None)
            if gene_data:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Trials", gene_data["total_trials"])
                c2.metric("Active", gene_data["active_trials"])
                c3.metric("Competition", gene_data["competition_level"])

                if gene_data.get("top_sponsors"):
                    st.markdown("**Top Sponsors:**")
                    for sponsor, count in gene_data["top_sponsors"]:
                        st.markdown(f"- {sponsor}: {count} trials")

                if gene_data.get("trials_summary"):
                    st.markdown("**Recent Trials:**")
                    for trial in gene_data["trials_summary"][:5]:
                        st.markdown(
                            f"- **[{trial['nct_id']}]** {trial['title']} "
                            f"| {trial['phase']} | {trial['status']} | {trial['sponsor']}"
                        )
    else:
        st.info("No clinical trial data. Run `scripts/run_advanced.py` first.")


# Tab 7: Docking / Structure
with tab7:
    st.subheader("Molecular Docking & Structural Analysis")

    # AlphaFold data
    af_data = load_alphafold_data(track_key)
    docking_data = load_docking_results()

    if not af_data.empty:
        st.markdown("### AlphaFold Druggability Assessment")

        # Bar chart
        af_sorted = af_data.sort_values("druggability_score", ascending=True)
        fig = px.bar(
            af_sorted, y="gene", x="druggability_score",
            orientation="h",
            color="druggability_score",
            color_continuous_scale="RdYlGn",
            title="Druggability Score by Target",
            labels={"druggability_score": "Druggability", "gene": "Gene"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display_cols = [c for c in ["gene", "uniprot_id", "has_alphafold",
                        "has_pdb_experimental", "protein_class",
                        "druggability_score", "modality_suggestion"] if c in af_data.columns]
        st.dataframe(
            af_data[display_cols],
            use_container_width=True, hide_index=True,
            column_config={
                "gene": "Gene",
                "uniprot_id": "UniProt",
                "has_alphafold": st.column_config.CheckboxColumn("AlphaFold"),
                "has_pdb_experimental": st.column_config.CheckboxColumn("PDB"),
                "protein_class": "Protein Class",
                "druggability_score": st.column_config.ProgressColumn(
                    "Druggability", min_value=0, max_value=1, format="%.3f"
                ),
                "modality_suggestion": "Suggested Modality",
            }
        )

    # Docking results
    if docking_data:
        st.markdown("### Binding Pocket Analysis")
        analyzed = [d for d in docking_data if d.get("status") == "analyzed"]

        if analyzed:
            pocket_data = []
            for d in analyzed:
                for pocket in d.get("pockets", []):
                    pocket_data.append({
                        "Gene": d["gene"],
                        "Pocket Rank": pocket["rank"],
                        "Volume (ų)": pocket["volume_estimate"],
                        "Nearby Residues": pocket["n_nearby_residues"],
                        "Hydrophobic Ratio": pocket["hydrophobic_ratio"],
                        "Pocket Score": pocket["druggability_score"],
                    })

            if pocket_data:
                pocket_df = pd.DataFrame(pocket_data)
                st.dataframe(pocket_df, use_container_width=True, hide_index=True)

                # Scatter
                fig = px.scatter(
                    pocket_df, x="Volume (ų)", y="Hydrophobic Ratio",
                    size="Pocket Score", color="Gene",
                    title="Binding Pocket Landscape",
                    hover_data=["Pocket Rank", "Nearby Residues"],
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No structures analyzed. Docking requires AlphaFold structures.")
    else:
        st.info("No docking data. Run `scripts/run_advanced.py` first.")


# Tab 8: Patents
with tab8:
    st.subheader("Patent Landscape & IP Opportunity")

    patent_name = "antimicrobial_resistance" if track_key == "amr" else "pruritus_itch"
    patent_path = DATA_DIR / "common" / "patents" / f"patent_analysis_{patent_name}.csv"

    if patent_path.exists():
        patent_df = pd.read_csv(patent_path)

        # IP Opportunity chart
        ip_counts = patent_df["ip_opportunity"].value_counts()
        fig = px.pie(
            values=ip_counts.values, names=ip_counts.index,
            title="IP Opportunity Distribution",
            color_discrete_map={
                "high": "#4caf50", "moderate": "#ffc107", "low": "#f44336"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display_cols = [c for c in ["gene", "total_patents_found", "recent_patents_3yr",
                        "patent_activity", "freedom_to_operate",
                        "ip_opportunity"] if c in patent_df.columns]
        st.dataframe(
            patent_df[display_cols],
            use_container_width=True, hide_index=True,
            column_config={
                "gene": "Gene",
                "total_patents_found": st.column_config.NumberColumn("Patents Found"),
                "recent_patents_3yr": st.column_config.NumberColumn("Recent (3yr)"),
                "patent_activity": "Activity",
                "freedom_to_operate": "FTO Status",
                "ip_opportunity": "IP Opportunity",
            }
        )

        # High opportunity targets
        high_ip = patent_df[patent_df["ip_opportunity"] == "high"]
        if not high_ip.empty:
            st.markdown("### 🟢 High IP Opportunity Targets")
            st.success(f"**{len(high_ip)} targets** with high patent opportunity!")
            for _, row in high_ip.iterrows():
                st.markdown(f"- **{row['gene']}** — {row.get('freedom_to_operate', 'N/A')}")
    else:
        st.info("No patent data. Run `scripts/run_advanced.py` first.")


# Tab 9: Metabolic Analysis
with tab9:
    st.subheader("Metabolism-Informed Analysis (CALMA-inspired)")
    st.caption("Genome-Scale Metabolic Models + FBA gene knockout simulation + Toxicity prediction")

    # Load FBA results
    fba_ecoli_path = DATA_DIR / "metabolic_analysis" / "metabolic_Escherichia_coli.csv"
    fba_sa_path = DATA_DIR / "metabolic_analysis" / "metabolic_Staphylococcus_aureus.csv"
    nn_path = DATA_DIR / "metabolic_analysis" / "nn_ranked_targets.csv"
    pathway_path = DATA_DIR / "metabolic_analysis" / "pathway_importance.csv"

    fba_frames = []
    if fba_ecoli_path.exists():
        fba_frames.append(pd.read_csv(fba_ecoli_path))
    if fba_sa_path.exists():
        fba_frames.append(pd.read_csv(fba_sa_path))

    if fba_frames:
        fba_all = pd.concat(fba_frames, ignore_index=True)
        fba_valid = fba_all[fba_all["growth_ratio"].notna()]

        # Overview metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Genes Simulated", len(fba_valid))
        mc2.metric("Lethal Knockouts", len(fba_valid[fba_valid["is_lethal"] == True]))
        mc3.metric("Low Toxicity", len(fba_valid[fba_valid["toxicity_risk"] == "low"]))
        best = fba_valid[(fba_valid["is_lethal"] == True) & (fba_valid["toxicity_risk"] == "low")]
        mc4.metric("Ideal Targets (Lethal+Safe)", len(best))

        st.markdown("---")

        # FBA Knockout Results
        st.markdown("### Gene Knockout Simulation (FBA)")
        st.markdown("Each bar shows how much bacterial growth is reduced when the gene is knocked out. "
                     "**0.0 = lethal** (complete growth arrest).")

        fba_plot = fba_valid.copy()
        fba_plot = fba_plot.sort_values("growth_ratio")

        fig = px.bar(
            fba_plot, y="gene", x="growth_ratio",
            color="toxicity_risk", orientation="h",
            color_discrete_map={"low": "#4caf50", "moderate": "#ff9800", "high": "#f44336", "unknown": "#9e9e9e"},
            title="Gene Knockout → Growth Ratio (lower = more potent target)",
            labels={"growth_ratio": "Growth Ratio (0=lethal)", "gene": "Gene", "toxicity_risk": "Toxicity Risk"},
            hover_data=["organism", "model", "n_affected_reactions", "potency_score", "target_quality"],
        )
        fig.add_vline(x=0.01, line_dash="dash", line_color="red", annotation_text="Lethal threshold")
        fig.update_layout(height=max(400, len(fba_plot) * 22))
        st.plotly_chart(fig, use_container_width=True)

        # Target Quality scatter
        st.markdown("### Potency vs Selectivity")
        fig2 = px.scatter(
            fba_valid, x="potency_score", y="selectivity",
            color="toxicity_risk", size="target_quality",
            hover_name="gene", text="gene",
            color_discrete_map={"low": "#4caf50", "moderate": "#ff9800", "high": "#f44336"},
            title="Target Quality Map (top-right = best: high potency + high selectivity)",
            labels={"potency_score": "Potency (1=lethal)", "selectivity": "Selectivity (1=safe)"},
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.add_shape(type="rect", x0=0.8, y0=0.6, x1=1.05, y1=1.05,
                       line=dict(color="green", dash="dot"), fillcolor="rgba(76,175,80,0.1)")
        fig2.add_annotation(x=0.92, y=1.02, text="Ideal Zone", showarrow=False, font=dict(color="green"))
        st.plotly_chart(fig2, use_container_width=True)

        # Detailed table
        st.markdown("### Detailed FBA Results")
        display_cols = [c for c in ["gene", "organism", "model", "growth_ratio", "is_lethal",
                        "n_affected_reactions", "affected_subsystems",
                        "potency_score", "toxicity_risk", "toxicity_score",
                        "selectivity", "target_quality"] if c in fba_valid.columns]
        st.dataframe(
            fba_valid[display_cols].sort_values("target_quality", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "gene": "Gene",
                "organism": "Organism",
                "model": "GEM Model",
                "growth_ratio": st.column_config.ProgressColumn("Growth Ratio", min_value=0, max_value=1, format="%.4f"),
                "is_lethal": st.column_config.CheckboxColumn("Lethal?"),
                "potency_score": st.column_config.ProgressColumn("Potency", min_value=0, max_value=1),
                "toxicity_risk": "Tox Risk",
                "toxicity_score": st.column_config.ProgressColumn("Tox Score", min_value=0, max_value=1),
                "selectivity": st.column_config.ProgressColumn("Selectivity", min_value=0, max_value=1),
                "target_quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=1),
            }
        )

    # Pathway Importance
    if pathway_path.exists():
        st.markdown("---")
        st.markdown("### Metabolic Pathway Importance (Neural Network)")
        st.markdown("Which metabolic subsystems most influence drug potency and toxicity predictions.")

        imp_df = pd.read_csv(pathway_path)
        imp_top = imp_df.head(15)

        fig3 = px.bar(
            imp_top, y="pathway", x=["potency_impact", "toxicity_impact"],
            orientation="h", barmode="group",
            title="Top 15 Metabolic Pathways by Importance",
            labels={"value": "Impact Score", "pathway": "Pathway"},
            color_discrete_map={"potency_impact": "#1565c0", "toxicity_impact": "#c62828"},
        )
        fig3.update_layout(height=500, legend_title="Impact Type")
        st.plotly_chart(fig3, use_container_width=True)

    # NN Rankings
    if nn_path.exists():
        st.markdown("---")
        st.markdown("### Metabolism-Informed NN — Final Rankings")

        nn_df = pd.read_csv(nn_path)
        nn_cols = [c for c in ["gene", "growth_ratio", "is_lethal", "potency_score",
                   "toxicity_risk", "nn_potency", "nn_toxicity", "nn_quality"] if c in nn_df.columns]
        st.dataframe(
            nn_df[nn_cols],
            use_container_width=True, hide_index=True,
            column_config={
                "gene": "Gene",
                "is_lethal": st.column_config.CheckboxColumn("Lethal?"),
                "nn_potency": st.column_config.ProgressColumn("NN Potency", min_value=0, max_value=1),
                "nn_toxicity": st.column_config.ProgressColumn("NN Toxicity", min_value=0, max_value=1),
                "nn_quality": st.column_config.ProgressColumn("NN Quality", min_value=0, max_value=1),
            }
        )

    if not fba_frames:
        st.info("No metabolic analysis data. Run `scripts/run_metabolic.py` first.")


# Tab 10: CALMA Analysis
with tab10:
    st.subheader("CALMA Analysis — Metabolism-Informed Drug Combinations")
    st.caption("4-Feature Sigma/Delta profiles + 3-Layer Subsystem ANN + Pareto Optimization")

    CALMA_DIR = DATA_DIR / "calma_results"
    landscape_path = CALMA_DIR / "calma_landscape.csv"
    weights_path = CALMA_DIR / "calma_weight_analysis.csv"
    knockoff_path = CALMA_DIR / "calma_knockoff_analysis.csv"
    experiment_path = CALMA_DIR / "experiment_design.txt"

    if landscape_path.exists():
        calma_df = pd.read_csv(landscape_path)

        # Metrics
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Total Combinations", len(calma_df))
        cc2.metric("Pareto Optimal", int(calma_df["pareto_optimal"].sum()) if "pareto_optimal" in calma_df.columns else 0)
        ideal_count = len(calma_df[calma_df["quadrant"].str.contains("IDEAL", na=False)]) if "quadrant" in calma_df.columns else 0
        cc3.metric("Ideal (High Pot + Low Tox)", ideal_count)
        reduction = round((1 - ideal_count / max(len(calma_df), 1)) * 100, 1)
        cc4.metric("Search Space Reduction", f"{reduction}%")

        st.markdown("---")

        # 1. 2D Landscape Scatter
        st.markdown("### Potency-Toxicity Landscape")
        if "calma_potency" in calma_df.columns and "calma_toxicity" in calma_df.columns:
            import numpy as np
            fig_land = px.scatter(
                calma_df, x="calma_potency", y="calma_toxicity",
                color="quadrant" if "quadrant" in calma_df.columns else None,
                size="calma_quality",
                hover_name=calma_df.apply(lambda r: f"{r.get('gene_a','')} + {r.get('gene_b','')}", axis=1),
                title="Drug Combination Landscape (Top-Left = IDEAL: High Potency + Low Toxicity)",
                labels={"calma_potency": "Predicted Potency", "calma_toxicity": "Predicted Toxicity"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            # Quadrant lines
            pot_med = calma_df["calma_potency"].median()
            tox_med = calma_df["calma_toxicity"].median()
            fig_land.add_hline(y=tox_med, line_dash="dash", line_color="gray", opacity=0.5)
            fig_land.add_vline(x=pot_med, line_dash="dash", line_color="gray", opacity=0.5)
            # Highlight Pareto
            if "pareto_optimal" in calma_df.columns:
                pareto_pts = calma_df[calma_df["pareto_optimal"] == True]
                if not pareto_pts.empty:
                    fig_land.add_trace(go.Scatter(
                        x=pareto_pts["calma_potency"], y=pareto_pts["calma_toxicity"],
                        mode="markers", marker=dict(symbol="star", size=18, color="gold", line=dict(width=2, color="black")),
                        name="Pareto Optimal", text=pareto_pts.apply(lambda r: f"{r.get('gene_a','')}+{r.get('gene_b','')}", axis=1),
                    ))
            fig_land.update_layout(height=550)
            st.plotly_chart(fig_land, use_container_width=True)

        # 2. Top Pareto Combinations
        if "pareto_optimal" in calma_df.columns:
            pareto_df = calma_df[calma_df["pareto_optimal"] == True].nlargest(10, "calma_quality")
            if not pareto_df.empty:
                st.markdown("### Pareto Optimal Combinations")
                pareto_df["combo"] = pareto_df["gene_a"] + " + " + pareto_df["gene_b"]
                fig_par = px.bar(
                    pareto_df, x="combo", y="calma_quality",
                    color="calma_toxicity", color_continuous_scale="RdYlGn_r",
                    title="Top Pareto Optimal Combinations (sorted by quality)",
                    labels={"calma_quality": "Quality Score", "combo": "Combination", "calma_toxicity": "Toxicity"},
                )
                st.plotly_chart(fig_par, use_container_width=True)

        st.markdown("---")

        # 3. Weight Analysis
        if weights_path.exists():
            st.markdown("### Pathway Weight Analysis (Model Interpretation)")
            w_df = pd.read_csv(weights_path).head(15)
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(y=w_df["pathway"], x=w_df["potency_weight"], name="Potency", orientation="h", marker_color="#1565c0"))
            fig_w.add_trace(go.Bar(y=w_df["pathway"], x=w_df["toxicity_weight"], name="Toxicity", orientation="h", marker_color="#c62828"))
            fig_w.update_layout(barmode="group", title="Subsystem Weights (Potency vs Toxicity)", height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_w, use_container_width=True)

        # 4. Feature Knock-off
        if knockoff_path.exists():
            st.markdown("### Feature Knock-off Analysis")
            st.caption("Effect of zeroing out each subsystem on predictions")
            ko_df = pd.read_csv(knockoff_path).head(15)
            fig_ko = go.Figure()
            fig_ko.add_trace(go.Bar(y=ko_df["pathway"], x=ko_df["potency_change_pct"], name="Potency Δ%", orientation="h", marker_color="#1565c0"))
            fig_ko.add_trace(go.Bar(y=ko_df["pathway"], x=ko_df["toxicity_change_pct"], name="Toxicity Δ%", orientation="h", marker_color="#c62828"))
            fig_ko.update_layout(barmode="group", title="Subsystem Knock-off Impact (%)", height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_ko, use_container_width=True)

        st.markdown("---")

        # 5. Experiment Design
        if experiment_path.exists():
            st.markdown("### Experimental Validation Protocol")
            with open(experiment_path) as f:
                exp_text = f.read()
            st.code(exp_text, language="markdown")

        # 6. Full Data Table
        st.markdown("### All Combinations (sortable)")
        display_cols = [c for c in ["gene_a", "gene_b", "calma_potency", "calma_toxicity",
                        "calma_quality", "bliss_score", "interaction", "quadrant",
                        "pareto_optimal"] if c in calma_df.columns]
        st.dataframe(
            calma_df[display_cols].sort_values("calma_quality", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No CALMA data. Run `scripts/run_calma_v2.py` first.")


# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#757575;font-size:12px;'>"
    "AI Drug Target Discovery Platform | "
    "Dual Track: AMR + Chronic Pruritus | "
    "Powered by Neo4j + Node2Vec + NLP"
    "</div>",
    unsafe_allow_html=True,
)
