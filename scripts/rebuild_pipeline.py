#!/usr/bin/env python3
"""
REBUILD PIPELINE — Fix data quality issues and re-run everything.
Clears old data, collects curated data, rebuilds graph, re-scores.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import AMR_CONFIG, PRURITUS_CONFIG, DATA_DIR
from src.common.knowledge_graph import KnowledgeGraph
from src.common.scoring import TargetScorer


def clear_graph():
    """Clear Neo4j and start fresh."""
    print("\n[1/6] Clearing Neo4j graph...")
    kg = KnowledgeGraph()
    kg.run_query("MATCH (n) DETACH DELETE n")
    kg.setup_amr_schema()
    kg.setup_pruritus_schema()
    stats = kg.get_graph_stats()
    print(f"  Graph cleared: {stats}")
    kg.close()


def collect_amr_v2():
    """Collect AMR data with curated essential genes."""
    print("\n[2/6] Collecting AMR data (V2 — curated)...")
    from src.amr.data_collector_v2 import AMRDataCollectorV2
    collector = AMRDataCollectorV2()
    return collector.collect_all()


def load_amr_to_graph(amr_data: dict):
    """Load AMR V2 data into Neo4j."""
    print("\n[3/6] Loading AMR data into graph...")
    kg = KnowledgeGraph()

    # Load organisms
    gram_types = {
        "Enterococcus faecium": "positive", "Staphylococcus aureus": "positive",
        "Klebsiella pneumoniae": "negative", "Acinetobacter baumannii": "negative",
        "Pseudomonas aeruginosa": "negative", "Enterobacter cloacae": "negative",
    }
    for org in AMR_CONFIG["eskape_organisms"]:
        taxon_id = org.replace(" ", "_").lower()
        kg.add_bacterium(taxon_id, org, gram=gram_types.get(org))

    # Load essential genes (curated)
    ess_df = amr_data["essential_genes"]
    uniprot_df = amr_data.get("uniprot", pd.DataFrame())
    uniprot_map = {}
    if not uniprot_df.empty:
        for _, row in uniprot_df.iterrows():
            key = f"{row['gene_name']}_{row['organism']}"
            uniprot_map[key] = row

    loaded = 0
    for _, row in ess_df.iterrows():
        gene = row["gene_name"]
        organism = row["organism"]
        gene_id = f"AMR_{gene}_{organism.split()[0][:4]}"

        kg.add_gene(gene_id, gene,
                     organism=organism,
                     essential=True,
                     essential_score=float(row["essential_score"]),
                     function=row["function"],
                     pathway=row["pathway"],
                     has_existing_drug=bool(row["has_existing_drug"]),
                     existing_drugs=row["existing_drugs"],
                     is_novel_target=bool(row["is_novel_target"]))

        # Link to organism
        taxon_id = organism.replace(" ", "_").lower()
        kg.add_relationship("Gene", "gene_id", gene_id,
                           "ESSENTIAL_IN", "Bacterium", "taxon_id", taxon_id)

        # Link to protein if available
        key = f"{gene}_{organism}"
        if key in uniprot_map:
            up = uniprot_map[key]
            kg.add_protein(up["uniprot_id"], up.get("protein_name", ""),
                          has_structure=bool(up.get("has_structure", False)))
            kg.add_relationship("Gene", "gene_id", gene_id,
                               "ENCODES", "Protein", "uniprot_id", up["uniprot_id"])

        # Add drug relationships
        if row["has_existing_drug"] and row["existing_drugs"] != "NONE":
            for drug_name in row["existing_drugs"].split(", "):
                drug_id = f"DRUG_{drug_name.lower().replace(' ', '_')}"
                kg.add_drug(drug_id, drug_name, approved=True)
                kg.add_relationship("Drug", "drug_id", drug_id,
                                   "TARGETS", "Gene", "gene_id", gene_id)
        loaded += 1

    print(f"  Loaded {loaded} essential gene entries")

    # Load PubMed articles
    pubmed_path = AMR_CONFIG["data_dir"] / "pubmed_amr_v2.json"
    if pubmed_path.exists():
        with open(pubmed_path) as f:
            articles = json.load(f)
        for a in articles[:500]:
            kg.add_paper(a["pmid"], a.get("title", ""), year=a.get("year"), track="amr")
        print(f"  Loaded {min(len(articles), 500)} papers")

    stats = kg.get_graph_stats()
    print(f"  Graph stats: {stats}")
    kg.close()


def load_pruritus_fresh():
    """Re-collect and load pruritus data."""
    print("\n[4/6] Re-loading Pruritus data...")
    from src.pruritus.data_collector import PruritusDataCollector
    from src.pruritus.graph_loader import PruritusGraphLoader

    collector = PruritusDataCollector()
    collector.collect_all()

    loader = PruritusGraphLoader()
    loader.load_all()


def run_scoring_v2():
    """Improved scoring with proper calibration."""
    print("\n[5/6] Running improved scoring...")
    scorer = TargetScorer()
    kg = KnowledgeGraph()

    # Get all genes with rich properties
    query = """
    MATCH (g:Gene)
    OPTIONAL MATCH (g)-[:ENCODES]->(p:Protein)
    OPTIONAL MATCH (g)-[:ESSENTIAL_IN]->(b:Bacterium)
    OPTIONAL MATCH (d:Drug)-[:TARGETS]->(g)
    OPTIONAL MATCH (g)-[r]-(connected)
    WITH g, p, b, collect(DISTINCT d.name) as drugs, count(DISTINCT connected) as connections
    RETURN g.gene_id as gene_id, g.name as name,
           g.essential as essential, g.essential_score as ess_score,
           g.function as function, g.pathway as pathway,
           g.is_novel_target as is_novel,
           g.has_existing_drug as has_drug,
           g.existing_drugs as drugs_str,
           g.opentargets_score as ot_score,
           g.genetic_association as genetic_assoc,
           g.is_known_target as is_known_pru,
           g.organism as organism,
           p.has_structure as has_structure,
           p.uniprot_id as uniprot_id,
           b.name as bacterium,
           drugs, connections
    """
    genes = kg.run_query(query)
    kg.close()

    print(f"  Total genes: {len(genes)}")

    scored = []
    for gene in genes:
        name = gene.get("name", "")
        gene_id = gene.get("gene_id", "")
        if not name or name == "None":
            continue

        is_essential = bool(gene.get("essential"))
        ess_score = float(gene.get("ess_score") or 0)
        has_drug = bool(gene.get("has_drug"))
        is_novel = bool(gene.get("is_novel"))
        is_known_pru = bool(gene.get("is_known_pru"))
        has_structure = bool(gene.get("has_structure"))
        ot_score = float(gene.get("ot_score") or 0)
        connections = int(gene.get("connections") or 0)
        pathway = gene.get("pathway") or ""
        organism = gene.get("organism") or ""

        # Determine track
        if "AMR_" in gene_id:
            track = "amr"
        elif "HS_" in gene_id or is_known_pru:
            track = "pruritus"
        else:
            track = "both"

        # --- GENETIC EVIDENCE ---
        genetic_ev = 0.0
        if is_essential:
            genetic_ev = max(ess_score, 0.7)
        if ot_score > 0:
            genetic_ev = max(genetic_ev, ot_score)
        genetic_ev = min(genetic_ev, 1.0)

        # --- EXPRESSION SPECIFICITY ---
        expression = 0.3
        if is_essential:
            expression = 0.8  # Essential = expressed and critical
        if connections > 5:
            expression = max(expression, min(connections / 15.0, 0.9))

        # --- DRUGGABILITY ---
        drug_score = 0.2
        if has_structure:
            drug_score += 0.3
        # Pathway-based druggability
        druggable_pathways = {
            "peptidoglycan_synthesis": 0.3, "lps_synthesis": 0.35,
            "dna_topology": 0.3, "fatty_acid_synthesis": 0.3,
            "folate_synthesis": 0.3, "cell_division": 0.25,
            "outer_membrane": 0.3, "signal_transduction": 0.25,
            "isoprenoid_synthesis": 0.3, "protein_secretion": 0.2,
        }
        if pathway in druggable_pathways:
            drug_score += druggable_pathways[pathway]
        drug_score = min(drug_score, 1.0)

        # --- NOVELTY ---
        if is_novel:
            novelty = 0.9  # No existing drug = highly novel
        elif has_drug:
            novelty = 0.2  # Already has drugs = low novelty
        elif is_known_pru:
            novelty = 0.3
        else:
            novelty = 0.7

        # --- COMPETITION ---
        if is_novel:
            competition = 0.95  # No competition
        elif has_drug:
            competition = 0.2  # Existing drugs = competitive
        else:
            competition = 0.7

        # --- LITERATURE TREND ---
        lit_trend = 0.5
        if is_essential and is_novel:
            lit_trend = 0.8  # Hot area
        if pathway in ("lps_synthesis", "outer_membrane"):
            lit_trend = 0.85  # Very active research area

        evidence = {
            "genetic_evidence": genetic_ev,
            "expression_specificity": expression,
            "druggability": drug_score,
            "novelty": novelty,
            "competition": competition,
            "literature_trend": lit_trend,
        }

        result = scorer.score_target(evidence)
        result["gene_id"] = gene_id
        result["gene_name"] = name
        result["track"] = track
        result["is_essential"] = is_essential
        result["essential_score"] = ess_score
        result["is_novel_target"] = is_novel
        result["has_existing_drug"] = has_drug
        result["pathway"] = pathway
        result["organism"] = organism or ""
        result["has_structure"] = has_structure
        result["uniprot_id"] = gene.get("uniprot_id") or ""
        result["connections"] = connections
        scored.append(result)

    df = pd.DataFrame(scored)
    df = df.sort_values("composite_score", ascending=False)
    df.to_csv(DATA_DIR / "common" / "all_scored_targets.csv", index=False)

    # Track-specific
    amr_df = df[df["track"].isin(["amr", "both"])].sort_values("composite_score", ascending=False)
    amr_df.to_csv(AMR_CONFIG["data_dir"] / "top_targets_scored.csv", index=False)

    pru_df = df[df["track"].isin(["pruritus", "both"])].sort_values("composite_score", ascending=False)
    pru_df.to_csv(PRURITUS_CONFIG["data_dir"] / "top_targets_scored.csv", index=False)

    # Print results
    print(f"\n{'='*60}")
    print(f"AMR — TOP 15 TARGETS (RECALIBRATED)")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(amr_df.head(15).iterrows(), 1):
        novel = "🆕 NOVEL" if row.get("is_novel_target") else "💊 Known"
        print(f"  {i:2d}. {row['gene_name']:12s} | Score: {row['composite_score']:.3f} | "
              f"{row['tier']:25s} | {novel} | {row.get('pathway', '')}")

    print(f"\n{'='*60}")
    print(f"PRURITUS — TOP 15 TARGETS (RECALIBRATED)")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(pru_df.head(15).iterrows(), 1):
        known = "✅ Known" if row.get("is_known_target") else "🆕 Novel"
        print(f"  {i:2d}. {row['gene_name']:12s} | Score: {row['composite_score']:.3f} | "
              f"{row['tier']:25s} | {known}")

    return df


def regenerate_reports():
    """Regenerate all reports with new data."""
    print("\n[6/6] Regenerating reports...")
    from src.common.report_generator import ReportGenerator
    gen = ReportGenerator()
    reports = gen.generate_all_reports()
    print(f"  Generated {len(reports)} reports")


def main():
    print("=" * 60)
    print("🔧 FULL PIPELINE REBUILD")
    print("=" * 60)

    clear_graph()
    amr_data = collect_amr_v2()
    load_amr_to_graph(amr_data)
    try:
        load_pruritus_fresh()
    except Exception as e:
        print(f"  Pruritus loading error (using existing data): {e}")
    results = run_scoring_v2()
    regenerate_reports()

    print(f"\n{'='*60}")
    print(f"✅ REBUILD COMPLETE")
    print(f"  Total targets scored: {len(results)}")
    print(f"  AMR: {len(results[results['track'].isin(['amr', 'both'])])}")
    print(f"  Pruritus: {len(results[results['track'].isin(['pruritus', 'both'])])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
