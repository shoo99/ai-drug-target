#!/usr/bin/env python3
"""
Day 2 Pipeline: NLP extraction + AI prediction + Scoring
Runs for both AMR and Pruritus tracks
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.common.nlp_extractor import NLPExtractor
from src.common.target_predictor import TargetPredictor
from src.common.scoring import TargetScorer
from src.common.knowledge_graph import KnowledgeGraph
from config.settings import AMR_CONFIG, PRURITUS_CONFIG, DATA_DIR


def run_nlp_analysis():
    """Run NLP extraction on both tracks."""
    print("=" * 60)
    print("STEP 1: NLP GENE EXTRACTION FROM PAPERS")
    print("=" * 60)

    # --- AMR Track ---
    print("\n[AMR Track]")
    amr_articles_path = AMR_CONFIG["data_dir"] / "pubmed_amr_articles.json"
    if amr_articles_path.exists():
        with open(amr_articles_path) as f:
            amr_articles = json.load(f)

        # Known AMR gene targets
        amr_known_genes = [
            "MURA", "MURB", "MURC", "MURD", "MURE", "MURF", "MURZ",
            "FTSI", "FTSZ", "FTSW", "FTSQ",
            "GYRA", "GYRB", "PARC", "PARE",
            "RPOB", "RPOC", "RPOA",
            "FOLP", "FOLA", "DHFR", "DHPS",
            "LPXA", "LPXB", "LPXC", "LPXD",
            "DNAK", "DNAJ", "DNAG",
            "BAMB", "BAMD", "BAME",
            "TOPA", "TOPB",
            "WALR", "WALK",
            "LEPA", "LEPB",
            "ACPP", "ACPA", "ACPS",
        ]
        extractor = NLPExtractor(known_genes=amr_known_genes)
        amr_results = extractor.process_articles(amr_articles)

        print(f"  Unique genes found: {len(amr_results['gene_counts'])}")
        print(f"  Relationships extracted: {len(amr_results['relationships'])}")

        # Top mentioned genes
        top_genes = amr_results["gene_counts"].most_common(30)
        print(f"  Top 15 genes: {[g[0] for g in top_genes[:15]]}")

        # Trending genes
        trending = extractor.get_trending_genes(amr_articles)
        if not trending.empty:
            trending.to_csv(AMR_CONFIG["data_dir"] / "trending_genes.csv", index=False)
            print(f"  Trending genes saved: {len(trending)}")

        # Save NLP results
        pd.DataFrame(top_genes, columns=["gene", "count"]).to_csv(
            AMR_CONFIG["data_dir"] / "nlp_gene_counts.csv", index=False
        )

        # Load new relationships into graph
        kg = KnowledgeGraph()
        loaded = 0
        for rel in amr_results["relationships"][:500]:
            g1, g2 = rel["gene1"], rel["gene2"]
            g1_id = f"NLP_{g1}"
            g2_id = f"NLP_{g2}"
            kg.add_gene(g1_id, g1, source="nlp")
            kg.add_gene(g2_id, g2, source="nlp")
            kg.add_relationship(
                "Gene", "gene_id", g1_id,
                rel["relationship"],
                "Gene", "gene_id", g2_id,
                source="nlp_pubmed"
            )
            loaded += 1
        kg.close()
        print(f"  Loaded {loaded} NLP relationships into graph")

    # --- Pruritus Track ---
    print("\n[Pruritus Track]")
    pru_articles_path = PRURITUS_CONFIG["data_dir"] / "pubmed_pruritus_articles.json"
    if pru_articles_path.exists():
        with open(pru_articles_path) as f:
            pru_articles = json.load(f)

        pru_known_genes = PRURITUS_CONFIG["known_targets"] + [
            "TSLP", "IL4", "IL13", "IL33", "IL25",
            "TRPM8", "TRPV3", "TRPV4",
            "MRGPRD", "MRGPRE",
            "SST", "SSTR2", "BNP", "NPPB",
            "GRPR", "GRP", "NMBR",
            "PAR2", "F2RL1",
            "OSMR", "IL31RA",
            "PIRT", "NTRK1",
            "SCN9A", "SCN10A", "SCN11A",
            "P2RX3", "P2RX4",
            "CYSLTR1", "CYSLTR2",
            "LTB4R", "LTB4R2",
            "PTGER2", "PTGER4",
        ]
        extractor = NLPExtractor(known_genes=pru_known_genes)
        pru_results = extractor.process_articles(pru_articles)

        print(f"  Unique genes found: {len(pru_results['gene_counts'])}")
        print(f"  Relationships extracted: {len(pru_results['relationships'])}")

        top_genes = pru_results["gene_counts"].most_common(30)
        print(f"  Top 15 genes: {[g[0] for g in top_genes[:15]]}")

        trending = extractor.get_trending_genes(pru_articles)
        if not trending.empty:
            trending.to_csv(PRURITUS_CONFIG["data_dir"] / "trending_genes.csv", index=False)

        pd.DataFrame(top_genes, columns=["gene", "count"]).to_csv(
            PRURITUS_CONFIG["data_dir"] / "nlp_gene_counts.csv", index=False
        )

        # Load into graph
        kg = KnowledgeGraph()
        loaded = 0
        for rel in pru_results["relationships"][:500]:
            g1, g2 = rel["gene1"], rel["gene2"]
            g1_id = f"NLP_{g1}"
            g2_id = f"NLP_{g2}"
            kg.add_gene(g1_id, g1, source="nlp")
            kg.add_gene(g2_id, g2, source="nlp")
            kg.add_relationship(
                "Gene", "gene_id", g1_id,
                rel["relationship"],
                "Gene", "gene_id", g2_id,
                source="nlp_pubmed"
            )
            loaded += 1
        kg.close()
        print(f"  Loaded {loaded} NLP relationships into graph")


def run_ai_prediction():
    """Run AI target prediction on the knowledge graph."""
    print("\n" + "=" * 60)
    print("STEP 2: AI TARGET PREDICTION (Link Prediction)")
    print("=" * 60)

    predictor = TargetPredictor()

    try:
        predictor.train()

        # AMR predictions
        print("\n[AMR Predictions]")
        amr_predictions = predictor.predict_novel_targets(track="amr")
        if not amr_predictions.empty:
            amr_predictions.to_csv(AMR_CONFIG["data_dir"] / "ai_predictions.csv", index=False)
            print(f"  Total predictions: {len(amr_predictions)}")
            print(f"  Novel (untargeted) candidates: "
                  f"{len(amr_predictions[~amr_predictions.get('already_targeted', True)])}")
            print("\n  Top 10 AI-predicted targets:")
            for _, row in amr_predictions.head(10).iterrows():
                name = row.get("gene_name", "unknown")
                score = row.get("avg_target_score", row.get("embedding_centrality", 0))
                print(f"    {name}: {score:.4f}")

        # Save predictions
        amr_predictions.to_csv(AMR_CONFIG["data_dir"] / "ai_predictions.csv", index=False)

    except Exception as e:
        print(f"[Predictor] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        predictor.close()


def run_scoring():
    """Score and rank all target candidates."""
    print("\n" + "=" * 60)
    print("STEP 3: TARGET SCORING & RANKING")
    print("=" * 60)

    scorer = TargetScorer()
    kg = KnowledgeGraph()

    # Get all gene nodes with their properties
    genes_query = """
    MATCH (g:Gene)
    OPTIONAL MATCH (g)-[:ENCODES]->(p:Protein)
    OPTIONAL MATCH (g)-[:ASSOCIATED_WITH]->(d:Disease)
    OPTIONAL MATCH (g)-[r]-(connected)
    WITH g, p, d, count(DISTINCT connected) as connections
    RETURN g.gene_id as gene_id, g.name as name,
           g.essential as essential,
           g.opentargets_score as ot_score,
           g.genetic_association as genetic_assoc,
           g.is_known_target as is_known,
           p.has_structure as has_structure,
           connections
    ORDER BY connections DESC
    """
    genes = kg.run_query(genes_query)
    kg.close()

    print(f"\n  Total genes to score: {len(genes)}")

    # Load AI predictions if available
    ai_pred_amr = pd.DataFrame()
    ai_pred_path = AMR_CONFIG["data_dir"] / "ai_predictions.csv"
    if ai_pred_path.exists():
        ai_pred_amr = pd.read_csv(ai_pred_path)

    # Load NLP trending data
    trending_amr = pd.DataFrame()
    trending_path = AMR_CONFIG["data_dir"] / "trending_genes.csv"
    if trending_path.exists():
        trending_amr = pd.read_csv(trending_path)

    trending_pru = pd.DataFrame()
    trending_path = PRURITUS_CONFIG["data_dir"] / "trending_genes.csv"
    if trending_path.exists():
        trending_pru = pd.read_csv(trending_path)

    # Score each gene
    scored_targets = []
    for gene in genes:
        gene_name = gene.get("name", "")
        gene_id = gene.get("gene_id", "")

        if not gene_name or gene_name == "None":
            continue

        # Build evidence dict
        ot_score = gene.get("ot_score") or 0
        genetic = gene.get("genetic_assoc") or 0
        connections = gene.get("connections") or 0
        has_structure = bool(gene.get("has_structure"))
        is_essential = bool(gene.get("essential"))
        is_known = bool(gene.get("is_known"))

        # Genetic evidence
        genetic_ev = max(float(ot_score), float(genetic))
        if is_essential:
            genetic_ev = max(genetic_ev, 0.8)

        # Expression specificity (proxy from connections)
        expression = min(connections / 10.0, 1.0)

        # Druggability
        drug_score = scorer.calculate_druggability(
            has_structure=has_structure,
            has_binding_pocket=has_structure,
        )

        # Novelty (inverse of known)
        novelty = 0.3 if is_known else 0.8

        # Literature trend
        lit_trend = 0.5
        if not trending_amr.empty and gene_name in trending_amr["gene"].values:
            row = trending_amr[trending_amr["gene"] == gene_name].iloc[0]
            lit_trend = min(row.get("trend_score", 1) / 5.0, 1.0)
        if not trending_pru.empty and gene_name in trending_pru["gene"].values:
            row = trending_pru[trending_pru["gene"] == gene_name].iloc[0]
            lit_trend = max(lit_trend, min(row.get("trend_score", 1) / 5.0, 1.0))

        # Competition (proxy: known = more competition)
        competition = 0.9 if not is_known else 0.3

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
        result["gene_name"] = gene_name
        result["is_known_target"] = is_known
        result["is_essential"] = is_essential
        result["connections"] = connections

        # Determine track
        if "HS_" in gene_id or is_known:
            result["track"] = "pruritus"
        elif "OT_" in gene_id:
            result["track"] = "both"
        else:
            result["track"] = "amr"

        scored_targets.append(result)

    df = pd.DataFrame(scored_targets)
    df = df.sort_values("composite_score", ascending=False)

    # Save results
    df.to_csv(DATA_DIR / "common" / "all_scored_targets.csv", index=False)

    # AMR targets
    amr_targets = df[df["track"].isin(["amr", "both"])].head(30)
    amr_targets.to_csv(AMR_CONFIG["data_dir"] / "top_targets_scored.csv", index=False)

    # Pruritus targets
    pru_targets = df[df["track"].isin(["pruritus", "both"])].head(30)
    pru_targets.to_csv(PRURITUS_CONFIG["data_dir"] / "top_targets_scored.csv", index=False)

    print(f"\n  Total scored: {len(df)}")
    print(f"\n{'='*60}")
    print("AMR — TOP 15 TARGET CANDIDATES")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(amr_targets.head(15).iterrows(), 1):
        print(f"  {i:2d}. {row['gene_name']:15s} | Score: {row['composite_score']:.3f} | "
              f"{row['tier']} | Known: {'Y' if row['is_known_target'] else 'N'}")

    print(f"\n{'='*60}")
    print("PRURITUS — TOP 15 TARGET CANDIDATES")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(pru_targets.head(15).iterrows(), 1):
        print(f"  {i:2d}. {row['gene_name']:15s} | Score: {row['composite_score']:.3f} | "
              f"{row['tier']} | Known: {'Y' if row['is_known_target'] else 'N'}")

    return df


def main():
    run_nlp_analysis()
    run_ai_prediction()
    results = run_scoring()

    print(f"\n{'='*60}")
    print("✅ DAY 2 ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Results saved to: {DATA_DIR}/")
    print(f"  Total targets scored: {len(results)}")


if __name__ == "__main__":
    main()
