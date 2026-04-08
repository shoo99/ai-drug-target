#!/usr/bin/env python3
"""Load LLM NLP results into Neo4j knowledge graph + re-run scoring."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.common.knowledge_graph import KnowledgeGraph
from config.settings import DATA_DIR, AMR_CONFIG, PRURITUS_CONFIG


def load_llm_genes():
    """Load LLM-extracted genes into knowledge graph."""
    print("\n[1/3] Loading LLM genes into Neo4j...")
    kg = KnowledgeGraph()

    # Gene counts
    gene_path = DATA_DIR / "llm_nlp" / "llm_gene_counts.csv"
    if not gene_path.exists():
        print("  No LLM gene data found")
        return

    df = pd.read_csv(gene_path)
    print(f"  {len(df)} unique genes to load")

    loaded = 0
    for _, row in df.iterrows():
        gene = str(row["gene"]).strip()
        count = int(row["count"])
        if not gene or len(gene) < 2 or len(gene) > 15:
            continue

        gene_id = f"LLM_{gene}"
        kg.add_gene(gene_id, gene,
                     source="llm_nlp",
                     llm_mention_count=count)
        loaded += 1

    print(f"  Loaded {loaded} genes")

    # Relations
    rel_path = DATA_DIR / "llm_nlp" / "llm_relations.csv"
    if rel_path.exists():
        rel_df = pd.read_csv(rel_path)
        rel_loaded = 0
        for _, row in rel_df.iterrows():
            g1 = str(row.get("gene1", "")).strip()
            g2 = str(row.get("gene2", "")).strip()
            rel = str(row.get("relation", "CO_MENTIONED")).upper().replace(" ", "_")
            if not g1 or not g2 or len(g1) < 2 or len(g2) < 2:
                continue

            # Validate relationship type
            valid_rels = {"INHIBITS", "ACTIVATES", "BINDS", "REGULATES",
                         "PHOSPHORYLATES", "UPREGULATES", "DOWNREGULATES",
                         "TARGETS", "INTERACTS_WITH", "CO_MENTIONED", "CO_EXPRESSED"}
            if rel not in valid_rels:
                rel = "CO_MENTIONED"

            g1_id = f"LLM_{g1}"
            g2_id = f"LLM_{g2}"
            kg.add_gene(g1_id, g1, source="llm_nlp")
            kg.add_gene(g2_id, g2, source="llm_nlp")
            try:
                kg.add_relationship("Gene", "gene_id", g1_id, rel,
                                   "Gene", "gene_id", g2_id, source="llm_nlp")
                rel_loaded += 1
            except Exception:
                pass
        print(f"  Loaded {rel_loaded} relations")

    # Drugs
    drug_path = DATA_DIR / "llm_nlp" / "llm_drugs.csv"
    if drug_path.exists():
        drug_df = pd.read_csv(drug_path)
        drug_loaded = 0
        for _, row in drug_df.iterrows():
            drug_name = str(row.get("name", "")).strip()
            target_gene = str(row.get("target_gene", "")).strip()
            mechanism = str(row.get("mechanism", "")).strip()
            if not drug_name or len(drug_name) < 2:
                continue

            drug_id = f"LLM_DRUG_{drug_name.replace(' ', '_')[:30]}"
            kg.add_drug(drug_id, drug_name, source="llm_nlp", mechanism=mechanism)

            if target_gene and len(target_gene) >= 2:
                gene_id = f"LLM_{target_gene.upper()}"
                kg.add_gene(gene_id, target_gene.upper(), source="llm_nlp")
                try:
                    kg.add_relationship("Drug", "drug_id", drug_id, "TARGETS",
                                       "Gene", "gene_id", gene_id,
                                       source="llm_nlp", mechanism=mechanism)
                    drug_loaded += 1
                except Exception:
                    pass
        print(f"  Loaded {drug_loaded} drug-target relationships")

    stats = kg.get_graph_stats()
    print(f"  Graph: {stats}")
    kg.close()


def rerun_scoring():
    """Re-run scoring with updated graph data."""
    print("\n[2/3] Re-running target scoring...")

    # Import and run the scoring from rebuild pipeline
    from scripts.rebuild_pipeline import run_scoring_v2
    results = run_scoring_v2()
    return results


def rerun_reports():
    """Regenerate reports with new data."""
    print("\n[3/3] Regenerating reports...")
    from src.common.report_generator import ReportGenerator
    gen = ReportGenerator()
    reports = gen.generate_all_reports()
    print(f"  Generated {len(reports)} reports")


def main():
    print("=" * 60)
    print("LOAD LLM RESULTS → GRAPH → SCORING → REPORTS")
    print("=" * 60)

    load_llm_genes()
    results = rerun_scoring()
    rerun_reports()

    print(f"\n{'='*60}")
    print("✅ COMPLETE")
    print(f"  Targets scored: {len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
