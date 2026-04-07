#!/usr/bin/env python3
"""Run 3 professional upgrades: BioBERT NLP + Temporal Validation + Sequence Toxicity."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import DATA_DIR, AMR_CONFIG


def run_biobert():
    """Test BioBERT NLP on sample articles."""
    print("\n" + "=" * 60)
    print("1/3 — BioBERT NLP UPGRADE")
    print("=" * 60)

    from src.common.biobert_nlp import BioBERTExtractor

    extractor = BioBERTExtractor()

    # Test on sample abstracts
    test_texts = [
        "LpxC is an essential enzyme in the lipid A biosynthesis pathway of gram-negative bacteria. "
        "Inhibition of LpxC leads to bacterial cell death and represents a promising antibiotic target. "
        "We found that CHIR-090 inhibits LpxC with an IC50 of 4 nM.",

        "BamA is the central component of the BAM complex responsible for outer membrane protein biogenesis. "
        "Darobactin targets BamA and shows potent activity against gram-negative pathogens including "
        "Pseudomonas aeruginosa and Klebsiella pneumoniae.",

        "The WalK/WalR two-component system is essential for cell wall metabolism in Staphylococcus aureus. "
        "Targeting WalK with small molecule inhibitors disrupts peptidoglycan homeostasis.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n  Test {i}:")
        genes = extractor.extract_genes(text)
        print(f"    Genes found: {[g['name'] for g in genes]}")
        print(f"    With confidence: {[(g['name'], g['confidence']) for g in genes]}")

        rels = extractor.extract_relations(text, [g["name"] for g in genes])
        if rels:
            print(f"    Relations: {[(r['gene1'], r['relation'], r['gene2']) for r in rels[:3]]}")

    # Process actual PubMed articles if available
    pubmed_path = AMR_CONFIG["data_dir"] / "pubmed_amr_v2.json"
    if pubmed_path.exists():
        with open(pubmed_path) as f:
            articles = json.load(f)
        print(f"\n  Processing {min(len(articles), 50)} real PubMed articles with BioBERT...")
        result = extractor.process_articles(articles[:50])
        print(f"    Unique genes: {result['n_unique_genes']}")
        print(f"    Relations: {result['n_relations']}")
        print(f"    Top genes: {result['gene_counts'].most_common(10)}")

        # Save
        output = DATA_DIR / "biobert_results"
        output.mkdir(exist_ok=True)
        pd.DataFrame(result["gene_counts"].most_common(50), columns=["gene", "count"]).to_csv(
            output / "biobert_gene_counts.csv", index=False)


def run_temporal():
    """Run temporal validation."""
    print("\n" + "=" * 60)
    print("2/3 — TEMPORAL VALIDATION")
    print("=" * 60)

    from src.common.temporal_validation import TemporalValidator

    validator = TemporalValidator()

    # Curated validation (doesn't need internet)
    results = validator.validate_with_curated_data()

    # PubMed-based temporal split (needs internet)
    pubmed_path = AMR_CONFIG["data_dir"] / "pubmed_amr_v2.json"
    if pubmed_path.exists():
        with open(pubmed_path) as f:
            articles = json.load(f)
        if articles:
            validator.run_temporal_split(articles, cutoff_year=2020)


def run_sequence_toxicity():
    """Run sequence-based toxicity prediction."""
    print("\n" + "=" * 60)
    print("3/3 — SEQUENCE HOMOLOGY TOXICITY")
    print("=" * 60)

    from src.common.sequence_toxicity import SequenceToxicityPredictor

    predictor = SequenceToxicityPredictor()

    # Load our top AMR targets with UniProt IDs
    uniprot_path = AMR_CONFIG["data_dir"] / "essential_genes_uniprot.csv"
    if uniprot_path.exists():
        df = pd.read_csv(uniprot_path)
        # Deduplicate
        df = df.drop_duplicates(subset="gene_name", keep="first")

        targets = []
        for _, row in df.head(20).iterrows():
            targets.append({
                "gene_name": row["gene_name"],
                "uniprot_id": row.get("uniprot_id", ""),
            })

        results = predictor.batch_assess(targets)
    else:
        print("  No UniProt data, running with gene names only")
        targets = [{"gene_name": g} for g in
                   ["lpxC", "bamA", "ftsZ", "murA", "gyrA", "rpoB", "folA",
                    "fabI", "walK", "walR", "dxr", "secA", "accA", "lptD"]]
        results = predictor.batch_assess(targets)


def main():
    print("=" * 60)
    print("🔬 PROFESSIONAL UPGRADES — 3 Core Improvements")
    print("=" * 60)

    run_biobert()
    run_temporal()
    run_sequence_toxicity()

    print(f"\n{'='*60}")
    print("✅ ALL 3 UPGRADES COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
