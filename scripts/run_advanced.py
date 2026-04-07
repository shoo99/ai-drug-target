#!/usr/bin/env python3
"""Run advanced analysis: Molecular Docking + Patents + Clinical Trials."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.common.molecular_docking import MolecularDockingPipeline
from src.common.patent_analyzer import PatentAnalyzer
from src.common.clinical_trials import ClinicalTrialsClient
from config.settings import AMR_CONFIG, PRURITUS_CONFIG


def get_top_targets(track_config, n=10):
    """Load top targets with UniProt IDs."""
    scored_path = track_config["data_dir"] / "top_targets_scored.csv"
    af_path = track_config["data_dir"] / "alphafold_druggability.csv"

    targets = []
    if scored_path.exists():
        scored = pd.read_csv(scored_path).head(n)
        af_data = {}
        if af_path.exists():
            af_df = pd.read_csv(af_path)
            af_data = {row["gene"]: row.to_dict() for _, row in af_df.iterrows()}

        for _, row in scored.iterrows():
            gene = row.get("gene_name", "")
            af_info = af_data.get(gene, {})
            targets.append({
                "gene_name": gene,
                "gene": gene,
                "uniprot_id": af_info.get("uniprot_id", ""),
                "composite_score": row.get("composite_score", 0),
            })
    return targets


def run_docking():
    """Run molecular docking analysis."""
    print("\n" + "🔬 " * 20)
    print("MOLECULAR DOCKING SIMULATION")
    print("🔬 " * 20)

    pipeline = MolecularDockingPipeline()

    # Get targets with UniProt IDs
    amr_targets = get_top_targets(AMR_CONFIG, n=10)
    pru_targets = get_top_targets(PRURITUS_CONFIG, n=10)

    # Filter to those with UniProt IDs
    all_targets = [t for t in amr_targets + pru_targets if t.get("uniprot_id")]
    print(f"\n  Targets with UniProt IDs: {len(all_targets)}")

    if all_targets:
        results = pipeline.run_full_analysis(all_targets)

        # Summary
        analyzed = [r for r in results if r["status"] == "analyzed"]
        print(f"\n  Successfully analyzed: {len(analyzed)}/{len(all_targets)}")
        for r in analyzed:
            print(f"    {r['gene']:15s} | Pockets: {r['n_pockets']} | "
                  f"Top score: {r['top_pocket_score']:.3f}")
    else:
        print("  No targets with UniProt IDs available for docking")


def run_patents():
    """Run patent landscape analysis."""
    analyzer = PatentAnalyzer()

    # AMR targets
    amr_targets = get_top_targets(AMR_CONFIG, n=15)
    if amr_targets:
        amr_patents = analyzer.batch_analyze(amr_targets, "antimicrobial resistance")

    # Pruritus targets
    pru_targets = get_top_targets(PRURITUS_CONFIG, n=15)
    if pru_targets:
        pru_patents = analyzer.batch_analyze(pru_targets, "pruritus itch")


def run_clinical_trials():
    """Run clinical trial analysis."""
    client = ClinicalTrialsClient()

    # AMR targets
    amr_targets = get_top_targets(AMR_CONFIG, n=15)
    if amr_targets:
        amr_trials = client.batch_analyze(amr_targets, "antimicrobial")

    # Pruritus targets
    pru_targets = get_top_targets(PRURITUS_CONFIG, n=15)
    if pru_targets:
        pru_trials = client.batch_analyze(pru_targets, "pruritus")


def main():
    print("=" * 60)
    print("ADVANCED ANALYSIS PIPELINE")
    print("=" * 60)

    run_docking()
    run_patents()
    run_clinical_trials()

    print("\n" + "=" * 60)
    print("✅ ALL ADVANCED ANALYSES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
