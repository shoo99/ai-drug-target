#!/usr/bin/env python3
"""Run full CALMA-inspired pipeline: combinations + toxicity + transcriptomics + FAERS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import DATA_DIR, AMR_CONFIG


def run_combinations():
    """Run drug combination simulation."""
    from src.common.drug_combinations import DrugCombinationSimulator

    print("\n" + "=" * 60)
    print("1/4 — DRUG COMBINATION LANDSCAPE")
    print("=" * 60)

    sim = DrugCombinationSimulator()
    sim.load_model("iML1515")

    # Use our top novel AMR targets
    novel_targets = [
        "lpxC", "lpxA", "lpxB", "lpxD",   # LPS synthesis
        "bamA", "bamD",                      # Outer membrane
        "ftsZ", "ftsW",                      # Cell division
        "murA", "murB", "murC", "murD",      # Peptidoglycan
        "fabI", "fabH",                      # Fatty acid
        "folA", "folP",                      # Folate
        "dxr", "ispD",                       # Isoprenoid
        "accA",                              # Fatty acid (ACC)
    ]

    df = sim.generate_combination_landscape(novel_targets)

    if not df.empty:
        # Best synergistic combinations
        syn = df[df["interaction"] == "synergistic"].head(10)
        if not syn.empty:
            print(f"\n  🔥 TOP SYNERGISTIC COMBINATIONS:")
            for _, row in syn.iterrows():
                print(f"    {row['gene_a']:8s} + {row['gene_b']:8s} | "
                      f"Bliss: {row['bliss_score']:.4f} | "
                      f"Combo growth: {row['growth_ab']:.4f}")

    return df


def run_organ_toxicity():
    """Run organ-specific toxicity assessment."""
    from src.common.human_toxicity import HumanToxicityPredictor

    print("\n" + "=" * 60)
    print("2/4 — ORGAN-SPECIFIC TOXICITY")
    print("=" * 60)

    fba_path = DATA_DIR / "metabolic_analysis" / "metabolic_Escherichia_coli.csv"
    if not fba_path.exists():
        print("  No FBA results found")
        return pd.DataFrame()

    fba_df = pd.read_csv(fba_path)
    fba_valid = fba_df[fba_df["growth_ratio"].notna()]

    predictor = HumanToxicityPredictor()
    return predictor.batch_assess(fba_valid)


def run_transcriptomics():
    """Run transcriptomics-constrained simulations."""
    from src.common.transcriptomics import TranscriptomicsIntegrator

    print("\n" + "=" * 60)
    print("3/4 — TRANSCRIPTOMICS-CONSTRAINED FBA")
    print("=" * 60)

    integrator = TranscriptomicsIntegrator()

    # Search GEO (may fail if no internet)
    try:
        integrator.search_antibiotic_datasets()
    except Exception as e:
        print(f"  GEO search skipped: {e}")

    # Simulate antibiotic stress conditions
    return integrator.simulate_antibiotic_conditions()


def run_faers():
    """Run FAERS adverse event mining."""
    from src.common.faers_mining import FAERSMiner

    print("\n" + "=" * 60)
    print("4/4 — FAERS ADVERSE EVENT MINING")
    print("=" * 60)

    miner = FAERSMiner()

    # Individual target safety
    target_genes = ["gyrA", "rpoB", "folA", "folP", "fabI", "murA", "dxr",
                    "lpxC", "bamA", "ftsZ", "walK", "secA"]
    safety_df = miner.analyze_target_safety(target_genes)

    # Combination safety
    combo_df = miner.analyze_combination_safety()

    return safety_df, combo_df


def main():
    print("=" * 60)
    print("FULL CALMA-INSPIRED PIPELINE")
    print("=" * 60)

    combo_df = run_combinations()
    tox_df = run_organ_toxicity()
    trans_df = run_transcriptomics()

    try:
        safety_df, faers_combo_df = run_faers()
    except Exception as e:
        print(f"  FAERS failed: {e}")
        safety_df, faers_combo_df = pd.DataFrame(), pd.DataFrame()

    print(f"\n{'='*60}")
    print("✅ FULL CALMA PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Drug combinations analyzed: {len(combo_df) if not combo_df.empty else 0}")
    print(f"  Organ toxicity assessed: {len(tox_df) if not tox_df.empty else 0}")
    print(f"  Transcriptomics simulations: {len(trans_df) if not trans_df.empty else 0}")
    print(f"  FAERS safety records: {len(safety_df) if not safety_df.empty else 0}")
    print(f"  FAERS combinations: {len(faers_combo_df) if not faers_combo_df.empty else 0}")
    print(f"\n  All results: {DATA_DIR / 'metabolic_analysis'}")


if __name__ == "__main__":
    main()
