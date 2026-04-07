#!/usr/bin/env python3
"""Run the full CALMA v2 pipeline with 4-feature sigma/delta + 3-layer ANN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.calma_engine import CALMAFeatureGenerator, CALMATrainer
from src.amr.data_collector_v2 import CURATED_ESSENTIAL_GENES


def main():
    print("=" * 60)
    print("CALMA V2 — FULL PIPELINE")
    print("=" * 60)

    # Get novel targets
    from src.amr.data_collector_v2 import KNOWN_ANTIBIOTIC_TARGETS
    novel_targets = []
    seen = set()
    for organism, genes in CURATED_ESSENTIAL_GENES.items():
        for gene_name in genes:
            if gene_name not in seen:
                existing_drugs = KNOWN_ANTIBIOTIC_TARGETS.get(gene_name, [])
                if len(existing_drugs) == 0:  # Novel target
                    novel_targets.append(gene_name)
                seen.add(gene_name)

    print(f"\n  Novel targets: {len(novel_targets)}")
    print(f"  Genes: {novel_targets}")

    # Step 1: Generate 4-feature sigma/delta
    print(f"\n{'='*60}")
    print("STEP 1: 4-Feature Sigma/Delta Generation")
    print(f"{'='*60}")

    generator = CALMAFeatureGenerator("iML1515")
    df, feature_cols = generator.generate_combination_features(novel_targets)

    if df.empty:
        print("  No data generated")
        return

    # Step 2: Train 3-layer ANN
    print(f"\n{'='*60}")
    print("STEP 2: 3-Layer Subsystem ANN Training")
    print(f"{'='*60}")

    trainer = CALMATrainer()
    metrics = trainer.train(df, feature_cols, generator.subsystems)

    # Step 3: Generate landscape
    print(f"\n{'='*60}")
    print("STEP 3: 2D Potency-Toxicity Landscape")
    print(f"{'='*60}")

    df = trainer.generate_landscape(df)

    # Step 4: Interpret
    subsystem_inputs = trainer.prepare_subsystem_inputs(df, feature_cols, generator.subsystems)
    interpretation = trainer.interpret_model(subsystem_inputs)

    # Step 5: Experiment design
    trainer.generate_experiment_design(df)

    # Final summary
    print(f"\n{'='*60}")
    print("CALMA V2 COMPLETE — SUMMARY")
    print(f"{'='*60}")

    if "quadrant" in df.columns:
        ideal = df[df["quadrant"].str.contains("IDEAL")]
        print(f"\n  🎯 TOP IDEAL COMBINATIONS (High Potency + Low Toxicity):")
        for _, row in ideal.nlargest(10, "calma_quality").iterrows():
            pareto = "⭐ PARETO" if row.get("pareto_optimal") else ""
            print(f"    {row['gene_a']:8s} + {row['gene_b']:8s} | "
                  f"Pot: {row['calma_potency']:.3f} | "
                  f"Tox: {row['calma_toxicity']:.3f} | "
                  f"Quality: {row['calma_quality']:.3f} {pareto}")

    print(f"\n  Results: data/calma_results/")


if __name__ == "__main__":
    main()
