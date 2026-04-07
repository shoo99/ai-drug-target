#!/usr/bin/env python3
"""Run CALMA-inspired metabolic analysis + metabolism-informed NN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from src.common.metabolic_analysis import MetabolicAnalyzer
from src.common.metabolism_nn import MetabolismNNTrainer
from src.amr.data_collector_v2 import CURATED_ESSENTIAL_GENES
from config.settings import AMR_CONFIG, DATA_DIR


def run_fba_analysis():
    """Run FBA gene knockout simulations."""
    analyzer = MetabolicAnalyzer()

    # Prepare targets from curated essential genes
    targets = []
    seen = set()
    for organism, genes in CURATED_ESSENTIAL_GENES.items():
        for gene_name, info in genes.items():
            if gene_name not in seen:
                targets.append({
                    "gene_name": gene_name,
                    "organism": organism,
                    "pathway": info["pathway"],
                })
                seen.add(gene_name)

    # Run for E. coli (most complete model)
    ecoli_targets = [t for t in targets]  # All genes, tested against E. coli model
    ecoli_results = analyzer.run_full_analysis(ecoli_targets, "Escherichia coli")

    # Run for S. aureus
    sa_targets = [t for t in targets if t["organism"] == "Staphylococcus aureus"]
    sa_results = analyzer.run_full_analysis(sa_targets, "Staphylococcus aureus")

    # Merge results
    all_results = pd.concat([ecoli_results, sa_results], ignore_index=True)
    all_results.to_csv(DATA_DIR / "metabolic_analysis" / "all_fba_results.csv", index=False)

    return all_results, ecoli_results


def run_metabolism_nn(fba_results: pd.DataFrame, ecoli_results: pd.DataFrame):
    """Train metabolism-informed neural network."""
    print(f"\n{'='*60}")
    print("METABOLISM-INFORMED NEURAL NETWORK")
    print(f"{'='*60}")

    # Use E. coli results for metabolic signatures
    analyzer = MetabolicAnalyzer()
    model = analyzer.load_model("Escherichia coli")

    # Get genes that were found in the model
    valid = ecoli_results[ecoli_results["growth_ratio"].notna()]
    gene_ids = valid["gene"].tolist()

    if len(gene_ids) < 5:
        print("  Not enough genes for NN training")
        return

    # Compute metabolic signatures
    signatures_df, subsystems = analyzer.compute_metabolic_signatures(model, gene_ids)

    # Merge with FBA results — use suffixes to avoid column name conflicts
    fba_cols = ["gene", "growth_ratio", "toxicity_score", "potency_score",
                "target_quality", "is_lethal", "toxicity_risk"]
    available_cols = [c for c in fba_cols if c in valid.columns]
    merged = signatures_df.merge(valid[available_cols], on="gene", how="inner",
                                  suffixes=("_sig", "_fba"))

    if len(merged) < 5:
        print("  Not enough merged data")
        return

    print(f"\n  Training data: {len(merged)} genes x {len(subsystems)} subsystems")

    # Train NN
    trainer = MetabolismNNTrainer()
    X, y = trainer.prepare_data(merged, subsystems)
    metrics = trainer.train(X, y, epochs=200)

    print(f"\n  Model Performance:")
    print(f"    Parameters: {metrics['n_params']} ({metrics['param_reduction']}% reduction)")
    print(f"    Potency R²: {metrics['cv_potency_r2']}")
    print(f"    Toxicity R²: {metrics['cv_toxicity_r2']}")

    # Get pathway importance
    print(f"\n  Pathway Importance (top 10):")
    importance = trainer.get_pathway_importance(X)
    for i, (pathway, scores) in enumerate(list(importance.items())[:10], 1):
        short_name = pathway[:40]
        print(f"    {i:2d}. {short_name:42s} | Potency: {scores['potency_impact']:.4f} | "
              f"Toxicity: {scores['toxicity_impact']:.4f}")

    # Save importance
    imp_df = pd.DataFrame([
        {"pathway": k, **v} for k, v in importance.items()
    ])
    imp_df.to_csv(DATA_DIR / "metabolic_analysis" / "pathway_importance.csv", index=False)

    # Generate final target rankings (potency * selectivity)
    predictions = trainer.predict(X)
    merged["nn_potency"] = predictions["predicted_potency"].values
    merged["nn_toxicity"] = predictions["predicted_toxicity"].values
    merged["nn_selectivity"] = predictions["predicted_selectivity"].values
    merged["nn_quality"] = merged["nn_potency"] * (1 - merged["nn_toxicity"])

    # Handle suffixed column names
    def get_col(df, base_name):
        if base_name in df.columns:
            return base_name
        suffixed = f"{base_name}_fba"
        if suffixed in df.columns:
            return suffixed
        matches = [c for c in df.columns if base_name in c]
        return matches[0] if matches else None

    gr_col = get_col(merged, "growth_ratio")
    lethal_col = get_col(merged, "is_lethal")
    pot_col = get_col(merged, "potency_score")
    tox_risk_col = get_col(merged, "toxicity_risk")
    tox_score_col = get_col(merged, "toxicity_score")
    tq_col = get_col(merged, "target_quality")

    final = pd.DataFrame({
        "gene": merged["gene"],
        "growth_ratio": merged[gr_col] if gr_col else 1.0,
        "is_lethal": merged[lethal_col] if lethal_col else False,
        "potency_score": merged[pot_col] if pot_col else 0,
        "toxicity_risk": merged[tox_risk_col] if tox_risk_col else "unknown",
        "toxicity_score": merged[tox_score_col] if tox_score_col else 0.5,
        "target_quality": merged[tq_col] if tq_col else 0,
        "nn_potency": merged["nn_potency"],
        "nn_toxicity": merged["nn_toxicity"],
        "nn_quality": merged["nn_quality"],
    })
    final = final.sort_values("nn_quality", ascending=False)
    final.to_csv(DATA_DIR / "metabolic_analysis" / "nn_ranked_targets.csv", index=False)

    print(f"\n{'='*60}")
    print("FINAL RANKINGS — Metabolism-Informed AI")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(final.head(15).iterrows(), 1):
        lethal = "💀 LETHAL" if row["is_lethal"] else "⚠️ Growth↓" if row["growth_ratio"] < 0.5 else "~ Viable"
        tox = str(row["toxicity_risk"])
        print(f"  {i:2d}. {row['gene']:12s} | {lethal:12s} | Tox: {tox:8s} | "
              f"NN Quality: {row['nn_quality']:.3f} | "
              f"FBA Quality: {row['target_quality']:.3f}")


def main():
    print("=" * 60)
    print("CALMA-INSPIRED METABOLIC ANALYSIS PIPELINE")
    print("=" * 60)

    all_results, ecoli_results = run_fba_analysis()
    run_metabolism_nn(all_results, ecoli_results)

    print(f"\n✅ Metabolic analysis complete!")
    print(f"   Results: {DATA_DIR / 'metabolic_analysis'}")


if __name__ == "__main__":
    main()
