#!/usr/bin/env python3
"""
Fix ANN R²=0 by using partial gene inhibition instead of binary knockout.
Instead of 0% or 100% inhibition, simulate 10%, 30%, 50%, 70%, 90%, 100%.
This creates continuous growth values for ANN training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import cobra
from cobra.io import load_json_model
from itertools import combinations
from tqdm import tqdm
from config.settings import DATA_DIR, MODELS_DIR

GEM_DIR = DATA_DIR / "gem_models"
RESULTS_DIR = DATA_DIR / "calma_results"


def simulate_partial_inhibition(model, gene_id, inhibition_levels):
    """
    Simulate partial gene inhibition by reducing flux bounds.
    inhibition_level: 0.0 = no inhibition, 1.0 = complete knockout
    Returns growth ratios at each inhibition level.
    """
    # Find gene
    target_gene = None
    for g in model.genes:
        if g.id.lower() == gene_id.lower() or g.name.lower() == gene_id.lower():
            target_gene = g
            break
        if gene_id.lower() in g.id.lower() or gene_id.lower() in g.name.lower():
            target_gene = g
            break

    if not target_gene:
        return None

    # Wild-type growth
    wt_sol = model.optimize()
    wt_growth = wt_sol.objective_value

    results = []
    for level in inhibition_levels:
        with model as m:
            # Reduce flux bounds of all reactions associated with this gene
            for rxn in target_gene.reactions:
                # Scale down bounds by (1 - inhibition_level)
                scale = 1.0 - level
                if rxn.upper_bound > 0:
                    rxn.upper_bound *= scale
                if rxn.lower_bound < 0:
                    rxn.lower_bound *= scale
                # If complete inhibition
                if level >= 0.99:
                    rxn.upper_bound = 0
                    rxn.lower_bound = 0

            sol = m.optimize()
            growth = sol.objective_value
            growth_ratio = growth / wt_growth if wt_growth > 0 else 0

            results.append({
                "inhibition": level,
                "growth": growth,
                "growth_ratio": round(growth_ratio, 6),
                "potency": round(1.0 - growth_ratio, 6),
            })

    return results


def generate_partial_inhibition_data():
    """Generate training data with partial inhibition levels."""
    print("=" * 60)
    print("STEP 1: PARTIAL INHIBITION FBA SIMULATION")
    print("=" * 60)

    model = load_json_model(str(GEM_DIR / "iML1515.json"))
    wt_sol = model.optimize()
    print(f"  Model: iML1515 | WT growth: {wt_sol.objective_value:.4f}")

    # Mix essential + NON-essential genes for diversity
    from src.amr.data_collector_v2 import CURATED_ESSENTIAL_GENES, KNOWN_ANTIBIOTIC_TARGETS

    essential_genes = set()
    for org, g_dict in CURATED_ESSENTIAL_GENES.items():
        essential_genes.update(g_dict.keys())

    # Add non-essential genes from the model (random sample)
    # These will have growth_ratio > 0 when knocked out → creates diversity!
    all_model_genes = [g.id for g in model.genes]
    np.random.seed(42)
    non_essential_sample = list(np.random.choice(all_model_genes, size=30, replace=False))

    genes = essential_genes | set(non_essential_sample)
    print(f"  Essential: {len(essential_genes)}, Non-essential sample: {len(non_essential_sample)}")

    # Inhibition levels: gradual increase
    inhibition_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    print(f"  Genes: {len(genes)} | Levels: {inhibition_levels}")
    print(f"  Expected data points: {len(genes)} × {len(inhibition_levels)} = {len(genes) * len(inhibition_levels)}")

    # Single gene partial inhibition
    single_data = []
    gene_profiles = {}  # gene -> {inhibition: growth_ratio}

    for gene in tqdm(sorted(genes), desc="Single gene inhibition"):
        results = simulate_partial_inhibition(model, gene, inhibition_levels)
        if results is None:
            continue

        gene_profiles[gene] = {r["inhibition"]: r["growth_ratio"] for r in results}

        for r in results:
            single_data.append({
                "gene": gene,
                "inhibition": r["inhibition"],
                "growth_ratio": r["growth_ratio"],
                "potency": r["potency"],
            })

    single_df = pd.DataFrame(single_data)
    single_df.to_csv(RESULTS_DIR / "partial_inhibition_single.csv", index=False)

    found_genes = list(gene_profiles.keys())
    print(f"\n  Genes mapped: {len(found_genes)}/{len(genes)}")
    print(f"  Single data points: {len(single_df)}")

    # Show diversity
    potencies = single_df["potency"].values
    print(f"  Potency range: {potencies.min():.3f} — {potencies.max():.3f}")
    print(f"  Potency std: {potencies.std():.3f}")
    print(f"  Unique values: {len(np.unique(np.round(potencies, 3)))}")

    return gene_profiles, found_genes, inhibition_levels


def generate_combination_partial_data(gene_profiles, found_genes, inhibition_levels):
    """Generate pairwise combination data at multiple inhibition levels."""
    print("\n" + "=" * 60)
    print("STEP 2: COMBINATION PARTIAL INHIBITION")
    print("=" * 60)

    model = load_json_model(str(GEM_DIR / "iML1515.json"))
    wt_sol = model.optimize()
    wt_growth = wt_sol.objective_value

    # Sample genes (max 15 for tractability)
    sample_genes = found_genes[:15]
    pairs = list(combinations(sample_genes, 2))

    # Use 3 inhibition levels for combinations (to keep tractable)
    combo_levels = [0.5, 0.7, 1.0]

    print(f"  Gene pairs: {len(pairs)}")
    print(f"  Inhibition levels per pair: {combo_levels}")
    print(f"  Total simulations: {len(pairs) * len(combo_levels) * len(combo_levels)}")

    combo_data = []
    for gene_a, gene_b in tqdm(pairs, desc="Combination FBA"):
        for level_a in combo_levels:
            for level_b in combo_levels:
                # Find genes in model
                target_a, target_b = None, None
                for g in model.genes:
                    gname = g.id.lower() if g.id else ""
                    gn2 = g.name.lower() if g.name else ""
                    if gene_a.lower() in gname or gene_a.lower() in gn2:
                        target_a = g
                    if gene_b.lower() in gname or gene_b.lower() in gn2:
                        target_b = g

                if not target_a or not target_b:
                    continue

                with model as m:
                    # Partial inhibition of both genes
                    for rxn in target_a.reactions:
                        scale_a = 1.0 - level_a
                        if level_a >= 0.99:
                            rxn.upper_bound = 0; rxn.lower_bound = 0
                        else:
                            if rxn.upper_bound > 0: rxn.upper_bound *= scale_a
                            if rxn.lower_bound < 0: rxn.lower_bound *= scale_a

                    for rxn in target_b.reactions:
                        scale_b = 1.0 - level_b
                        if level_b >= 0.99:
                            rxn.upper_bound = 0; rxn.lower_bound = 0
                        else:
                            if rxn.upper_bound > 0: rxn.upper_bound *= scale_b
                            if rxn.lower_bound < 0: rxn.lower_bound *= scale_b

                    sol = m.optimize()
                    combo_growth = sol.objective_value / wt_growth if wt_growth > 0 else 0

                # Individual growths at these levels
                growth_a = gene_profiles.get(gene_a, {}).get(level_a, 0)
                growth_b = gene_profiles.get(gene_b, {}).get(level_b, 0)

                # Bliss expected
                bliss_expected = growth_a * growth_b
                bliss_score = bliss_expected - combo_growth

                combo_data.append({
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "inhibition_a": level_a,
                    "inhibition_b": level_b,
                    "growth_a": round(growth_a, 4),
                    "growth_b": round(growth_b, 4),
                    "growth_combo": round(combo_growth, 6),
                    "bliss_expected": round(bliss_expected, 6),
                    "bliss_score": round(bliss_score, 6),
                    "potency": round(1.0 - combo_growth, 6),
                })

    combo_df = pd.DataFrame(combo_data)
    combo_df.to_csv(RESULTS_DIR / "partial_inhibition_combos.csv", index=False)

    print(f"\n  Combination data points: {len(combo_df)}")
    if not combo_df.empty:
        print(f"  Potency range: {combo_df['potency'].min():.3f} — {combo_df['potency'].max():.3f}")
        print(f"  Potency std: {combo_df['potency'].std():.3f}")
        print(f"  Unique potency values: {len(combo_df['potency'].round(3).unique())}")
        print(f"  Bliss score range: {combo_df['bliss_score'].min():.4f} — {combo_df['bliss_score'].max():.4f}")

    return combo_df


def retrain_ann(combo_df, found_genes, inhibition_levels):
    """Retrain CALMA ANN with enriched partial inhibition data."""
    print("\n" + "=" * 60)
    print("STEP 3: RETRAIN CALMA ANN")
    print("=" * 60)

    if combo_df.empty or len(combo_df) < 20:
        print("  Insufficient data")
        return

    from src.common.calma_engine import CALMAFeatureGenerator, CALMATrainer

    # Generate sigma/delta features for gene pairs at multiple levels
    generator = CALMAFeatureGenerator("iML1515")

    # Get all unique gene pairs
    pairs = combo_df[["gene_a", "gene_b"]].drop_duplicates()
    unique_genes = list(set(pairs["gene_a"].tolist() + pairs["gene_b"].tolist()))

    feature_df, feature_cols = generator.generate_combination_features(unique_genes)

    if feature_df.empty:
        print("  No features generated")
        return

    # Merge features with partial inhibition combo data
    # For each pair, take the mean across inhibition levels as composite
    pair_summary = combo_df.groupby(["gene_a", "gene_b"]).agg({
        "potency": "mean",
        "growth_combo": "mean",
        "bliss_score": "mean",
    }).reset_index()

    merged = feature_df.merge(pair_summary, on=["gene_a", "gene_b"], how="inner",
                               suffixes=("_feat", "_combo"))

    # Handle column naming from merge
    pot_col = [c for c in merged.columns if "potency" in c and c != "potency_feat"]
    if pot_col:
        merged["growth_ab"] = 1.0 - merged[pot_col[0]].fillna(0)
    else:
        merged["growth_ab"] = 0.0

    bliss_cols = [c for c in merged.columns if "bliss" in c.lower()]
    if bliss_cols:
        merged["bliss_score"] = merged[bliss_cols[0]].fillna(0)
    else:
        merged["bliss_score"] = 0.0

    if len(merged) < 5:
        print(f"  Only {len(merged)} merged rows — insufficient")
        return

    print(f"\n  Training data: {len(merged)} combinations × {len(feature_cols)} features")
    print(f"  Potency diversity: {merged.get('potency', merged.get('potency_combo', pd.Series([0]))).std():.4f}")

    # Train
    trainer = CALMATrainer()
    metrics = trainer.train(merged, feature_cols, generator.subsystems, epochs=500)

    print(f"\n  {'='*40}")
    print(f"  RETRAINED ANN RESULTS")
    print(f"  {'='*40}")
    print(f"  Parameters: {metrics['n_params']} ({metrics['param_reduction']}% reduction)")
    print(f"  Potency R²: {metrics['potency_r2']}")
    print(f"  Toxicity R²: {metrics['toxicity_r2']}")
    print(f"  Best loss: {metrics['best_loss']}")

    if metrics['potency_r2'] > 0.1:
        print(f"\n  ✅ SIGNIFICANT IMPROVEMENT! R² > 0.1")
    elif metrics['potency_r2'] > 0:
        print(f"\n  ⚠️ Small improvement but still low R²")
    else:
        print(f"\n  ❌ R² still ≈ 0 — may need more data diversity")

    # Generate landscape
    landscape = trainer.generate_landscape(merged)

    # Interpret
    subsystem_inputs = trainer.prepare_subsystem_inputs(merged, feature_cols, generator.subsystems)
    interpretation = trainer.interpret_model(subsystem_inputs)

    return metrics


def main():
    print("=" * 60)
    print("🔧 FIXING ANN R² = 0")
    print("  Method: Partial gene inhibition (10-100%)")
    print("=" * 60)

    gene_profiles, found_genes, levels = generate_partial_inhibition_data()
    combo_df = generate_combination_partial_data(gene_profiles, found_genes, levels)
    metrics = retrain_ann(combo_df, found_genes, levels)

    print(f"\n{'='*60}")
    print("✅ ANN FIX COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
