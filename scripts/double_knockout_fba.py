#!/usr/bin/env python3
"""
Paper 2 — Phase 1: Double Knockout FBA Simulation Pipeline

Performs pairwise gene knockout simulations across 3 ESKAPE GEM models.
Includes both essential×essential and essential×non-essential pairs
to avoid the degeneracy problem (all-lethal pairs are indistinguishable).

Output: data/combination/double_ko_results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import cobra
from cobra.io import load_json_model
from itertools import combinations
from config.settings import DATA_DIR

GEM_DIR = DATA_DIR / "gem_models"
OUT_DIR = DATA_DIR / "combination"
OUT_DIR.mkdir(exist_ok=True)

# GEM models and their species
MODELS = {
    "iML1515": {"species": "E. coli", "file": "iML1515.json"},
    "iYS1720": {"species": "S. aureus", "file": "iYS1720.json"},
    "iYL1228": {"species": "K. pneumoniae", "file": "iYL1228.json"},
}

# 39 curated essential genes from Paper 1
ESSENTIAL_GENES = [
    "murA", "murB", "murC", "murD", "murE", "murF",
    "lpxA", "lpxB", "lpxC", "lpxD",
    "bamA", "bamD",
    "ftsZ", "ftsA", "ftsW",
    "gyrA", "gyrB",
    "rpoB", "rpoC",
    "fabI", "fabH", "accA",
    "folA", "folP",
    "dxr", "ispD",
    "walK", "walR",
    "dnaA", "dnaE", "dnaN",
    "lptD", "msbA",
    "secA", "secY",
    "infA", "infB",
    "rpsA", "rplB",
]


def find_gene_in_model(model, gene_name):
    """Find a gene in the model by name/id (case-insensitive, partial match)."""
    gene_lower = gene_name.lower()
    for g in model.genes:
        if g.id.lower() == gene_lower or g.name.lower() == gene_lower:
            return g
        if gene_lower in g.id.lower() or gene_lower in g.name.lower():
            return g
    return None


def single_knockout_fba(model, gene, wt_growth):
    """Perform single gene knockout and return growth ratio."""
    with model as m:
        for rxn in gene.reactions:
            rxn.upper_bound = 0
            rxn.lower_bound = 0
        sol = m.optimize()
        growth = sol.objective_value if sol.status == "optimal" else 0.0
    return growth / wt_growth if wt_growth > 0 else 0.0


def double_knockout_fba(model, gene_a, gene_b, wt_growth):
    """Perform double gene knockout and return growth ratio."""
    with model as m:
        for rxn in gene_a.reactions:
            rxn.upper_bound = 0
            rxn.lower_bound = 0
        for rxn in gene_b.reactions:
            rxn.upper_bound = 0
            rxn.lower_bound = 0
        sol = m.optimize()
        growth = sol.objective_value if sol.status == "optimal" else 0.0
    return growth / wt_growth if wt_growth > 0 else 0.0


def partial_double_inhibition(model, gene_a, gene_b, wt_growth,
                               level_a=0.5, level_b=0.5):
    """Partial inhibition of two genes simultaneously."""
    with model as m:
        scale_a = 1.0 - level_a
        scale_b = 1.0 - level_b
        for rxn in gene_a.reactions:
            if rxn.upper_bound > 0:
                rxn.upper_bound *= scale_a
            if rxn.lower_bound < 0:
                rxn.lower_bound *= scale_a
        for rxn in gene_b.reactions:
            if rxn.upper_bound > 0:
                rxn.upper_bound *= scale_b
            if rxn.lower_bound < 0:
                rxn.lower_bound *= scale_b
        sol = m.optimize()
        growth = sol.objective_value if sol.status == "optimal" else 0.0
    return growth / wt_growth if wt_growth > 0 else 0.0


def get_non_essential_genes(model, essential_mapped, n=30, seed=42):
    """Sample non-essential genes from the model (genes not in essential list)."""
    essential_ids = {g.id for g in essential_mapped}
    candidates = [g for g in model.genes if g.id not in essential_ids and len(g.reactions) > 0]
    np.random.seed(seed)
    if len(candidates) <= n:
        return candidates
    indices = np.random.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in indices]


def compute_bliss_synergy(fitness_a, fitness_b, fitness_ab):
    """
    Bliss independence synergy score.
    fitness = 1 - growth_ratio (inhibition fraction)
    S > 0 = synergy, S < 0 = antagonism
    """
    expected = fitness_a + fitness_b - (fitness_a * fitness_b)
    return fitness_ab - expected


def run_model(model_name, model_info):
    """Run all pairwise knockouts for one GEM model."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} ({model_info['species']})")
    print(f"{'='*60}")

    model = load_json_model(str(GEM_DIR / model_info["file"]))
    wt_sol = model.optimize()
    wt_growth = wt_sol.objective_value
    print(f"  WT growth: {wt_growth:.6f}")

    # Map essential genes
    essential_mapped = []
    essential_names = []
    for gname in ESSENTIAL_GENES:
        g = find_gene_in_model(model, gname)
        if g:
            essential_mapped.append(g)
            essential_names.append(gname)

    print(f"  Essential genes mapped: {len(essential_mapped)}/{len(ESSENTIAL_GENES)}")

    # Get non-essential genes
    non_essential = get_non_essential_genes(model, essential_mapped, n=30)
    ne_names = [f"NE_{g.id}" for g in non_essential]
    print(f"  Non-essential genes sampled: {len(non_essential)}")

    # Combined gene pool
    all_genes = essential_mapped + non_essential
    all_names = essential_names + ne_names
    all_types = ["essential"] * len(essential_mapped) + ["non-essential"] * len(non_essential)

    # Step 1: Single knockouts (baseline)
    print(f"\n  Phase 1: Single knockouts ({len(all_genes)} genes)...")
    single_ko = {}
    for i, (gene, name) in enumerate(zip(all_genes, all_names)):
        ratio = single_knockout_fba(model, gene, wt_growth)
        single_ko[name] = ratio
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(all_genes)} done")

    # Step 2: Double knockouts (all unique pairs)
    pairs = list(combinations(range(len(all_genes)), 2))
    total_pairs = len(pairs)
    print(f"\n  Phase 2: Double knockouts ({total_pairs} pairs)...")

    results = []
    for idx, (i, j) in enumerate(pairs):
        gene_a, gene_b = all_genes[i], all_genes[j]
        name_a, name_b = all_names[i], all_names[j]
        type_a, type_b = all_types[i], all_types[j]

        # Full double knockout
        gr_ab = double_knockout_fba(model, gene_a, gene_b, wt_growth)

        # Single KO values
        gr_a = single_ko[name_a]
        gr_b = single_ko[name_b]

        # Fitness (inhibition)
        fit_a = 1.0 - gr_a
        fit_b = 1.0 - gr_b
        fit_ab = 1.0 - gr_ab

        # Bliss synergy
        bliss = compute_bliss_synergy(fit_a, fit_b, fit_ab)

        # Classification
        if bliss > 0.10:
            interaction = "synergy"
        elif bliss < -0.10:
            interaction = "antagonism"
        else:
            interaction = "additive"

        # Pair type
        if type_a == "essential" and type_b == "essential":
            pair_type = "ess_ess"
        elif type_a == "non-essential" and type_b == "non-essential":
            pair_type = "ne_ne"
        else:
            pair_type = "ess_ne"

        results.append({
            "species": model_info["species"],
            "model": model_name,
            "gene1": name_a,
            "gene2": name_b,
            "type1": type_a,
            "type2": type_b,
            "pair_type": pair_type,
            "growth_wt": round(wt_growth, 6),
            "growth_KO_A": round(gr_a * wt_growth, 6),
            "growth_KO_B": round(gr_b * wt_growth, 6),
            "growth_KO_AB": round(gr_ab * wt_growth, 6),
            "growth_ratio_A": round(gr_a, 6),
            "growth_ratio_B": round(gr_b, 6),
            "growth_ratio_AB": round(gr_ab, 6),
            "fitness_A": round(fit_a, 6),
            "fitness_B": round(fit_b, 6),
            "fitness_AB": round(fit_ab, 6),
            "synergy_bliss": round(bliss, 6),
            "interaction": interaction,
        })

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{total_pairs} pairs done")

    print(f"  Completed: {len(results)} pairs")
    return results


def run_partial_inhibition_grid(model_name, model_info, top_pairs, n_levels=5):
    """
    Run partial inhibition grid (Loewe-style) for top synergistic pairs.
    Grid: n_levels × n_levels inhibition levels per pair.
    """
    print(f"\n  Phase 3: Partial inhibition grid for top {len(top_pairs)} pairs...")

    model = load_json_model(str(GEM_DIR / model_info["file"]))
    wt_sol = model.optimize()
    wt_growth = wt_sol.objective_value

    levels = np.linspace(0.1, 1.0, n_levels)  # 0.1, 0.325, 0.55, 0.775, 1.0
    grid_results = []

    for pair in top_pairs:
        gene_a_obj = find_gene_in_model(model, pair["gene1"])
        gene_b_obj = find_gene_in_model(model, pair["gene2"])
        if not gene_a_obj or not gene_b_obj:
            continue

        for la in levels:
            for lb in levels:
                gr = partial_double_inhibition(model, gene_a_obj, gene_b_obj,
                                                wt_growth, la, lb)
                grid_results.append({
                    "species": model_info["species"],
                    "model": model_name,
                    "gene1": pair["gene1"],
                    "gene2": pair["gene2"],
                    "inhibition_A": round(la, 3),
                    "inhibition_B": round(lb, 3),
                    "growth_ratio": round(gr, 6),
                    "fitness": round(1.0 - gr, 6),
                })

    print(f"  Grid simulations: {len(grid_results)}")
    return grid_results


def main():
    print("=" * 60)
    print("  PAPER 2 — DOUBLE KNOCKOUT FBA PIPELINE")
    print("  Pairwise gene knockout across 3 ESKAPE models")
    print("=" * 60)

    all_results = []
    all_grid_results = []

    for model_name, model_info in MODELS.items():
        # Phase 1+2: Single + Double knockouts
        results = run_model(model_name, model_info)
        all_results.extend(results)

        # Summarize interactions
        df = pd.DataFrame(results)
        print(f"\n  Interaction summary ({model_name}):")
        print(f"    Synergy:     {(df['interaction']=='synergy').sum()}")
        print(f"    Additive:    {(df['interaction']=='additive').sum()}")
        print(f"    Antagonism:  {(df['interaction']=='antagonism').sum()}")
        print(f"    By pair type:")
        for pt in ["ess_ess", "ess_ne", "ne_ne"]:
            sub = df[df["pair_type"] == pt]
            syn = (sub["interaction"] == "synergy").sum()
            print(f"      {pt}: {len(sub)} pairs, {syn} synergy")

        # Phase 3: Partial inhibition grid for top synergy pairs
        top_syn = df.nlargest(20, "synergy_bliss").to_dict("records")
        grid = run_partial_inhibition_grid(model_name, model_info, top_syn)
        all_grid_results.extend(grid)

    # Save results
    df_all = pd.DataFrame(all_results)
    out_path = OUT_DIR / "double_ko_results.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"  RESULTS SAVED: {out_path}")
    print(f"  Total pairs: {len(df_all)}")
    print(f"  Species: {df_all['species'].nunique()}")

    # Save grid results
    if all_grid_results:
        df_grid = pd.DataFrame(all_grid_results)
        grid_path = OUT_DIR / "partial_inhibition_grid.csv"
        df_grid.to_csv(grid_path, index=False)
        print(f"  Grid simulations: {len(df_grid)} → {grid_path}")

    # Summary statistics
    print(f"\n  OVERALL SUMMARY:")
    for species in df_all["species"].unique():
        sub = df_all[df_all["species"] == species]
        syn = (sub["interaction"] == "synergy").sum()
        ant = (sub["interaction"] == "antagonism").sum()
        add = (sub["interaction"] == "additive").sum()
        print(f"    {species}: {len(sub)} pairs | "
              f"synergy={syn} additive={add} antagonism={ant}")

    # Key finding: ess_ess vs ess_ne synergy comparison
    print(f"\n  KEY COMPARISON (Paper 2 hypothesis):")
    for pt in ["ess_ess", "ess_ne", "ne_ne"]:
        sub = df_all[df_all["pair_type"] == pt]
        if len(sub) > 0:
            mean_bliss = sub["synergy_bliss"].mean()
            std_bliss = sub["synergy_bliss"].std()
            syn_pct = (sub["interaction"] == "synergy").mean() * 100
            print(f"    {pt}: mean Bliss={mean_bliss:.4f} ± {std_bliss:.4f}, "
                  f"synergy rate={syn_pct:.1f}%")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
