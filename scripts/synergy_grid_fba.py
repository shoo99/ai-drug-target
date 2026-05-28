#!/usr/bin/env python3
"""
Paper 2 — Phase 2: Partial Inhibition Synergy Grid

Instead of full knockout (binary results), use partial inhibition
to create dose-response surfaces and compute Bliss synergy.

Key insight: FBA is LP, so single-gene dose-response has a sharp threshold.
Cross-pathway combinations may have different thresholds, enabling synergy.

Strategy:
1. Find each gene's "critical inhibition threshold" (where growth drops)
2. Sample densely around the threshold (0.7-1.0 range, 15 levels)
3. Compute 15×15 grid for each cross-pathway pair
4. Calculate Bliss synergy at each grid point
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

# Pathway classification for cross-pathway pair selection
GENE_PATHWAYS = {
    "murA": "peptidoglycan", "murB": "peptidoglycan", "murC": "peptidoglycan",
    "murD": "peptidoglycan", "murE": "peptidoglycan", "murF": "peptidoglycan",
    "lpxA": "LPS", "lpxB": "LPS", "lpxC": "LPS", "lpxD": "LPS",
    "bamA": "outer_membrane", "bamD": "outer_membrane",
    "ftsZ": "cell_division", "ftsA": "cell_division", "ftsW": "cell_division",
    "gyrA": "DNA_topology", "gyrB": "DNA_topology",
    "rpoB": "transcription", "rpoC": "transcription",
    "fabI": "fatty_acid", "fabH": "fatty_acid", "accA": "fatty_acid",
    "folA": "folate", "folP": "folate",
    "dxr": "isoprenoid", "ispD": "isoprenoid",
    "walK": "signaling", "walR": "signaling",
    "dnaA": "DNA_replication", "dnaE": "DNA_replication", "dnaN": "DNA_replication",
    "lptD": "LPS_transport", "msbA": "LPS_transport",
    "secA": "secretion", "secY": "secretion",
    "infA": "translation_init", "infB": "translation_init",
    "rpsA": "ribosome", "rplB": "ribosome",
}

MODELS = {
    "iML1515": {"species": "E. coli", "file": "iML1515.json"},
    "iYS1720": {"species": "S. aureus", "file": "iYS1720.json"},
    "iYL1228": {"species": "K. pneumoniae", "file": "iYL1228.json"},
}

# Inhibition levels: dense near threshold (0.7-1.0)
INHIBITION_LEVELS = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.96, 1.0
])


def find_gene(model, name):
    name_lower = name.lower()
    for g in model.genes:
        if g.id.lower() == name_lower or g.name.lower() == name_lower:
            return g
        if name_lower in g.id.lower() or name_lower in g.name.lower():
            return g
    return None


def partial_inhibit(model, gene, level, wt_growth):
    """Single gene partial inhibition. Returns growth ratio."""
    if level <= 0.0:
        return 1.0
    with model as m:
        scale = 1.0 - level
        for rxn in gene.reactions:
            if level >= 0.99:
                rxn.upper_bound = 0
                rxn.lower_bound = 0
            else:
                if rxn.upper_bound > 0:
                    rxn.upper_bound *= scale
                if rxn.lower_bound < 0:
                    rxn.lower_bound *= scale
        sol = m.optimize()
        growth = sol.objective_value if sol.status == "optimal" else 0.0
    return growth / wt_growth if wt_growth > 0 else 0.0


def double_partial_inhibit(model, gene_a, gene_b, level_a, level_b, wt_growth):
    """Double gene partial inhibition. Returns growth ratio."""
    if level_a <= 0.0 and level_b <= 0.0:
        return 1.0
    with model as m:
        for gene, level in [(gene_a, level_a), (gene_b, level_b)]:
            if level <= 0.0:
                continue
            scale = 1.0 - level
            for rxn in gene.reactions:
                if level >= 0.99:
                    rxn.upper_bound = 0
                    rxn.lower_bound = 0
                else:
                    if rxn.upper_bound > 0:
                        rxn.upper_bound *= scale
                    if rxn.lower_bound < 0:
                        rxn.lower_bound *= scale
        sol = m.optimize()
        growth = sol.objective_value if sol.status == "optimal" else 0.0
    return growth / wt_growth if wt_growth > 0 else 0.0


def compute_single_dose_response(model, gene, wt_growth):
    """Compute dose-response curve for a single gene."""
    curve = {}
    for level in INHIBITION_LEVELS:
        gr = partial_inhibit(model, gene, level, wt_growth)
        curve[level] = max(0.0, gr)
    return curve


def find_threshold(curve):
    """Find the inhibition level where growth drops below 50%."""
    levels = sorted(curve.keys())
    for l in levels:
        if curve[l] < 0.5:
            return l
    return 1.0


def run_synergy_analysis(model_name, model_info):
    """Run complete synergy analysis for one model."""
    print(f"\n{'='*60}")
    print(f"  {model_name} ({model_info['species']})")
    print(f"{'='*60}")

    model = load_json_model(str(GEM_DIR / model_info["file"]))
    wt_sol = model.optimize()
    wt_growth = wt_sol.objective_value
    print(f"  WT growth: {wt_growth:.6f}")

    # Map genes
    mapped = {}
    for gname, pathway in GENE_PATHWAYS.items():
        g = find_gene(model, gname)
        if g:
            mapped[gname] = {"gene_obj": g, "pathway": pathway}

    print(f"  Mapped genes: {len(mapped)}/{len(GENE_PATHWAYS)}")

    # Phase 1: Single-gene dose-response curves
    print(f"\n  Phase 1: Single-gene dose-response ({len(mapped)} genes × {len(INHIBITION_LEVELS)} levels)...")
    dose_response = {}
    for gname, info in mapped.items():
        curve = compute_single_dose_response(model, info["gene_obj"], wt_growth)
        threshold = find_threshold(curve)
        dose_response[gname] = {"curve": curve, "threshold": threshold}

    # Report thresholds
    print(f"\n  Gene thresholds (inhibition level for 50% growth loss):")
    for gname in sorted(dose_response, key=lambda x: dose_response[x]["threshold"]):
        t = dose_response[gname]["threshold"]
        p = mapped[gname]["pathway"]
        print(f"    {gname:8s} ({p:15s}): threshold={t:.2f}")

    # Phase 2: Select cross-pathway pairs
    gene_names = list(mapped.keys())
    all_pairs = list(combinations(gene_names, 2))
    cross_pathway = [(a, b) for a, b in all_pairs
                     if mapped[a]["pathway"] != mapped[b]["pathway"]]
    same_pathway = [(a, b) for a, b in all_pairs
                    if mapped[a]["pathway"] == mapped[b]["pathway"]]

    print(f"\n  Total pairs: {len(all_pairs)}")
    print(f"  Cross-pathway: {len(cross_pathway)}")
    print(f"  Same-pathway: {len(same_pathway)}")

    # Phase 3: Synergy grid for ALL cross-pathway pairs
    # Use reduced grid focused near thresholds
    grid_results = []
    synergy_summary = []

    total = len(cross_pathway)
    print(f"\n  Phase 3: Synergy grid ({total} cross-pathway pairs × {len(INHIBITION_LEVELS)}² grid)...")

    for idx, (ga, gb) in enumerate(cross_pathway):
        gene_a = mapped[ga]["gene_obj"]
        gene_b = mapped[gb]["gene_obj"]
        curve_a = dose_response[ga]["curve"]
        curve_b = dose_response[gb]["curve"]

        pair_synergy_scores = []

        for la in INHIBITION_LEVELS:
            for lb in INHIBITION_LEVELS:
                if la == 0 and lb == 0:
                    continue

                # Observed combined effect
                gr_ab = double_partial_inhibit(model, gene_a, gene_b, la, lb, wt_growth)
                gr_ab = max(0.0, gr_ab)

                # Expected (Bliss independence): E_AB = E_A × E_B (growth ratios multiply)
                gr_a = curve_a.get(la, 1.0)
                gr_b = curve_b.get(lb, 1.0)
                expected_bliss = gr_a * gr_b

                # Synergy: observed < expected → synergistic (more inhibition than expected)
                bliss_excess = expected_bliss - gr_ab  # positive = synergy

                grid_results.append({
                    "species": model_info["species"],
                    "model": model_name,
                    "gene1": ga,
                    "gene2": gb,
                    "pathway1": mapped[ga]["pathway"],
                    "pathway2": mapped[gb]["pathway"],
                    "inhibition_A": round(la, 3),
                    "inhibition_B": round(lb, 3),
                    "growth_A_single": round(gr_a, 6),
                    "growth_B_single": round(gr_b, 6),
                    "growth_AB_observed": round(gr_ab, 6),
                    "growth_AB_bliss_expected": round(expected_bliss, 6),
                    "bliss_excess": round(bliss_excess, 6),
                })

                pair_synergy_scores.append(bliss_excess)

        # Summarize per pair
        scores = np.array(pair_synergy_scores)
        synergy_summary.append({
            "species": model_info["species"],
            "model": model_name,
            "gene1": ga,
            "gene2": gb,
            "pathway1": mapped[ga]["pathway"],
            "pathway2": mapped[gb]["pathway"],
            "threshold_A": dose_response[ga]["threshold"],
            "threshold_B": dose_response[gb]["threshold"],
            "mean_bliss_excess": round(scores.mean(), 6),
            "max_bliss_excess": round(scores.max(), 6),
            "min_bliss_excess": round(scores.min(), 6),
            "n_synergistic": int((scores > 0.05).sum()),
            "n_antagonistic": int((scores < -0.05).sum()),
            "n_grid_points": len(scores),
        })

        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{total} pairs done")

    # Also run same-pathway pairs (for comparison)
    print(f"\n  Phase 4: Same-pathway pairs ({len(same_pathway)} pairs)...")
    for ga, gb in same_pathway:
        gene_a = mapped[ga]["gene_obj"]
        gene_b = mapped[gb]["gene_obj"]
        curve_a = dose_response[ga]["curve"]
        curve_b = dose_response[gb]["curve"]

        pair_synergy_scores = []
        for la in INHIBITION_LEVELS:
            for lb in INHIBITION_LEVELS:
                if la == 0 and lb == 0:
                    continue
                gr_ab = double_partial_inhibit(model, gene_a, gene_b, la, lb, wt_growth)
                gr_ab = max(0.0, gr_ab)
                gr_a = curve_a.get(la, 1.0)
                gr_b = curve_b.get(lb, 1.0)
                expected_bliss = gr_a * gr_b
                bliss_excess = expected_bliss - gr_ab

                grid_results.append({
                    "species": model_info["species"],
                    "model": model_name,
                    "gene1": ga,
                    "gene2": gb,
                    "pathway1": mapped[ga]["pathway"],
                    "pathway2": mapped[gb]["pathway"],
                    "inhibition_A": round(la, 3),
                    "inhibition_B": round(lb, 3),
                    "growth_A_single": round(gr_a, 6),
                    "growth_B_single": round(gr_b, 6),
                    "growth_AB_observed": round(gr_ab, 6),
                    "growth_AB_bliss_expected": round(expected_bliss, 6),
                    "bliss_excess": round(bliss_excess, 6),
                })
                pair_synergy_scores.append(bliss_excess)

        scores = np.array(pair_synergy_scores)
        synergy_summary.append({
            "species": model_info["species"],
            "model": model_name,
            "gene1": ga,
            "gene2": gb,
            "pathway1": mapped[ga]["pathway"],
            "pathway2": mapped[gb]["pathway"],
            "threshold_A": dose_response[ga]["threshold"],
            "threshold_B": dose_response[gb]["threshold"],
            "mean_bliss_excess": round(scores.mean(), 6),
            "max_bliss_excess": round(scores.max(), 6),
            "min_bliss_excess": round(scores.min(), 6),
            "n_synergistic": int((scores > 0.05).sum()),
            "n_antagonistic": int((scores < -0.05).sum()),
            "n_grid_points": len(scores),
        })

    return grid_results, synergy_summary, dose_response


def main():
    print("=" * 60)
    print("  PAPER 2 — PARTIAL INHIBITION SYNERGY GRID")
    print("=" * 60)

    all_grid = []
    all_summary = []

    for model_name, model_info in MODELS.items():
        grid, summary, dr = run_synergy_analysis(model_name, model_info)
        all_grid.extend(grid)
        all_summary.extend(summary)

    # Save
    df_grid = pd.DataFrame(all_grid)
    df_summary = pd.DataFrame(all_summary)

    grid_path = OUT_DIR / "synergy_grid_full.csv"
    summary_path = OUT_DIR / "synergy_summary.csv"

    df_grid.to_csv(grid_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  Grid: {len(df_grid)} data points → {grid_path}")
    print(f"  Summary: {len(df_summary)} pairs → {summary_path}")

    # Top synergistic pairs
    print(f"\n  TOP 20 SYNERGISTIC PAIRS (by max Bliss excess):")
    top = df_summary.nlargest(20, "max_bliss_excess")
    for _, row in top.iterrows():
        print(f"    {row['gene1']:8s} ({row['pathway1']:15s}) × "
              f"{row['gene2']:8s} ({row['pathway2']:15s}) | "
              f"max_syn={row['max_bliss_excess']:.4f} "
              f"mean={row['mean_bliss_excess']:.4f} "
              f"[{row['species']}]")

    # Cross-pathway vs same-pathway comparison
    print(f"\n  CROSS-PATHWAY vs SAME-PATHWAY:")
    cross = df_summary[df_summary["pathway1"] != df_summary["pathway2"]]
    same = df_summary[df_summary["pathway1"] == df_summary["pathway2"]]
    if len(cross) > 0:
        print(f"    Cross-pathway: n={len(cross)}, "
              f"mean max_syn={cross['max_bliss_excess'].mean():.4f}")
    if len(same) > 0:
        print(f"    Same-pathway:  n={len(same)}, "
              f"mean max_syn={same['max_bliss_excess'].mean():.4f}")

    # Count truly synergistic pairs (max_bliss_excess > 0.05)
    truly_syn = df_summary[df_summary["max_bliss_excess"] > 0.05]
    print(f"\n  Pairs with any synergistic grid point (>0.05): {len(truly_syn)}/{len(df_summary)}")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
