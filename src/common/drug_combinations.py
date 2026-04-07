"""
Drug Combination Simulator — CALMA-inspired approach
Simulates multi-gene knockouts and computes sigma/delta interaction profiles
for predicting synergistic vs antagonistic drug combinations.
"""
import numpy as np
import pandas as pd
import cobra
from cobra.io import load_json_model
from itertools import combinations
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

GEM_DIR = DATA_DIR / "gem_models"
RESULTS_DIR = DATA_DIR / "metabolic_analysis"


class DrugCombinationSimulator:
    """Simulate drug combinations using double-gene knockouts in GEM models."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.wt_growth = None
        self.wt_fluxes = None
        self.subsystems = []
        self.wt_subsystem_flux = {}

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_name: str):
        """Load GEM model and compute wild-type reference."""
        path = GEM_DIR / f"{model_name}.json"
        self.model = load_json_model(str(path))

        # Wild-type solution
        wt_sol = self.model.optimize()
        self.wt_growth = wt_sol.objective_value
        self.wt_fluxes = wt_sol.fluxes

        # Subsystem fluxes
        self.subsystems = sorted(set(
            r.subsystem for r in self.model.reactions if r.subsystem
        ))
        self.wt_subsystem_flux = defaultdict(float)
        for rxn in self.model.reactions:
            if rxn.subsystem:
                self.wt_subsystem_flux[rxn.subsystem] += abs(self.wt_fluxes.get(rxn.id, 0))

        print(f"  Model loaded: {len(self.model.reactions)} rxns, "
              f"{len(self.model.genes)} genes, WT growth: {self.wt_growth:.4f}")

    def _find_gene(self, gene_id: str):
        """Find gene in model by name or ID."""
        for g in self.model.genes:
            if g.id.lower() == gene_id.lower() or g.name.lower() == gene_id.lower():
                return g
            if gene_id.lower() in g.id.lower() or gene_id.lower() in g.name.lower():
                return g
        return None

    def single_knockout(self, gene_id: str) -> dict:
        """Single gene knockout → growth + subsystem flux profile."""
        with self.model as m:
            gene = self._find_gene(gene_id)
            if not gene:
                return None

            gene.knock_out()
            sol = m.optimize()
            growth = sol.objective_value

            # Subsystem flux profile
            profile = {}
            for subsystem in self.subsystems:
                ko_flux = 0
                for rxn in m.reactions:
                    if rxn.subsystem == subsystem and rxn.id in sol.fluxes.index:
                        ko_flux += abs(sol.fluxes[rxn.id])
                wt_flux = self.wt_subsystem_flux.get(subsystem, 0)
                profile[subsystem] = (ko_flux - wt_flux) / (wt_flux + 1e-10)

            return {"growth": growth, "profile": profile, "gene_model_id": gene.id}

    def double_knockout(self, gene1: str, gene2: str) -> dict:
        """Double gene knockout → combined growth + flux profile."""
        with self.model as m:
            g1 = self._find_gene(gene1)
            g2 = self._find_gene(gene2)
            if not g1 or not g2:
                return None

            g1.knock_out()
            g2.knock_out()
            sol = m.optimize()
            growth = sol.objective_value

            profile = {}
            for subsystem in self.subsystems:
                ko_flux = 0
                for rxn in m.reactions:
                    if rxn.subsystem == subsystem and rxn.id in sol.fluxes.index:
                        ko_flux += abs(sol.fluxes[rxn.id])
                wt_flux = self.wt_subsystem_flux.get(subsystem, 0)
                profile[subsystem] = (ko_flux - wt_flux) / (wt_flux + 1e-10)

            return {"growth": growth, "profile": profile}

    def compute_sigma_delta(self, profile_a: dict, profile_b: dict) -> dict:
        """
        Compute CALMA-style sigma (shared effect) and delta (unique effect) scores.
        Sigma: average of two drug effects (shared pathway disruption)
        Delta: difference of two drug effects (unique pathway disruption)
        """
        sigma = {}
        delta = {}
        for subsystem in self.subsystems:
            a = profile_a.get(subsystem, 0)
            b = profile_b.get(subsystem, 0)
            sigma[subsystem] = (a + b) / 2  # Shared effect
            delta[subsystem] = abs(a - b)    # Unique effect

        return {"sigma": sigma, "delta": delta}

    def compute_synergy_score(self, growth_a: float, growth_b: float,
                               growth_ab: float) -> dict:
        """
        Compute Bliss independence synergy score.
        Bliss expected = growth_a * growth_b / wt_growth
        If actual < expected → synergistic (combo kills more than expected)
        If actual > expected → antagonistic (combo kills less than expected)

        NOTE: This is a potency synergy score, NOT a safety score.
        Antagonistic for potency ≠ safe. A separate toxicity assessment is needed.
        For clinical safety, see human_toxicity.py and faers_mining.py.
        """
        ratio_a = growth_a / self.wt_growth if self.wt_growth > 0 else 1
        ratio_b = growth_b / self.wt_growth if self.wt_growth > 0 else 1
        ratio_ab = growth_ab / self.wt_growth if self.wt_growth > 0 else 1

        bliss_expected = ratio_a * ratio_b
        bliss_score = bliss_expected - ratio_ab  # positive = synergistic

        if bliss_score > 0.1:
            interaction = "synergistic"
        elif bliss_score < -0.1:
            interaction = "antagonistic"
        else:
            interaction = "additive"

        return {
            "growth_a": round(ratio_a, 4),
            "growth_b": round(ratio_b, 4),
            "growth_ab": round(ratio_ab, 4),
            "bliss_expected": round(bliss_expected, 4),
            "bliss_score": round(bliss_score, 4),
            "interaction": interaction,
        }

    def generate_combination_landscape(self, gene_list: list[str]) -> pd.DataFrame:
        """Generate full combination landscape for a set of target genes."""
        print(f"\n  Generating combination landscape for {len(gene_list)} genes...")
        print(f"  Total combinations: {len(gene_list) * (len(gene_list)-1) // 2}")

        # Single knockouts first
        singles = {}
        for gene in tqdm(gene_list, desc="Single KOs"):
            result = self.single_knockout(gene)
            if result:
                singles[gene] = result

        found_genes = list(singles.keys())
        print(f"  Genes found in model: {len(found_genes)}/{len(gene_list)}")

        if len(found_genes) < 2:
            return pd.DataFrame()

        # Double knockouts
        results = []
        pairs = list(combinations(found_genes, 2))

        for gene_a, gene_b in tqdm(pairs, desc="Double KOs"):
            double = self.double_knockout(gene_a, gene_b)
            if not double:
                continue

            # Synergy
            synergy = self.compute_synergy_score(
                singles[gene_a]["growth"],
                singles[gene_b]["growth"],
                double["growth"]
            )

            # Sigma/Delta
            sd = self.compute_sigma_delta(
                singles[gene_a]["profile"],
                singles[gene_b]["profile"]
            )

            # Top sigma/delta pathways
            top_sigma = sorted(sd["sigma"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            top_delta = sorted(sd["delta"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]

            results.append({
                "gene_a": gene_a,
                "gene_b": gene_b,
                **synergy,
                "top_shared_pathways": "; ".join(f"{p[0][:30]}({p[1]:.3f})" for p in top_sigma),
                "top_unique_pathways": "; ".join(f"{p[0][:30]}({p[1]:.3f})" for p in top_delta),
                "mean_sigma": round(np.mean(list(sd["sigma"].values())), 4),
                "mean_delta": round(np.mean(list(sd["delta"].values())), 4),
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("bliss_score", ascending=False)
            output = RESULTS_DIR / "drug_combinations.csv"
            df.to_csv(output, index=False)
            print(f"\n  Saved: {output}")

            synergistic = len(df[df["interaction"] == "synergistic"])
            antagonistic = len(df[df["interaction"] == "antagonistic"])
            additive = len(df[df["interaction"] == "additive"])
            print(f"  Synergistic: {synergistic} | Additive: {additive} | Antagonistic: {antagonistic}")

        return df
