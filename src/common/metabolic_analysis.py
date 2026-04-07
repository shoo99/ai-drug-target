"""
Metabolism-Informed Drug Target Analysis — CALMA-inspired approach
Integrates Genome-Scale Metabolic Models (GEMs) with neural networks
for predicting drug target potency and toxicity.

Inspired by: Arora et al., npj Drug Discovery (2026)
"A Metabolism-Informed Neural Network Identifies Pathways Influencing
the Potency and Toxicity of Antimicrobial Combinations"
"""
import json
import numpy as np
import pandas as pd
import cobra
from cobra.io import load_json_model
from cobra.flux_analysis import single_gene_deletion, flux_variability_analysis
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from config.settings import DATA_DIR

GEM_DIR = DATA_DIR / "gem_models"

# Map our organisms to available GEM models
ORGANISM_MODELS = {
    "Escherichia coli": "iML1515",
    "Staphylococcus aureus": "iYS1720",
    # Other ESKAPE can use E. coli model as proxy for gram-negatives
    "Klebsiella pneumoniae": "iML1515",
    "Pseudomonas aeruginosa": "iML1515",
    "Acinetobacter baumannii": "iML1515",
    "Enterobacter cloacae": "iML1515",
    "Enterococcus faecium": "iSB619",
}

# Human metabolic model subsystems for toxicity assessment
HUMAN_ESSENTIAL_PATHWAYS = {
    "oxidative_phosphorylation", "citric_acid_cycle", "glycolysis",
    "pentose_phosphate", "fatty_acid_oxidation", "amino_acid_metabolism",
    "nucleotide_metabolism", "urea_cycle",
}


class MetabolicAnalyzer:
    """Genome-scale metabolic analysis for drug target evaluation."""

    def __init__(self):
        self.models = {}
        self.results_dir = DATA_DIR / "metabolic_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, organism: str) -> cobra.Model:
        """Load GEM model for organism."""
        model_name = ORGANISM_MODELS.get(organism)
        if not model_name:
            return None

        if model_name not in self.models:
            path = GEM_DIR / f"{model_name}.json"
            if path.exists():
                self.models[model_name] = load_json_model(str(path))
            else:
                return None
        return self.models[model_name]

    def simulate_gene_knockout(self, model: cobra.Model, gene_id: str) -> dict:
        """Simulate single gene knockout and measure growth impact."""
        with model as m:
            # Find gene in model
            target_gene = None
            for gene in m.genes:
                if gene.id.lower() == gene_id.lower() or gene.name.lower() == gene_id.lower():
                    target_gene = gene
                    break

            if not target_gene:
                # Try partial match
                for gene in m.genes:
                    if gene_id.lower() in gene.id.lower() or gene_id.lower() in gene.name.lower():
                        target_gene = gene
                        break

            if not target_gene:
                return {"status": "gene_not_found", "gene": gene_id}

            # Wild-type growth
            wt_solution = m.optimize()
            wt_growth = wt_solution.objective_value

            # Knockout
            target_gene.knock_out()
            ko_solution = m.optimize()
            ko_growth = ko_solution.objective_value

            # Growth ratio
            growth_ratio = ko_growth / wt_growth if wt_growth > 0 else 0

            # Affected reactions
            affected_reactions = []
            for rxn in target_gene.reactions:
                affected_reactions.append({
                    "id": rxn.id,
                    "name": rxn.name,
                    "subsystem": rxn.subsystem or "unknown",
                })

            # Flux changes
            flux_changes = {}
            if wt_solution.status == "optimal" and ko_solution.status == "optimal":
                for rxn in m.reactions:
                    wt_flux = wt_solution.fluxes.get(rxn.id, 0)
                    ko_flux = ko_solution.fluxes.get(rxn.id, 0)
                    if abs(wt_flux) > 1e-6 or abs(ko_flux) > 1e-6:
                        change = abs(ko_flux - wt_flux) / (abs(wt_flux) + 1e-10)
                        if change > 0.1:  # >10% change
                            flux_changes[rxn.id] = {
                                "wt_flux": round(wt_flux, 4),
                                "ko_flux": round(ko_flux, 4),
                                "change_ratio": round(change, 4),
                                "subsystem": rxn.subsystem or "unknown",
                            }

            return {
                "status": "success",
                "gene": gene_id,
                "gene_model_id": target_gene.id,
                "wt_growth": round(wt_growth, 6),
                "ko_growth": round(ko_growth, 6),
                "growth_ratio": round(growth_ratio, 6),
                "is_lethal": growth_ratio < 0.01,
                "is_growth_reducing": growth_ratio < 0.5,
                "n_affected_reactions": len(affected_reactions),
                "affected_reactions": affected_reactions,
                "n_flux_changes": len(flux_changes),
                "affected_subsystems": list(set(
                    r["subsystem"] for r in affected_reactions
                )),
                "top_flux_changes": dict(
                    sorted(flux_changes.items(),
                           key=lambda x: x[1]["change_ratio"], reverse=True)[:10]
                ),
            }

    def compute_metabolic_signatures(self, model: cobra.Model,
                                      gene_ids: list[str]) -> pd.DataFrame:
        """Compute metabolic flux signatures for multiple gene knockouts.
        This generates the 'sigma/delta scores' concept from CALMA."""
        print(f"  Computing metabolic signatures for {len(gene_ids)} genes...")

        # Get subsystem list
        subsystems = sorted(set(
            rxn.subsystem for rxn in model.reactions if rxn.subsystem
        ))
        subsystem_to_idx = {s: i for i, s in enumerate(subsystems)}

        # Wild-type fluxes per subsystem
        wt_solution = model.optimize()
        wt_subsystem_flux = defaultdict(float)
        for rxn in model.reactions:
            if rxn.subsystem and rxn.id in wt_solution.fluxes.index:
                wt_subsystem_flux[rxn.subsystem] += abs(wt_solution.fluxes[rxn.id])

        signatures = []
        for gene_id in tqdm(gene_ids, desc="Metabolic signatures"):
            ko_result = self.simulate_gene_knockout(model, gene_id)

            if ko_result["status"] != "success":
                # Zero signature for unfound genes
                sig = {s: 0.0 for s in subsystems}
                sig["gene"] = gene_id
                sig["growth_ratio"] = 1.0
                sig["is_lethal"] = False
                signatures.append(sig)
                continue

            # Compute per-subsystem flux change (delta scores)
            sig = {}
            with model as m:
                target = None
                for g in m.genes:
                    if g.id.lower() == gene_id.lower() or g.name.lower() == gene_id.lower():
                        target = g
                        break
                    if gene_id.lower() in g.id.lower() or gene_id.lower() in g.name.lower():
                        target = g
                        break

                if target:
                    target.knock_out()
                    ko_sol = m.optimize()

                    ko_subsystem_flux = defaultdict(float)
                    if ko_sol.status == "optimal":
                        for rxn in m.reactions:
                            if rxn.subsystem and rxn.id in ko_sol.fluxes.index:
                                ko_subsystem_flux[rxn.subsystem] += abs(ko_sol.fluxes[rxn.id])

                    for subsystem in subsystems:
                        wt_val = wt_subsystem_flux.get(subsystem, 0)
                        ko_val = ko_subsystem_flux.get(subsystem, 0)
                        # Delta score: normalized flux change
                        delta = (ko_val - wt_val) / (wt_val + 1e-10)
                        sig[subsystem] = round(delta, 6)

            sig["gene"] = gene_id
            sig["growth_ratio"] = ko_result["growth_ratio"]
            sig["is_lethal"] = ko_result["is_lethal"]
            signatures.append(sig)

        df = pd.DataFrame(signatures)
        return df, subsystems

    def assess_toxicity_risk(self, affected_subsystems: list[str]) -> dict:
        """Assess toxicity risk based on overlap with human essential pathways."""
        # Map bacterial subsystems to human pathway categories
        subsystem_to_human = {
            "Oxidative phosphorylation": "oxidative_phosphorylation",
            "Citric acid cycle": "citric_acid_cycle",
            "Glycolysis/Gluconeogenesis": "glycolysis",
            "Pentose phosphate pathway": "pentose_phosphate",
            "Fatty acid oxidation": "fatty_acid_oxidation",
            "Purine metabolism": "nucleotide_metabolism",
            "Pyrimidine metabolism": "nucleotide_metabolism",
            "Nucleotide salvage pathway": "nucleotide_metabolism",
            "Amino acid metabolism": "amino_acid_metabolism",
            "Urea cycle": "urea_cycle",
        }

        human_overlap = set()
        for subsystem in affected_subsystems:
            for bact_name, human_name in subsystem_to_human.items():
                if bact_name.lower() in subsystem.lower():
                    human_overlap.add(human_name)

        overlap_ratio = len(human_overlap) / max(len(affected_subsystems), 1)

        if overlap_ratio > 0.5 or len(human_overlap) > 3:
            toxicity_risk = "high"
            toxicity_score = 0.8
        elif overlap_ratio > 0.2 or len(human_overlap) > 1:
            toxicity_risk = "moderate"
            toxicity_score = 0.5
        else:
            toxicity_risk = "low"
            toxicity_score = 0.2

        return {
            "toxicity_risk": toxicity_risk,
            "toxicity_score": round(toxicity_score, 3),
            "human_pathway_overlap": list(human_overlap),
            "n_overlapping_pathways": len(human_overlap),
            "selectivity": round(1 - toxicity_score, 3),
        }

    def run_full_analysis(self, targets: list[dict], organism: str) -> pd.DataFrame:
        """Run complete metabolic analysis for a set of drug targets."""
        model = self.load_model(organism)
        if model is None:
            print(f"  No GEM model for {organism}")
            return pd.DataFrame()

        model_name = ORGANISM_MODELS[organism]
        print(f"\n{'='*60}")
        print(f"METABOLIC ANALYSIS — {organism}")
        print(f"Model: {model_name} ({len(model.reactions)} rxns, {len(model.genes)} genes)")
        print(f"{'='*60}")

        results = []
        gene_ids = [t.get("gene_name", t.get("gene", "")) for t in targets]

        # Individual knockouts
        for gene_id in tqdm(gene_ids, desc="Gene knockouts"):
            ko = self.simulate_gene_knockout(model, gene_id)

            if ko["status"] == "success":
                toxicity = self.assess_toxicity_risk(ko["affected_subsystems"])

                results.append({
                    "gene": gene_id,
                    "organism": organism,
                    "model": model_name,
                    "wt_growth": ko["wt_growth"],
                    "ko_growth": ko["ko_growth"],
                    "growth_ratio": ko["growth_ratio"],
                    "is_lethal": ko["is_lethal"],
                    "is_growth_reducing": ko["is_growth_reducing"],
                    "n_affected_reactions": ko["n_affected_reactions"],
                    "affected_subsystems": ", ".join(ko["affected_subsystems"][:5]),
                    "n_flux_changes": ko["n_flux_changes"],
                    "toxicity_risk": toxicity["toxicity_risk"],
                    "toxicity_score": toxicity["toxicity_score"],
                    "selectivity": toxicity["selectivity"],
                    "human_overlap": ", ".join(toxicity["human_pathway_overlap"]),
                    # Composite: high potency + low toxicity = good target
                    "potency_score": round(1.0 - ko["growth_ratio"], 4),
                    "target_quality": round(
                        (1.0 - ko["growth_ratio"]) * toxicity["selectivity"], 4
                    ),
                })
            else:
                results.append({
                    "gene": gene_id,
                    "organism": organism,
                    "model": model_name,
                    "wt_growth": None,
                    "ko_growth": None,
                    "growth_ratio": None,
                    "is_lethal": None,
                    "n_affected_reactions": 0,
                    "affected_subsystems": "",
                    "toxicity_risk": "unknown",
                    "toxicity_score": None,
                    "selectivity": None,
                    "potency_score": None,
                    "target_quality": None,
                })

        df = pd.DataFrame(results)

        # Print summary
        successful = df[df["growth_ratio"].notna()]
        if not successful.empty:
            lethal = successful[successful["is_lethal"] == True]
            low_tox = successful[successful["toxicity_risk"] == "low"]

            print(f"\n  Results:")
            print(f"    Genes analyzed: {len(successful)}/{len(df)}")
            print(f"    Lethal knockouts: {len(lethal)}")
            print(f"    Low toxicity risk: {len(low_tox)}")

            # Best targets: lethal + low toxicity
            best = successful[
                (successful["is_lethal"] == True) &
                (successful["toxicity_risk"] == "low")
            ].sort_values("target_quality", ascending=False)

            if not best.empty:
                print(f"\n  ⭐ TOP TARGETS (Lethal + Low Toxicity):")
                for _, row in best.head(10).iterrows():
                    print(f"    {row['gene']:12s} | Kill: {row['potency_score']:.3f} | "
                          f"Selectivity: {row['selectivity']:.3f} | "
                          f"Quality: {row['target_quality']:.3f}")

        # Save
        output_path = self.results_dir / f"metabolic_{organism.replace(' ', '_')}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n  Saved: {output_path}")

        return df
