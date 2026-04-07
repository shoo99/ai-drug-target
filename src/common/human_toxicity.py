"""
Human GEM Toxicity Model — Predict human cell toxicity using Recon3D metabolic model.
Compares bacterial drug target pathways against human essential metabolism.
"""
import numpy as np
import pandas as pd
import cobra
from cobra.io import load_json_model
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

GEM_DIR = DATA_DIR / "gem_models"
RESULTS_DIR = DATA_DIR / "metabolic_analysis"

# Human essential pathways — disruption causes toxicity
ORGAN_PATHWAY_MAP = {
    "kidney": {
        "Transport, extracellular", "Urea cycle", "Amino acid metabolism",
        "Oxidative phosphorylation", "Citric acid cycle",
        "Nucleotide salvage pathway", "Purine and Pyrimidine Biosynthesis",
    },
    "liver": {
        "Fatty acid oxidation", "Bile acid biosynthesis", "Cholesterol metabolism",
        "Drug metabolism", "Citric acid cycle", "Glycolysis/Gluconeogenesis",
        "Amino acid metabolism", "Urea cycle",
    },
    "heart": {
        "Oxidative phosphorylation", "Fatty acid oxidation", "Citric acid cycle",
        "Calcium signaling",
    },
    "neuron": {
        "Neurotransmitter metabolism", "Oxidative phosphorylation",
        "Glutamate Metabolism", "Glycolysis/Gluconeogenesis",
    },
}


class HumanToxicityPredictor:
    """Predict organ-specific toxicity by comparing bacterial target
    pathways against human essential metabolic pathways."""

    def __init__(self):
        self.human_model = None
        self.human_subsystems = set()
        self.human_subsystem_genes = defaultdict(set)

    def load_human_model(self):
        """Load human metabolic model or use pathway database."""
        # Try to load Recon3D if available
        recon_path = GEM_DIR / "Recon3D.json"
        if recon_path.exists():
            self.human_model = load_json_model(str(recon_path))
            self.human_subsystems = set(
                r.subsystem for r in self.human_model.reactions if r.subsystem
            )
            for rxn in self.human_model.reactions:
                if rxn.subsystem:
                    for gene in rxn.genes:
                        self.human_subsystem_genes[rxn.subsystem].add(gene.id)
            print(f"  Loaded Recon3D: {len(self.human_model.reactions)} rxns, "
                  f"{len(self.human_subsystems)} subsystems")
        else:
            # Use curated pathway knowledge
            print("  Using curated human pathway database (Recon3D not found)")
            self.human_subsystems = set()
            for organ_paths in ORGAN_PATHWAY_MAP.values():
                self.human_subsystems.update(organ_paths)

    def assess_organ_toxicity(self, affected_bacterial_subsystems: list[str]) -> dict:
        """Predict organ-specific toxicity based on pathway overlap."""
        organ_scores = {}

        for organ, human_pathways in ORGAN_PATHWAY_MAP.items():
            overlap = 0
            overlap_pathways = []

            # Curated bacterial→human pathway mapping (exact matches only)
            # Avoids false matches like "fatty acid synthesis" ↔ "fatty acid oxidation"
            CONSERVED_PATHWAY_MAP = {
                "oxidative phosphorylation": "oxidative_phosphorylation",
                "citric acid cycle": "citric_acid_cycle",
                "tca cycle": "citric_acid_cycle",
                "glycolysis": "glycolysis",
                "gluconeogenesis": "glycolysis",
                "pentose phosphate": "pentose_phosphate",
                "purine": "nucleotide_metabolism",
                "pyrimidine": "nucleotide_metabolism",
                "nucleotide salvage": "nucleotide_metabolism",
                "urea cycle": "urea_cycle",
                "amino acid": "amino_acid_metabolism",
                "alanine": "amino_acid_metabolism",
                "glutamate": "amino_acid_metabolism",
                "glycine": "amino_acid_metabolism",
                "valine": "amino_acid_metabolism",
                "tyrosine": "amino_acid_metabolism",
                "arginine": "amino_acid_metabolism",
            }

            for bact_sub in affected_bacterial_subsystems:
                bact_lower = bact_sub.lower()
                matched_human = None
                for bact_keyword, human_category in CONSERVED_PATHWAY_MAP.items():
                    if bact_keyword in bact_lower and human_category in human_pathways:
                        matched_human = human_category
                        break

                if matched_human:
                    overlap += 1
                    overlap_pathways.append(f"{bact_sub}↔{matched_human}")

            n_total = max(len(affected_bacterial_subsystems), 1)
            overlap_ratio = overlap / n_total

            if overlap_ratio > 0.4:
                risk = "high"
                score = 0.8 + overlap_ratio * 0.2
            elif overlap_ratio > 0.2:
                risk = "moderate"
                score = 0.4 + overlap_ratio
            elif overlap > 0:
                risk = "low"
                score = 0.2 + overlap_ratio * 0.5
            else:
                risk = "minimal"
                score = 0.05

            organ_scores[organ] = {
                "risk": risk,
                "score": round(min(score, 1.0), 3),
                "overlap_count": overlap,
                "overlap_ratio": round(overlap_ratio, 3),
                "overlap_pathways": overlap_pathways[:3],
            }

        # Overall toxicity
        max_score = max(o["score"] for o in organ_scores.values())
        overall_risk = "high" if max_score > 0.6 else "moderate" if max_score > 0.3 else "low"

        return {
            "overall_toxicity_score": round(max_score, 3),
            "overall_risk": overall_risk,
            "organ_scores": organ_scores,
            "safest_organs": [k for k, v in organ_scores.items() if v["risk"] == "minimal"],
            "at_risk_organs": [k for k, v in organ_scores.items() if v["risk"] in ("high", "moderate")],
        }

    def assess_combination_toxicity(self, affected_subsystems_a: list[str],
                                     affected_subsystems_b: list[str]) -> dict:
        """Assess toxicity of a drug combination."""
        combined = list(set(affected_subsystems_a + affected_subsystems_b))
        individual_a = self.assess_organ_toxicity(affected_subsystems_a)
        individual_b = self.assess_organ_toxicity(affected_subsystems_b)
        combination = self.assess_organ_toxicity(combined)

        # Check if combination is antagonistic for toxicity (good!)
        combo_tox = combination["overall_toxicity_score"]
        max_individual = max(individual_a["overall_toxicity_score"],
                            individual_b["overall_toxicity_score"])

        if combo_tox < max_individual * 0.8:
            tox_interaction = "toxicity_antagonistic"  # Good - combo less toxic
        elif combo_tox > max_individual * 1.2:
            tox_interaction = "toxicity_synergistic"   # Bad - combo more toxic
        else:
            tox_interaction = "toxicity_additive"

        return {
            "toxicity_drug_a": individual_a["overall_toxicity_score"],
            "toxicity_drug_b": individual_b["overall_toxicity_score"],
            "toxicity_combination": combo_tox,
            "toxicity_interaction": tox_interaction,
            "combination_organ_risks": combination["organ_scores"],
            "at_risk_organs": combination["at_risk_organs"],
        }

    def batch_assess(self, fba_results: pd.DataFrame) -> pd.DataFrame:
        """Assess organ toxicity for all FBA results."""
        print(f"\n{'='*60}")
        print("ORGAN-SPECIFIC TOXICITY ASSESSMENT")
        print(f"{'='*60}")

        self.load_human_model()
        results = []

        for _, row in tqdm(fba_results.iterrows(), total=len(fba_results), desc="Toxicity assessment"):
            gene = row.get("gene", "")
            subsystems_str = str(row.get("affected_subsystems", ""))
            subsystems = [s.strip() for s in subsystems_str.split(",") if s.strip()]

            if not subsystems:
                results.append({
                    "gene": gene,
                    "kidney_risk": "unknown", "kidney_score": 0,
                    "liver_risk": "unknown", "liver_score": 0,
                    "heart_risk": "unknown", "heart_score": 0,
                    "neuron_risk": "unknown", "neuron_score": 0,
                    "overall_toxicity": 0, "overall_risk": "unknown",
                })
                continue

            tox = self.assess_organ_toxicity(subsystems)

            result = {"gene": gene}
            for organ in ["kidney", "liver", "heart", "neuron"]:
                organ_data = tox["organ_scores"].get(organ, {})
                result[f"{organ}_risk"] = organ_data.get("risk", "unknown")
                result[f"{organ}_score"] = organ_data.get("score", 0)

            result["overall_toxicity"] = tox["overall_toxicity_score"]
            result["overall_risk"] = tox["overall_risk"]
            result["at_risk_organs"] = ", ".join(tox["at_risk_organs"])
            result["safe_organs"] = ", ".join(tox["safest_organs"])
            results.append(result)

        df = pd.DataFrame(results)
        output = RESULTS_DIR / "organ_toxicity.csv"
        df.to_csv(output, index=False)
        print(f"  Saved: {output}")

        # Summary
        if not df.empty:
            for organ in ["kidney", "liver", "heart", "neuron"]:
                col = f"{organ}_risk"
                if col in df.columns:
                    low = len(df[df[col].isin(["low", "minimal"])])
                    print(f"  {organ.capitalize()} safe targets: {low}/{len(df)}")

        return df
