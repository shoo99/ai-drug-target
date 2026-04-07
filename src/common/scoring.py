"""
Target Scoring Framework — Multi-dimensional scoring for drug target candidates
"""
import numpy as np
from config.settings import SCORING_WEIGHTS


class TargetScorer:
    def __init__(self, weights: dict = None):
        self.weights = weights or SCORING_WEIGHTS

    def score_target(self, evidence: dict) -> dict:
        """
        Score a target candidate based on multiple evidence dimensions.

        evidence dict should contain:
        - genetic_evidence: float 0-1 (GWAS, essentiality, etc.)
        - expression_specificity: float 0-1 (disease-relevant tissue expression)
        - druggability: float 0-1 (tractability, binding pockets, etc.)
        - novelty: float 0-1 (1 = completely novel, 0 = well-known target)
        - competition: float 0-1 (1 = no competition, 0 = crowded)
        - literature_trend: float 0-1 (recent publication growth)
        """
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, weight in self.weights.items():
            value = evidence.get(dimension, 0.0)
            value = np.clip(value, 0.0, 1.0)
            scores[dimension] = value
            weighted_sum += value * weight
            total_weight += weight

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        scores["composite_score"] = round(composite, 4)

        # Classification
        if composite >= 0.7:
            scores["tier"] = "Tier 1 - High Priority"
        elif composite >= 0.5:
            scores["tier"] = "Tier 2 - Promising"
        elif composite >= 0.3:
            scores["tier"] = "Tier 3 - Exploratory"
        else:
            scores["tier"] = "Tier 4 - Low Priority"

        return scores

    def score_batch(self, targets: list[dict]) -> list[dict]:
        """Score multiple targets and return sorted by composite score."""
        scored = []
        for target in targets:
            gene_info = target.get("gene_info", {})
            evidence = target.get("evidence", {})
            result = self.score_target(evidence)
            result["gene_id"] = gene_info.get("gene_id", "")
            result["gene_name"] = gene_info.get("name", "")
            scored.append(result)
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored

    def calculate_genetic_evidence(self, gwas_pvalue: float = None,
                                    essentiality: float = None,
                                    disgenet_score: float = None) -> float:
        """Combine genetic evidence sources into single score."""
        scores = []
        if gwas_pvalue is not None and gwas_pvalue > 0:
            gwas_score = min(-np.log10(gwas_pvalue) / 20.0, 1.0)
            scores.append(gwas_score)
        if essentiality is not None:
            scores.append(essentiality)
        if disgenet_score is not None:
            scores.append(disgenet_score)
        return np.mean(scores) if scores else 0.0

    def calculate_druggability(self, has_structure: bool = False,
                                has_binding_pocket: bool = False,
                                protein_class: str = None,
                                tractability_bucket: int = None) -> float:
        """Estimate druggability from structural/functional features."""
        score = 0.0
        if has_structure:
            score += 0.3
        if has_binding_pocket:
            score += 0.3
        druggable_classes = {
            "kinase": 0.4, "gpcr": 0.4, "ion_channel": 0.35,
            "nuclear_receptor": 0.35, "protease": 0.3, "enzyme": 0.25,
        }
        if protein_class and protein_class.lower() in druggable_classes:
            score += druggable_classes[protein_class.lower()]
        if tractability_bucket is not None:
            score = max(score, 1.0 - (tractability_bucket - 1) * 0.1)
        return min(score, 1.0)

    def calculate_novelty(self, num_publications: int,
                           num_patents: int = 0,
                           num_clinical_trials: int = 0,
                           pub_baseline: int = 500,
                           patent_baseline: int = 20,
                           trial_baseline: int = 5) -> float:
        """Higher novelty for less-studied targets.
        Baselines are configurable per disease area (AMR vs pruritus etc.)"""
        pub_penalty = min(num_publications / max(pub_baseline, 1), 0.5)
        patent_penalty = min(num_patents / max(patent_baseline, 1), 0.3)
        trial_penalty = min(num_clinical_trials / max(trial_baseline, 1), 0.2)
        return max(1.0 - pub_penalty - patent_penalty - trial_penalty, 0.0)

    def calculate_competition(self, num_companies: int,
                                num_active_trials: int = 0,
                                num_patents_5yr: int = 0) -> float:
        """1.0 = no competition, 0.0 = highly competitive."""
        company_penalty = min(num_companies / 10.0, 0.4)
        trial_penalty = min(num_active_trials / 10.0, 0.3)
        patent_penalty = min(num_patents_5yr / 20.0, 0.3)
        return max(1.0 - company_penalty - trial_penalty - patent_penalty, 0.0)
