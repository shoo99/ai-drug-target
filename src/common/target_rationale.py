"""
LLM-powered Target Rationale Generator
Generates evidence-based explanations for why each drug target is promising.
Combines FBA data, scoring, clinical trials, sequence toxicity into human-readable rationale.
"""
import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

RESULTS_DIR = DATA_DIR / "rationales"

RATIONALE_PROMPT = """You are a drug discovery scientist writing a target assessment.
Based on the following data about a drug target, write a concise, professional rationale (3-5 sentences) explaining why this target is or isn't promising for drug development.

Be specific. Cite the data provided. Use scientific language appropriate for a pharma R&D audience.

Target Data:
{target_data}

Write the rationale:"""

COMBINATION_RATIONALE_PROMPT = """You are a drug discovery scientist. Based on the following data about a drug combination targeting two genes simultaneously, write a professional assessment (3-5 sentences) of this combination's potential.

Combination Data:
{combo_data}

Write the assessment:"""


class TargetRationaleGenerator:
    """Generate LLM-powered rationales for drug targets."""

    def __init__(self, ollama_url: str = None, ollama_model: str = None,
                 ollama_api_key: str = None):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "gemma4:e4b")
        self.ollama_api_key = ollama_api_key or os.getenv("OLLAMA_API_KEY", "")
        self.headers = {"X-API-Key": self.ollama_api_key}

    def _call_llm(self, prompt: str) -> str:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        }
        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[LLM error: {e}]"
        return "[LLM unavailable]"

    def _collect_target_data(self, gene_name: str) -> dict:
        """Collect all available data for a target from various analysis results."""
        data = {"gene": gene_name}

        # Scored targets
        for track in ["amr", "pruritus"]:
            scored_path = DATA_DIR / track / "top_targets_scored.csv"
            if scored_path.exists():
                df = pd.read_csv(scored_path)
                match = df[df["gene_name"] == gene_name]
                if not match.empty:
                    row = match.iloc[0]
                    data["composite_score"] = row.get("composite_score")
                    data["tier"] = row.get("tier")
                    data["is_novel"] = row.get("is_novel_target")
                    data["is_essential"] = row.get("is_essential")
                    data["pathway"] = row.get("pathway")
                    data["track"] = track

        # FBA results
        for org in ["Escherichia_coli", "Staphylococcus_aureus"]:
            fba_path = DATA_DIR / "metabolic_analysis" / f"metabolic_{org}.csv"
            if fba_path.exists():
                df = pd.read_csv(fba_path)
                match = df[df["gene"] == gene_name]
                if not match.empty:
                    row = match.iloc[0]
                    data["fba_organism"] = org.replace("_", " ")
                    data["fba_growth_ratio"] = row.get("growth_ratio")
                    data["fba_is_lethal"] = row.get("is_lethal")
                    data["fba_potency"] = row.get("potency_score")
                    data["fba_toxicity_risk"] = row.get("toxicity_risk")
                    data["fba_selectivity"] = row.get("selectivity")
                    data["fba_affected_subsystems"] = row.get("affected_subsystems")

        # Sequence toxicity
        seq_path = DATA_DIR / "sequence_toxicity" / "sequence_selectivity.csv"
        if seq_path.exists():
            df = pd.read_csv(seq_path)
            match = df[df["gene"] == gene_name]
            if not match.empty:
                row = match.iloc[0]
                data["human_homolog"] = row.get("human_gene", "none")
                data["sequence_identity"] = row.get("sequence_identity")
                data["selectivity_score"] = row.get("selectivity_score")
                data["toxicity_recommendation"] = row.get("recommendation")

        # Clinical trials
        for name in ["trials_antimicrobial.csv", "trials_pruritus.csv"]:
            ct_path = DATA_DIR / "common" / "clinical_trials" / name
            if ct_path.exists():
                df = pd.read_csv(ct_path)
                match = df[df["gene"] == gene_name]
                if not match.empty:
                    row = match.iloc[0]
                    data["clinical_trials_total"] = row.get("total_trials")
                    data["clinical_trials_active"] = row.get("active_trials")
                    data["competition_level"] = row.get("competition_level")

        # AlphaFold
        for track in ["amr", "pruritus"]:
            af_path = DATA_DIR / track / "alphafold_druggability.csv"
            if af_path.exists():
                df = pd.read_csv(af_path)
                match = df[df["gene"] == gene_name]
                if not match.empty:
                    row = match.iloc[0]
                    data["has_alphafold"] = row.get("has_alphafold")
                    data["druggability_score"] = row.get("druggability_score")
                    data["protein_class"] = row.get("protein_class")
                    data["modality_suggestion"] = row.get("modality_suggestion")

        return data

    def generate_rationale(self, gene_name: str) -> dict:
        """Generate a professional rationale for a single target."""
        target_data = self._collect_target_data(gene_name)

        # Format data for prompt
        data_str = json.dumps(target_data, indent=2, default=str)
        prompt = RATIONALE_PROMPT.format(target_data=data_str)

        rationale = self._call_llm(prompt)

        return {
            "gene": gene_name,
            "rationale": rationale,
            "data": target_data,
        }

    def generate_all_rationales(self, gene_list: list[str] = None) -> pd.DataFrame:
        """Generate rationales for all top targets."""
        print(f"\n{'='*60}")
        print("LLM TARGET RATIONALE GENERATION")
        print(f"{'='*60}")

        if gene_list is None:
            # Get top targets from scored files
            genes = set()
            for track in ["amr", "pruritus"]:
                scored_path = DATA_DIR / track / "top_targets_scored.csv"
                if scored_path.exists():
                    df = pd.read_csv(scored_path)
                    genes.update(df["gene_name"].head(15).tolist())
            gene_list = sorted(genes)

        print(f"  Generating rationales for {len(gene_list)} targets...")

        results = []
        for gene in tqdm(gene_list, desc="Rationale generation"):
            result = self.generate_rationale(gene)
            results.append({
                "gene": gene,
                "rationale": result["rationale"],
                "score": result["data"].get("composite_score"),
                "tier": result["data"].get("tier"),
                "track": result["data"].get("track"),
                "is_lethal": result["data"].get("fba_is_lethal"),
                "toxicity_risk": result["data"].get("fba_toxicity_risk"),
                "human_homolog": result["data"].get("human_homolog"),
                "competition": result["data"].get("competition_level"),
            })

            print(f"\n  {gene}:")
            print(f"  {result['rationale'][:200]}...")

            time.sleep(0.5)

        df = pd.DataFrame(results)
        output = RESULTS_DIR / "target_rationales.csv"
        df.to_csv(output, index=False)

        # Save full JSON
        with open(RESULTS_DIR / "target_rationales.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n  Saved: {output}")
        return df

    def generate_combination_rationales(self) -> pd.DataFrame:
        """Generate rationales for top drug combinations."""
        print(f"\n{'='*60}")
        print("LLM COMBINATION RATIONALE GENERATION")
        print(f"{'='*60}")

        landscape_path = DATA_DIR / "calma_results" / "calma_landscape.csv"
        if not landscape_path.exists():
            print("  No CALMA landscape data")
            return pd.DataFrame()

        df = pd.read_csv(landscape_path)

        # Top Pareto + ideal combinations
        if "pareto_optimal" in df.columns:
            top = df[df["pareto_optimal"] == True].head(5)
        else:
            top = df.nlargest(5, "calma_quality") if "calma_quality" in df.columns else df.head(5)

        results = []
        for _, row in tqdm(top.iterrows(), total=len(top), desc="Combo rationales"):
            combo_data = {
                "gene_a": row.get("gene_a"),
                "gene_b": row.get("gene_b"),
                "potency": row.get("calma_potency"),
                "toxicity": row.get("calma_toxicity"),
                "quality": row.get("calma_quality"),
                "interaction": row.get("interaction"),
                "bliss_score": row.get("bliss_score"),
                "quadrant": row.get("quadrant"),
                "pareto_optimal": row.get("pareto_optimal"),
            }

            # Add individual target data
            for gene_key in ["gene_a", "gene_b"]:
                gene = row.get(gene_key)
                if gene:
                    target_data = self._collect_target_data(gene)
                    combo_data[f"{gene_key}_fba_lethal"] = target_data.get("fba_is_lethal")
                    combo_data[f"{gene_key}_pathway"] = target_data.get("pathway")

            data_str = json.dumps(combo_data, indent=2, default=str)
            prompt = COMBINATION_RATIONALE_PROMPT.format(combo_data=data_str)
            rationale = self._call_llm(prompt)

            results.append({
                "gene_a": row.get("gene_a"),
                "gene_b": row.get("gene_b"),
                "rationale": rationale,
                **combo_data,
            })

            print(f"\n  {row.get('gene_a')} + {row.get('gene_b')}:")
            print(f"  {rationale[:200]}...")

            time.sleep(0.5)

        result_df = pd.DataFrame(results)
        output = RESULTS_DIR / "combination_rationales.csv"
        result_df.to_csv(output, index=False)
        print(f"\n  Saved: {output}")
        return result_df
