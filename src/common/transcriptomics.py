"""
Transcriptomics Integration — Integrate GEO expression data into GEM constraints.
Uses antibiotic-treated transcriptome data to make FBA simulations more realistic.
"""
import time
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

GEO_DIR = DATA_DIR / "transcriptomics"
RESULTS_DIR = DATA_DIR / "metabolic_analysis"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class TranscriptomicsIntegrator:
    """Integrate transcriptomics data from GEO into GEM models."""

    def __init__(self):
        GEO_DIR.mkdir(parents=True, exist_ok=True)

    def search_geo_datasets(self, query: str, max_results: int = 20) -> list[dict]:
        """Search GEO for relevant transcriptomics datasets."""
        url = f"{EUTILS_BASE}/esearch.fcgi"
        params = {
            "db": "gds",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                print(f"  Found {len(ids)} GEO datasets for: {query}")
                return ids
        except Exception as e:
            print(f"  GEO search error: {e}")
        return []

    def fetch_dataset_info(self, gds_ids: list[str]) -> list[dict]:
        """Fetch metadata for GEO datasets."""
        if not gds_ids:
            return []

        url = f"{EUTILS_BASE}/esummary.fcgi"
        params = {
            "db": "gds",
            "id": ",".join(gds_ids[:10]),
            "retmode": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for gds_id, info in data.get("result", {}).items():
                    if gds_id == "uids":
                        continue
                    results.append({
                        "gds_id": gds_id,
                        "accession": info.get("accession", ""),
                        "title": info.get("title", ""),
                        "summary": info.get("summary", "")[:200],
                        "organism": info.get("taxon", ""),
                        "n_samples": info.get("n_samples", 0),
                        "gse": info.get("gse", ""),
                    })
                return results
        except Exception as e:
            print(f"  GEO fetch error: {e}")
        return []

    def search_antibiotic_datasets(self) -> pd.DataFrame:
        """Search for antibiotic-treated transcriptomics datasets."""
        print(f"\n{'='*60}")
        print("TRANSCRIPTOMICS DATA SEARCH")
        print(f"{'='*60}")

        queries = [
            'antibiotic treatment transcriptome "Escherichia coli"',
            'antimicrobial stress response RNA-seq E. coli',
            'drug resistance transcriptome ESKAPE',
            'ciprofloxacin transcriptome bacteria',
            'vancomycin response transcriptome',
        ]

        all_datasets = []
        seen_ids = set()

        for query in queries:
            try:
                ids = self.search_geo_datasets(query)
                if ids:
                    new_ids = [i for i in ids if i not in seen_ids]
                    seen_ids.update(new_ids)
                    if new_ids:
                        infos = self.fetch_dataset_info(new_ids)
                        all_datasets.extend(infos)
                time.sleep(1)
            except Exception as e:
                print(f"  Query failed: {e}")
                continue

        df = pd.DataFrame(all_datasets)
        if not df.empty:
            df.to_csv(GEO_DIR / "antibiotic_datasets.csv", index=False)
            print(f"\n  Total datasets found: {len(df)}")
        return df

    def generate_gene_constraints(self, upregulated: list[str],
                                   downregulated: list[str],
                                   model) -> dict:
        """
        Generate GEM constraints from transcriptomics data.
        Upregulated genes → increase flux bounds
        Downregulated genes → decrease flux bounds
        """
        constraints = {}

        for gene in model.genes:
            gene_name = gene.name.lower() if gene.name else gene.id.lower()

            if any(g.lower() == gene_name for g in upregulated):
                # Upregulated: allow higher flux
                for rxn in gene.reactions:
                    constraints[rxn.id] = {"factor": 2.0, "direction": "up"}
            elif any(g.lower() == gene_name for g in downregulated):
                # Downregulated: reduce flux
                for rxn in gene.reactions:
                    constraints[rxn.id] = {"factor": 0.2, "direction": "down"}

        return constraints

    def apply_constraints_to_model(self, model, constraints: dict):
        """Apply transcriptomics-derived constraints to GEM model."""
        n_applied = 0
        for rxn_id, constraint in constraints.items():
            if rxn_id in model.reactions:
                rxn = model.reactions.get_by_id(rxn_id)
                factor = constraint["factor"]
                if constraint["direction"] == "up":
                    rxn.upper_bound *= factor
                else:
                    rxn.upper_bound *= factor
                    rxn.lower_bound *= factor
                n_applied += 1
        return n_applied

    def simulate_antibiotic_conditions(self) -> pd.DataFrame:
        """
        Simulate GEM under antibiotic stress conditions using
        known transcriptomic responses.
        """
        print(f"\n{'='*60}")
        print("TRANSCRIPTOMICS-CONSTRAINED FBA SIMULATION")
        print(f"{'='*60}")

        # Known antibiotic-induced gene expression changes in E. coli
        # (curated from literature)
        antibiotic_responses = {
            "ciprofloxacin": {
                "upregulated": ["recA", "lexA", "sulA", "umuC", "umuD",
                                "dinB", "recN", "ruvA", "ruvB"],
                "downregulated": ["ompF", "ompC", "ftsZ", "murA"],
                "mechanism": "SOS response, DNA damage repair",
            },
            "trimethoprim": {
                "upregulated": ["folA", "thyA", "purN", "purT"],
                "downregulated": ["folP", "folC"],
                "mechanism": "Folate pathway stress response",
            },
            "rifampin": {
                "upregulated": ["rpoB", "rpoC", "rpoA"],
                "downregulated": ["rrn operons"],
                "mechanism": "Transcription inhibition response",
            },
            "fosfomycin": {
                "upregulated": ["murA", "glpT", "uhpT"],
                "downregulated": [],
                "mechanism": "Cell wall stress, transporter upregulation",
            },
        }

        import cobra
        from cobra.io import load_json_model

        model_path = DATA_DIR / "gem_models" / "iML1515.json"
        if not model_path.exists():
            print("  iML1515 model not found")
            return pd.DataFrame()

        results = []
        for antibiotic, response in antibiotic_responses.items():
            print(f"\n  Simulating {antibiotic} stress...")

            model = load_json_model(str(model_path))

            # Wild-type
            wt_sol = model.optimize()
            wt_growth = wt_sol.objective_value

            # Apply constraints
            constraints = self.generate_gene_constraints(
                response["upregulated"],
                response["downregulated"],
                model
            )
            n_applied = self.apply_constraints_to_model(model, constraints)

            # Stressed growth
            stress_sol = model.optimize()
            stress_growth = stress_sol.objective_value

            growth_impact = stress_growth / wt_growth if wt_growth > 0 else 0

            results.append({
                "antibiotic": antibiotic,
                "mechanism": response["mechanism"],
                "n_upregulated": len(response["upregulated"]),
                "n_downregulated": len(response["downregulated"]),
                "n_constraints_applied": n_applied,
                "wt_growth": round(wt_growth, 4),
                "stress_growth": round(stress_growth, 4),
                "growth_impact": round(growth_impact, 4),
                "growth_reduction_pct": round((1 - growth_impact) * 100, 1),
            })

            print(f"    Constraints applied: {n_applied}")
            print(f"    Growth impact: {growth_impact:.4f} ({(1-growth_impact)*100:.1f}% reduction)")

        df = pd.DataFrame(results)
        output = RESULTS_DIR / "transcriptomics_fba.csv"
        df.to_csv(output, index=False)
        print(f"\n  Saved: {output}")
        return df
