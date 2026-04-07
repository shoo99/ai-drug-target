"""
FAERS (FDA Adverse Event Reporting System) Mining
Analyze real-world drug adverse event data to validate combination safety.
Uses openFDA API for public access.
"""
import time
import requests
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

RESULTS_DIR = DATA_DIR / "metabolic_analysis"
OPENFDA_API = "https://api.fda.gov/drug/event.json"

# Map our target genes to known drugs that target them
GENE_TO_DRUGS = {
    "gyrA": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
    "gyrB": ["novobiocin"],
    "rpoB": ["rifampin", "rifampicin"],
    "folA": ["trimethoprim"],
    "folP": ["sulfamethoxazole"],
    "fabI": ["triclosan", "isoniazid"],
    "murA": ["fosfomycin"],
    "dxr": ["fosmidomycin"],
}

# Common antibiotics for combination analysis
COMMON_ANTIBIOTICS = [
    "vancomycin", "azithromycin", "trimethoprim", "ciprofloxacin",
    "levofloxacin", "amoxicillin", "doxycycline", "metronidazole",
    "ceftriaxone", "meropenem", "gentamicin", "rifampin",
]


class FAERSMiner:
    """Mine FDA Adverse Event Reporting System for drug safety signals."""

    def search_adverse_events(self, drug_name: str, limit: int = 100) -> dict:
        """Search FAERS for adverse events associated with a drug."""
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": limit,
        }
        try:
            resp = requests.get(OPENFDA_API, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                return {
                    "drug": drug_name,
                    "total_reports": data.get("meta", {}).get("results", {}).get("total", 0),
                    "top_adverse_events": [
                        {"event": r["term"], "count": r["count"]}
                        for r in results[:20]
                    ],
                }
            elif resp.status_code == 404:
                return {"drug": drug_name, "total_reports": 0, "top_adverse_events": []}
        except Exception as e:
            return {"drug": drug_name, "error": str(e), "total_reports": 0, "top_adverse_events": []}
        return {"drug": drug_name, "total_reports": 0, "top_adverse_events": []}

    def search_combination_events(self, drug_a: str, drug_b: str) -> dict:
        """Search for adverse events when two drugs are used together."""
        params = {
            "search": (f'patient.drug.medicinalproduct:"{drug_a}" AND '
                       f'patient.drug.medicinalproduct:"{drug_b}"'),
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 50,
        }
        try:
            resp = requests.get(OPENFDA_API, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])

                # Categorize by organ system using MedDRA preferred terms
                # Exact match against curated adverse event terms to reduce false positives
                KIDNEY_TERMS = {
                    "renal failure", "renal failure acute", "renal impairment",
                    "blood creatinine increased", "nephrotoxicity", "acute kidney injury",
                    "renal tubular necrosis", "nephritis interstitial", "oliguria",
                    "glomerulonephritis", "renal disorder", "chronic kidney disease",
                }
                LIVER_TERMS = {
                    "hepatotoxicity", "hepatic failure", "hepatitis",
                    "alanine aminotransferase increased", "aspartate aminotransferase increased",
                    "drug-induced liver injury", "jaundice", "cholestasis",
                    "liver function test abnormal", "hepatic enzyme increased",
                    "hepatocellular injury", "liver disorder",
                }
                kidney_events = [r for r in results
                                 if r["term"].lower() in KIDNEY_TERMS]
                liver_events = [r for r in results
                                if r["term"].lower() in LIVER_TERMS]

                return {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "total_combo_reports": data.get("meta", {}).get("results", {}).get("total", 0),
                    "top_events": [{"event": r["term"], "count": r["count"]} for r in results[:10]],
                    "kidney_events": sum(r["count"] for r in kidney_events),
                    "liver_events": sum(r["count"] for r in liver_events),
                    "n_kidney_types": len(kidney_events),
                    "n_liver_types": len(liver_events),
                }
            elif resp.status_code == 404:
                return {"drug_a": drug_a, "drug_b": drug_b, "total_combo_reports": 0}
        except Exception as e:
            return {"drug_a": drug_a, "drug_b": drug_b, "error": str(e), "total_combo_reports": 0}
        return {"drug_a": drug_a, "drug_b": drug_b, "total_combo_reports": 0}

    def analyze_target_safety(self, gene_list: list[str]) -> pd.DataFrame:
        """Analyze real-world safety data for drugs targeting our genes."""
        print(f"\n{'='*60}")
        print("FAERS ADVERSE EVENT ANALYSIS")
        print(f"{'='*60}")

        results = []
        for gene in tqdm(gene_list, desc="FAERS analysis"):
            drugs = GENE_TO_DRUGS.get(gene, [])
            if not drugs:
                results.append({
                    "gene": gene, "known_drug": "NONE (novel target)",
                    "total_reports": 0, "safety_note": "No existing drug — cannot assess from FAERS",
                })
                continue

            for drug in drugs:
                ae_data = self.search_adverse_events(drug)
                top_events = ae_data.get("top_adverse_events", [])

                results.append({
                    "gene": gene,
                    "known_drug": drug,
                    "total_reports": ae_data.get("total_reports", 0),
                    "top_event_1": top_events[0]["event"] if len(top_events) > 0 else "",
                    "top_event_1_count": top_events[0]["count"] if len(top_events) > 0 else 0,
                    "top_event_2": top_events[1]["event"] if len(top_events) > 1 else "",
                    "top_event_3": top_events[2]["event"] if len(top_events) > 2 else "",
                })
                time.sleep(0.5)

        df = pd.DataFrame(results)
        output = RESULTS_DIR / "faers_safety.csv"
        df.to_csv(output, index=False)
        print(f"  Saved: {output}")
        return df

    def analyze_combination_safety(self) -> pd.DataFrame:
        """Analyze safety of antibiotic combinations from FAERS."""
        print(f"\n  Analyzing antibiotic combination safety...")

        # Key combinations to test (from CALMA paper findings)
        combos_to_test = [
            ("azithromycin", "trimethoprim"),
            ("azithromycin", "vancomycin"),
            ("isoniazid", "trimethoprim"),
            ("ciprofloxacin", "vancomycin"),
            ("ciprofloxacin", "gentamicin"),
            ("meropenem", "vancomycin"),
            ("ceftriaxone", "metronidazole"),
            ("amoxicillin", "ciprofloxacin"),
            ("doxycycline", "rifampin"),
            ("levofloxacin", "metronidazole"),
        ]

        results = []
        for drug_a, drug_b in tqdm(combos_to_test, desc="Combination FAERS"):
            # Individual safety profiles
            ae_a = self.search_adverse_events(drug_a, limit=20)
            time.sleep(0.3)
            ae_b = self.search_adverse_events(drug_b, limit=20)
            time.sleep(0.3)

            # Combination
            combo = self.search_combination_events(drug_a, drug_b)
            time.sleep(0.3)

            # Compare: is combination worse than individuals?
            individual_max_reports = max(
                ae_a.get("total_reports", 0),
                ae_b.get("total_reports", 0)
            )
            combo_reports = combo.get("total_combo_reports", 0)

            results.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "reports_a": ae_a.get("total_reports", 0),
                "reports_b": ae_b.get("total_reports", 0),
                "reports_combo": combo_reports,
                "kidney_events_combo": combo.get("kidney_events", 0),
                "liver_events_combo": combo.get("liver_events", 0),
                "top_combo_event": combo.get("top_events", [{}])[0].get("event", "") if combo.get("top_events") else "",
            })

        df = pd.DataFrame(results)
        output = RESULTS_DIR / "faers_combinations.csv"
        df.to_csv(output, index=False)
        print(f"  Saved: {output}")

        # Print key findings
        if not df.empty:
            print(f"\n  Key findings:")
            for _, row in df.iterrows():
                if row["reports_combo"] > 0:
                    print(f"    {row['drug_a']} + {row['drug_b']}: "
                          f"{row['reports_combo']} combo reports | "
                          f"Kidney: {row['kidney_events_combo']} | "
                          f"Liver: {row['liver_events_combo']}")

        return df
