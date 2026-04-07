"""
TWOSIDES Database Client — Drug combination side effect data
TWOSIDES contains polypharmacy side effects for drug pairs,
derived from FAERS data using statistical methods.
This is the same toxicity data source used in the CALMA paper.
"""
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

TWOSIDES_DIR = DATA_DIR / "twosides"

# Known antibiotics with TWOSIDES/STITCH IDs
ANTIBIOTIC_MAP = {
    "ciprofloxacin": {"stitch": "CID000002764", "drugbank": "DB00537"},
    "levofloxacin": {"stitch": "CID000149096", "drugbank": "DB01137"},
    "vancomycin": {"stitch": "CID000014969", "drugbank": "DB00512"},
    "azithromycin": {"stitch": "CID000055185", "drugbank": "DB00207"},
    "trimethoprim": {"stitch": "CID000005578", "drugbank": "DB00440"},
    "isoniazid": {"stitch": "CID000003767", "drugbank": "DB00951"},
    "rifampin": {"stitch": "CID000005381", "drugbank": "DB01045"},
    "fosfomycin": {"stitch": "CID000092749", "drugbank": "DB00828"},
    "amoxicillin": {"stitch": "CID000033613", "drugbank": "DB01060"},
    "doxycycline": {"stitch": "CID000054671", "drugbank": "DB00254"},
    "gentamicin": {"stitch": "CID000003467", "drugbank": "DB00798"},
    "meropenem": {"stitch": "CID000441130", "drugbank": "DB00760"},
    "ceftriaxone": {"stitch": "CID000005479530", "drugbank": "DB01212"},
    "metronidazole": {"stitch": "CID000004173", "drugbank": "DB00916"},
    "sulfamethoxazole": {"stitch": "CID000005329", "drugbank": "DB01015"},
}

# Organ-specific adverse events (MedDRA preferred terms)
ORGAN_ADVERSE_EVENTS = {
    "kidney": [
        "Renal failure", "Renal failure acute", "Renal impairment",
        "Blood creatinine increased", "Nephrotoxicity", "Renal tubular necrosis",
        "Acute kidney injury", "Nephritis", "Oliguria", "Anuria",
    ],
    "liver": [
        "Hepatotoxicity", "Hepatic failure", "Hepatitis",
        "Alanine aminotransferase increased", "Aspartate aminotransferase increased",
        "Liver injury", "Jaundice", "Cholestasis", "Drug-induced liver injury",
    ],
    "heart": [
        "QT prolongation", "Cardiac arrest", "Torsade de pointes",
        "Arrhythmia", "Bradycardia", "Tachycardia", "Myocardial infarction",
    ],
    "gi": [
        "Diarrhoea", "Nausea", "Vomiting", "Clostridium difficile colitis",
        "Gastrointestinal haemorrhage", "Pancreatitis",
    ],
}


class TWOSIDESClient:
    """Access TWOSIDES drug combination side effect data."""

    def __init__(self):
        TWOSIDES_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_from_openfda(self, drug_a: str, drug_b: str,
                            organ: str = None) -> dict:
        """
        Fetch drug combination adverse events from openFDA.
        This replicates TWOSIDES-style analysis using live FDA data.
        """
        url = "https://api.fda.gov/drug/event.json"

        # Search for co-prescribed drugs
        search = (f'patient.drug.medicinalproduct:"{drug_a}" AND '
                  f'patient.drug.medicinalproduct:"{drug_b}"')

        # Count adverse events
        params = {
            "search": search,
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 100,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                total = data.get("meta", {}).get("results", {}).get("total", 0)

                # Categorize by organ
                organ_counts = {org: 0 for org in ORGAN_ADVERSE_EVENTS}
                organ_events = {org: [] for org in ORGAN_ADVERSE_EVENTS}

                for r in results:
                    event = r["term"]
                    count = r["count"]
                    for org, terms in ORGAN_ADVERSE_EVENTS.items():
                        if any(t.lower() in event.lower() for t in terms):
                            organ_counts[org] += count
                            organ_events[org].append({"event": event, "count": count})

                return {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "total_reports": total,
                    "total_adverse_events": len(results),
                    "organ_counts": organ_counts,
                    "organ_events": organ_events,
                    "top_events": [{"event": r["term"], "count": r["count"]}
                                   for r in results[:15]],
                }
            elif resp.status_code == 404:
                return {"drug_a": drug_a, "drug_b": drug_b, "total_reports": 0,
                        "organ_counts": {org: 0 for org in ORGAN_ADVERSE_EVENTS}}
        except Exception as e:
            return {"drug_a": drug_a, "drug_b": drug_b, "error": str(e), "total_reports": 0}

        return {"drug_a": drug_a, "drug_b": drug_b, "total_reports": 0}

    def compute_prr(self, drug_a: str, drug_b: str, event: str) -> float:
        """
        Compute Proportional Reporting Ratio (PRR) for a drug pair + event.
        PRR = (A/B) / (C/D) where:
        A = reports of event with drug combo
        B = all reports with drug combo
        C = reports of event without drug combo
        D = all reports without drug combo

        PRR > 2 suggests signal; PRR > 10 is strong signal.
        """
        url = "https://api.fda.gov/drug/event.json"

        try:
            # A: event with drug combo
            params_a = {
                "search": (f'patient.drug.medicinalproduct:"{drug_a}" AND '
                           f'patient.drug.medicinalproduct:"{drug_b}" AND '
                           f'patient.reaction.reactionmeddrapt:"{event}"'),
                "limit": 1,
            }
            resp_a = requests.get(url, params=params_a, timeout=10)
            a = resp_a.json().get("meta", {}).get("results", {}).get("total", 0) if resp_a.status_code == 200 else 0

            # B: all reports with drug combo
            params_b = {
                "search": (f'patient.drug.medicinalproduct:"{drug_a}" AND '
                           f'patient.drug.medicinalproduct:"{drug_b}"'),
                "limit": 1,
            }
            resp_b = requests.get(url, params=params_b, timeout=10)
            b = resp_b.json().get("meta", {}).get("results", {}).get("total", 0) if resp_b.status_code == 200 else 0

            if b == 0 or a == 0:
                return 0.0

            # Simplified PRR (using combo vs single drug ratio as proxy)
            params_c = {
                "search": f'patient.drug.medicinalproduct:"{drug_a}" AND patient.reaction.reactionmeddrapt:"{event}"',
                "limit": 1,
            }
            resp_c = requests.get(url, params=params_c, timeout=10)
            c = resp_c.json().get("meta", {}).get("results", {}).get("total", 0) if resp_c.status_code == 200 else 0

            params_d = {
                "search": f'patient.drug.medicinalproduct:"{drug_a}"',
                "limit": 1,
            }
            resp_d = requests.get(url, params=params_d, timeout=10)
            d = resp_d.json().get("meta", {}).get("results", {}).get("total", 0) if resp_d.status_code == 200 else 0

            if d == 0 or c == 0:
                return 0.0

            prr = (a / b) / (c / d)
            return round(prr, 3)

        except Exception:
            return 0.0

    def analyze_antibiotic_combinations(self) -> pd.DataFrame:
        """Analyze all antibiotic pair combinations from openFDA."""
        from itertools import combinations as combo_gen
        import time

        print(f"\n{'='*60}")
        print("TWOSIDES-STYLE ANALYSIS — Antibiotic Combinations")
        print(f"{'='*60}")

        antibiotics = list(ANTIBIOTIC_MAP.keys())
        pairs = list(combo_gen(antibiotics, 2))

        # Prioritize CALMA-relevant pairs
        priority_pairs = [
            ("azithromycin", "trimethoprim"),
            ("azithromycin", "vancomycin"),
            ("isoniazid", "trimethoprim"),
            ("ciprofloxacin", "vancomycin"),
            ("ciprofloxacin", "gentamicin"),
            ("vancomycin", "gentamicin"),
            ("meropenem", "vancomycin"),
            ("trimethoprim", "sulfamethoxazole"),
            ("amoxicillin", "gentamicin"),
            ("doxycycline", "rifampin"),
            ("vancomycin", "meropenem"),
            ("ciprofloxacin", "metronidazole"),
        ]

        results = []
        for drug_a, drug_b in tqdm(priority_pairs, desc="TWOSIDES analysis"):
            data = self.fetch_from_openfda(drug_a, drug_b)
            if data.get("total_reports", 0) > 0:
                organ_counts = data.get("organ_counts", {})
                results.append({
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "total_reports": data["total_reports"],
                    "kidney_events": organ_counts.get("kidney", 0),
                    "liver_events": organ_counts.get("liver", 0),
                    "heart_events": organ_counts.get("heart", 0),
                    "gi_events": organ_counts.get("gi", 0),
                    "top_event": data.get("top_events", [{}])[0].get("event", "N/A") if data.get("top_events") else "N/A",
                })
            else:
                results.append({
                    "drug_a": drug_a, "drug_b": drug_b,
                    "total_reports": 0,
                    "kidney_events": 0, "liver_events": 0,
                    "heart_events": 0, "gi_events": 0,
                    "top_event": "No data",
                })
            time.sleep(0.5)

        df = pd.DataFrame(results)
        output = TWOSIDES_DIR / "antibiotic_combo_safety.csv"
        df.to_csv(output, index=False)
        print(f"\n  Saved: {output}")

        # Summary
        if not df.empty:
            has_data = df[df["total_reports"] > 0]
            print(f"  Pairs with data: {len(has_data)}/{len(df)}")
            if not has_data.empty:
                safest = has_data.nsmallest(3, "kidney_events")
                print(f"\n  Safest for kidneys:")
                for _, r in safest.iterrows():
                    print(f"    {r['drug_a']} + {r['drug_b']}: {r['kidney_events']} kidney events")

        return df
