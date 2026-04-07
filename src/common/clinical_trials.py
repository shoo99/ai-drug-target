"""
ClinicalTrials.gov Integration — Fetch and analyze clinical trial data
for drug target candidates
"""
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from config.settings import DATA_DIR

TRIALS_DIR = DATA_DIR / "common" / "clinical_trials"


class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov v2 API."""

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self):
        TRIALS_DIR.mkdir(parents=True, exist_ok=True)

    def search_trials(self, query: str, max_results: int = 50) -> list[dict]:
        """Search ClinicalTrials.gov for clinical trials matching query."""
        url = f"{self.BASE_URL}/studies"
        params = {
            "query.term": query,
            "pageSize": min(max_results, 100),
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,StartDate,"
                      "CompletionDate,Condition,InterventionName,"
                      "InterventionType,LeadSponsorName,EnrollmentCount",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            studies = data.get("studies", [])

            results = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                conditions_module = protocol.get("conditionsModule", {})
                interventions_module = protocol.get("armsInterventionsModule", {})
                sponsor_module = protocol.get("sponsorCollaboratorsModule", {})

                # Extract interventions
                interventions = []
                for inv in interventions_module.get("interventions", []):
                    interventions.append({
                        "name": inv.get("name", ""),
                        "type": inv.get("type", ""),
                    })

                # Extract phases
                phases = design_module.get("phases", [])

                results.append({
                    "nct_id": id_module.get("nctId", ""),
                    "title": id_module.get("briefTitle", ""),
                    "status": status_module.get("overallStatus", ""),
                    "phases": phases,
                    "phase_str": ", ".join(phases) if phases else "N/A",
                    "start_date": status_module.get("startDateStruct", {}).get("date", ""),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date", ""),
                    "conditions": conditions_module.get("conditions", []),
                    "interventions": interventions,
                    "intervention_names": [i["name"] for i in interventions],
                    "sponsor": sponsor_module.get("leadSponsor", {}).get("name", ""),
                    "enrollment": design_module.get("enrollmentInfo", {}).get("count"),
                })

            return results
        except Exception as e:
            print(f"  ClinicalTrials.gov error: {e}")
            return []

    def analyze_target_trials(self, gene_symbol: str,
                               disease_context: str = "") -> dict:
        """Comprehensive clinical trial analysis for a drug target."""
        queries = [
            f"{gene_symbol} drug",
            f"{gene_symbol} inhibitor",
            f"{gene_symbol} antibody",
        ]
        if disease_context:
            queries.append(f"{gene_symbol} {disease_context}")

        all_trials = []
        seen_ncts = set()

        for query in queries:
            trials = self.search_trials(query, max_results=20)
            for trial in trials:
                nct = trial.get("nct_id", "")
                if nct and nct not in seen_ncts:
                    seen_ncts.add(nct)
                    all_trials.append(trial)
            time.sleep(0.5)

        # Analyze
        active_statuses = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING"}
        completed_statuses = {"COMPLETED"}

        active_trials = [t for t in all_trials if t["status"] in active_statuses]
        completed_trials = [t for t in all_trials if t["status"] in completed_statuses]

        # Phase distribution
        phase_dist = {"PHASE1": 0, "PHASE2": 0, "PHASE3": 0, "PHASE4": 0}
        for trial in all_trials:
            for phase in trial.get("phases", []):
                if phase in phase_dist:
                    phase_dist[phase] += 1

        # Sponsors (competitors)
        sponsors = {}
        for trial in all_trials:
            sponsor = trial.get("sponsor", "Unknown")
            sponsors[sponsor] = sponsors.get(sponsor, 0) + 1
        top_sponsors = sorted(sponsors.items(), key=lambda x: x[1], reverse=True)[:5]

        # Competition assessment
        n_total = len(all_trials)
        if n_total > 15:
            competition = "very_high"
        elif n_total > 8:
            competition = "high"
        elif n_total > 3:
            competition = "moderate"
        elif n_total > 0:
            competition = "low"
        else:
            competition = "none"

        result = {
            "gene": gene_symbol,
            "disease_context": disease_context,
            "total_trials": n_total,
            "active_trials": len(active_trials),
            "completed_trials": len(completed_trials),
            "phase_distribution": phase_dist,
            "most_advanced_phase": self._get_most_advanced(all_trials),
            "competition_level": competition,
            "top_sponsors": top_sponsors,
            "n_unique_sponsors": len(sponsors),
            "trials_summary": [
                {
                    "nct_id": t["nct_id"],
                    "title": t["title"][:100],
                    "status": t["status"],
                    "phase": t["phase_str"],
                    "sponsor": t["sponsor"],
                }
                for t in all_trials[:10]
            ],
        }
        return result

    def _get_most_advanced(self, trials: list[dict]) -> str:
        phase_order = {"PHASE4": 4, "PHASE3": 3, "PHASE2": 2, "PHASE1": 1}
        max_phase = 0
        max_name = "None"
        for trial in trials:
            for phase in trial.get("phases", []):
                if phase in phase_order and phase_order[phase] > max_phase:
                    max_phase = phase_order[phase]
                    max_name = phase
        return max_name

    def batch_analyze(self, targets: list[dict],
                       disease_context: str = "") -> pd.DataFrame:
        """Analyze clinical trials for multiple targets."""
        print(f"\n{'='*60}")
        print(f"CLINICAL TRIAL ANALYSIS — {disease_context or 'All'}")
        print(f"{'='*60}")

        results = []
        for target in tqdm(targets, desc="Clinical trial analysis"):
            gene = target.get("gene_name", target.get("gene", ""))
            if not gene:
                continue

            analysis = self.analyze_target_trials(gene, disease_context)
            results.append(analysis)
            time.sleep(0.3)

            status = f"Total: {analysis['total_trials']} | Active: {analysis['active_trials']} | " \
                     f"Competition: {analysis['competition_level']}"
            print(f"  {gene:15s} | {status}")

        df = pd.DataFrame(results)
        if not df.empty:
            output_path = TRIALS_DIR / f"trials_{disease_context.replace(' ', '_') or 'all'}.csv"
            # Save simplified version
            save_cols = ["gene", "total_trials", "active_trials", "completed_trials",
                         "most_advanced_phase", "competition_level", "n_unique_sponsors"]
            df[save_cols].to_csv(output_path, index=False)

            # Save full data as JSON
            json_path = TRIALS_DIR / f"trials_{disease_context.replace(' ', '_') or 'all'}_full.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            print(f"\n  Results saved: {output_path}")

            # Summary
            no_trials = len(df[df["competition_level"] == "none"])
            low_comp = len(df[df["competition_level"] == "low"])
            print(f"\n  Summary:")
            print(f"    No existing trials: {no_trials} (blue ocean!)")
            print(f"    Low competition: {low_comp}")
            print(f"    High competition: {len(df[df['competition_level'].isin(['high', 'very_high'])])}")

        return df
