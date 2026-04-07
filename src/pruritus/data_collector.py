"""
Pruritus Data Collector — Fetch chronic itch/pruritus data from public databases
Sources: OpenTargets, UniProt, ChEMBL, GWAS Catalog, PubMed
"""
import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config.settings import PRURITUS_CONFIG, CHEMBL_API

DATA_DIR = PRURITUS_CONFIG["data_dir"]
KNOWN_TARGETS = PRURITUS_CONFIG["known_targets"]
DISEASE_TERMS = PRURITUS_CONFIG["disease_terms"]


class PruritusDataCollector:

    def fetch_opentargets_data(self) -> pd.DataFrame:
        """Fetch pruritus-related targets from OpenTargets."""
        from src.common.opentargets import OpenTargetsClient
        print("[OpenTargets] Fetching pruritus-related targets...")
        client = OpenTargetsClient()

        # Search for relevant disease IDs
        disease_ids = {
            "HP_0000989": "pruritus",
            "EFO_0000274": "atopic dermatitis",
            "EFO_0004232": "eczema",
            "MONDO_0004980": "atopic dermatitis",
        }

        all_targets = []
        seen = set()
        for efo_id, name in disease_ids.items():
            try:
                targets = client.get_disease_targets(efo_id, size=300)
                for t in targets:
                    key = t["ensembl_id"]
                    if key not in seen:
                        seen.add(key)
                        t["source_disease"] = name
                        t["source_efo_id"] = efo_id
                        all_targets.append(t)
                print(f"  {name} ({efo_id}): {len(targets)} targets")
            except Exception as e:
                print(f"  {name}: error - {e}")
            time.sleep(0.5)

        df = pd.DataFrame(all_targets)
        if not df.empty:
            df.to_csv(DATA_DIR / "opentargets_pruritus.csv", index=False)
        print(f"[OpenTargets] Total unique targets: {len(df)}")
        return df

    def fetch_known_target_details(self) -> pd.DataFrame:
        """Fetch detailed info for known pruritus targets from UniProt."""
        print("[UniProt] Fetching known pruritus target details...")
        results = []

        for gene in tqdm(KNOWN_TARGETS, desc="Fetching target details"):
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": f'(gene:{gene}) AND (organism_id:9606)',
                "format": "json",
                "size": 1,
                "fields": "accession,gene_names,protein_name,cc_function,cc_subcellular_location,go_f,xref_pfam,structure_3d,cc_tissue_specificity",
            }
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                hits = data.get("results", [])
                if hits:
                    r = hits[0]
                    genes = r.get("genes", [{}])
                    gene_name = genes[0].get("geneName", {}).get("value", "") if genes else gene

                    # Extract function description
                    comments = r.get("comments", [])
                    function_text = ""
                    subcellular = ""
                    for c in comments:
                        if c.get("commentType") == "FUNCTION":
                            texts = c.get("texts", [])
                            function_text = texts[0].get("value", "") if texts else ""
                        if c.get("commentType") == "SUBCELLULAR LOCATION":
                            locs = c.get("subcellularLocations", [])
                            subcellular = ", ".join(
                                loc.get("location", {}).get("value", "")
                                for loc in locs[:3]
                            )

                    results.append({
                        "gene_symbol": gene_name,
                        "uniprot_id": r.get("primaryAccession", ""),
                        "protein_name": r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                        "function": function_text[:500],
                        "subcellular_location": subcellular,
                        "has_structure": bool(r.get("structure3D")),
                        "is_known_target": True,
                    })
            except Exception:
                pass
            time.sleep(0.5)

        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(DATA_DIR / "known_targets_details.csv", index=False)
        print(f"[UniProt] Fetched details for {len(df)} known targets")
        return df

    def fetch_gwas_data(self) -> pd.DataFrame:
        """Fetch GWAS associations for pruritus-related traits."""
        print("[GWAS] Fetching genetic associations...")
        base_url = "https://www.ebi.ac.uk/gwas/rest/api"

        traits = ["pruritus", "atopic dermatitis", "eczema", "itch", "dermatitis"]
        all_assocs = []

        for trait in traits:
            url = f"{base_url}/efoTraits/search/findBySearchTerm"
            params = {"searchTerm": trait, "page": 0, "size": 5}
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                efo_traits = data.get("_embedded", {}).get("efoTraits", [])

                for efo_trait in efo_traits:
                    trait_uri = efo_trait.get("shortForm", "")
                    trait_name = efo_trait.get("trait", "")

                    # Get associations for this trait
                    assoc_url = f"{base_url}/efoTraits/{trait_uri}/associations"
                    assoc_params = {"page": 0, "size": 100}
                    assoc_resp = requests.get(assoc_url, params=assoc_params, timeout=15)
                    if assoc_resp.status_code != 200:
                        continue

                    assoc_data = assoc_resp.json()
                    associations = assoc_data.get("_embedded", {}).get("associations", [])

                    for assoc in associations:
                        pvalue = assoc.get("pvalue", 1)
                        loci = assoc.get("loci", [])
                        for locus in loci:
                            genes_list = locus.get("authorReportedGenes", [])
                            for gene_entry in genes_list:
                                gene_name = gene_entry.get("geneName", "")
                                if gene_name:
                                    all_assocs.append({
                                        "gene": gene_name,
                                        "trait": trait_name,
                                        "efo_id": trait_uri,
                                        "pvalue": pvalue,
                                        "risk_allele": assoc.get("riskFrequency", ""),
                                    })

                print(f"  {trait}: found associations")
            except Exception as e:
                print(f"  {trait}: error - {e}")
            time.sleep(1)

        df = pd.DataFrame(all_assocs)
        if not df.empty:
            df.to_csv(DATA_DIR / "gwas_pruritus.csv", index=False)
        print(f"[GWAS] Total associations: {len(df)}")
        return df

    def fetch_pruritus_pubmed(self) -> list[dict]:
        """Fetch pruritus research papers from PubMed."""
        from src.common.pubmed_miner import PubMedMiner
        print("[PubMed] Fetching pruritus research papers...")
        miner = PubMedMiner()

        queries = [
            "chronic pruritus novel therapeutic target 2020:2026[dp]",
            "itch receptor drug target molecular mechanism",
            "pruritus treatment new pathway",
            "atopic dermatitis itch mediator target",
            "neuropathic itch molecular target",
            "MrgprX itch receptor",
        ]

        all_articles = []
        seen_pmids = set()
        for query in queries:
            articles = miner.search_and_fetch(query, max_results=200)
            for a in articles:
                if a["pmid"] not in seen_pmids:
                    seen_pmids.add(a["pmid"])
                    all_articles.append(a)

        print(f"[PubMed] Total unique articles: {len(all_articles)}")
        with open(DATA_DIR / "pubmed_pruritus_articles.json", "w") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        return all_articles

    def collect_all(self) -> dict:
        """Run all pruritus data collection."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        results = {}

        print("=" * 60)
        print("PRURITUS DATA COLLECTION — Starting")
        print("=" * 60)

        results["opentargets"] = self.fetch_opentargets_data()
        results["known_targets"] = self.fetch_known_target_details()
        results["gwas"] = self.fetch_gwas_data()
        results["pubmed"] = self.fetch_pruritus_pubmed()

        print("=" * 60)
        print("PRURITUS DATA COLLECTION — Complete")
        print(f"  OpenTargets targets: {len(results['opentargets'])}")
        print(f"  Known target details: {len(results['known_targets'])}")
        print(f"  GWAS associations: {len(results['gwas'])}")
        print(f"  PubMed articles: {len(results['pubmed'])}")
        print("=" * 60)

        return results
