"""
AMR Data Collector — Fetch antimicrobial resistance data from public databases
Sources: CARD, DEG, UniProt, ChEMBL, OpenTargets, PubMed
"""
import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config.settings import AMR_CONFIG, CHEMBL_API

DATA_DIR = AMR_CONFIG["data_dir"]
ESKAPE = AMR_CONFIG["eskape_organisms"]


class AMRDataCollector:

    # --- CARD (Comprehensive Antibiotic Resistance Database) ---

    def fetch_card_ontology(self) -> pd.DataFrame:
        """Fetch CARD antibiotic resistance ontology (ARO) data."""
        print("[CARD] Fetching ARO ontology...")
        url = "https://card.mcmaster.ca/download/0/broadstreet-v3.3.0.tar.bz2"
        # CARD bulk download is large; use their API for key data instead
        # Fetch resistance genes via CARD's REST-like endpoints
        aro_url = "https://card.mcmaster.ca/api/v1/ontology"
        try:
            resp = requests.get(aro_url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[CARD] Fetched ontology data")
                return data
        except Exception as e:
            print(f"[CARD] API failed ({e}), using fallback approach")
        return None

    def fetch_essential_genes_deg(self) -> pd.DataFrame:
        """
        Fetch essential genes from DEG (Database of Essential Genes).
        Focus on ESKAPE organisms.
        """
        print("[DEG] Fetching essential genes for ESKAPE organisms...")
        # DEG organism IDs for ESKAPE pathogens
        deg_organisms = {
            "Staphylococcus aureus": "DEG1001",
            "Pseudomonas aeruginosa": "DEG1004",
            "Acinetobacter baumannii": "DEG1018",
            "Klebsiella pneumoniae": "DEG1029",
            "Enterococcus faecium": "DEG1031",
        }

        all_genes = []
        for organism, deg_id in deg_organisms.items():
            url = f"http://tubic.org/deg/api/organism/{deg_id}"
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    genes = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else []
                    for g in genes:
                        g["organism"] = organism
                    all_genes.extend(genes)
                    print(f"  {organism}: {len(genes)} essential genes")
            except Exception:
                print(f"  {organism}: DEG API unavailable, will use UniProt fallback")

        if all_genes:
            df = pd.DataFrame(all_genes)
            df.to_csv(DATA_DIR / "deg_essential_genes.csv", index=False)
            return df
        return pd.DataFrame()

    def fetch_uniprot_essential_genes(self) -> pd.DataFrame:
        """Fetch essential genes from UniProt for ESKAPE organisms."""
        print("[UniProt] Fetching essential gene annotations...")
        all_results = []

        for organism in ESKAPE:
            query = f'(organism_name:"{organism}") AND (keyword:KW-0339)'  # Essential gene keyword
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": query,
                "format": "json",
                "size": 500,
                "fields": "accession,gene_names,protein_name,organism_name,go_f,go_p,go_c,xref_pfam,structure_3d",
            }
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                for r in results:
                    genes = r.get("genes", [{}])
                    gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""
                    all_results.append({
                        "uniprot_id": r.get("primaryAccession", ""),
                        "gene_name": gene_name,
                        "protein_name": r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                        "organism": organism,
                        "has_structure": bool(r.get("structure3D")),
                    })
                print(f"  {organism}: {len(results)} proteins")
                time.sleep(1)
            except Exception as e:
                print(f"  {organism}: UniProt error - {e}")

        df = pd.DataFrame(all_results)
        if not df.empty:
            df.to_csv(DATA_DIR / "uniprot_essential_proteins.csv", index=False)
        print(f"[UniProt] Total: {len(df)} essential proteins")
        return df

    def fetch_antibiotics_chembl(self) -> pd.DataFrame:
        """Fetch known antibiotics and their targets from ChEMBL."""
        print("[ChEMBL] Fetching antibiotic compounds and targets...")
        url = f"{CHEMBL_API}/mechanism.json"
        params = {
            "target_organism__in": ",".join(ESKAPE),
            "limit": 1000,
            "format": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            mechanisms = data.get("mechanisms", [])
            results = []
            for m in mechanisms:
                results.append({
                    "drug_chembl_id": m.get("molecule_chembl_id", ""),
                    "drug_name": m.get("molecule_name", ""),
                    "target_chembl_id": m.get("target_chembl_id", ""),
                    "target_name": m.get("target_name", ""),
                    "mechanism": m.get("mechanism_of_action", ""),
                    "action_type": m.get("action_type", ""),
                })
            df = pd.DataFrame(results)
            if not df.empty:
                df.to_csv(DATA_DIR / "chembl_antibiotics.csv", index=False)
            print(f"[ChEMBL] Found {len(df)} drug-target mechanisms")
            return df
        except Exception as e:
            print(f"[ChEMBL] Error: {e}")
            return pd.DataFrame()

    def fetch_amr_opentargets(self) -> pd.DataFrame:
        """Fetch AMR-related targets from OpenTargets."""
        from src.common.opentargets import OpenTargetsClient
        print("[OpenTargets] Fetching AMR-related targets...")
        client = OpenTargetsClient()

        # Relevant disease IDs
        disease_ids = {
            "EFO_0007041": "bacterial infectious disease",
            "MONDO_0005550": "antimicrobial resistance",
        }

        all_targets = []
        for efo_id, name in disease_ids.items():
            try:
                targets = client.get_disease_targets(efo_id, size=200)
                for t in targets:
                    t["source_disease"] = name
                    t["source_efo_id"] = efo_id
                all_targets.extend(targets)
                print(f"  {name}: {len(targets)} targets")
            except Exception as e:
                print(f"  {name}: error - {e}")

        df = pd.DataFrame(all_targets)
        if not df.empty:
            df.to_csv(DATA_DIR / "opentargets_amr.csv", index=False)
        return df

    def fetch_amr_pubmed(self) -> list[dict]:
        """Fetch recent AMR research papers from PubMed."""
        from src.common.pubmed_miner import PubMedMiner
        print("[PubMed] Fetching AMR research papers...")
        miner = PubMedMiner()

        queries = [
            "antimicrobial resistance novel drug target 2023:2026[dp]",
            "essential gene ESKAPE pathogen drug target",
            "antibiotic resistance new therapeutic target",
            "gram negative bacteria novel antibacterial target",
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
        # Save
        with open(DATA_DIR / "pubmed_amr_articles.json", "w") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        return all_articles

    def collect_all(self) -> dict:
        """Run all AMR data collection."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        results = {}

        print("=" * 60)
        print("AMR DATA COLLECTION — Starting")
        print("=" * 60)

        results["uniprot"] = self.fetch_uniprot_essential_genes()
        results["chembl"] = self.fetch_antibiotics_chembl()
        results["opentargets"] = self.fetch_amr_opentargets()
        results["pubmed"] = self.fetch_amr_pubmed()

        print("=" * 60)
        print("AMR DATA COLLECTION — Complete")
        print(f"  UniProt proteins: {len(results['uniprot'])}")
        print(f"  ChEMBL mechanisms: {len(results['chembl'])}")
        print(f"  OpenTargets targets: {len(results['opentargets'])}")
        print(f"  PubMed articles: {len(results['pubmed'])}")
        print("=" * 60)

        return results
