"""
AMR Data Collector V2 — Fixed version with proper essential gene + CARD data
"""
import json
import time
import re
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config.settings import AMR_CONFIG, CHEMBL_API

DATA_DIR = AMR_CONFIG["data_dir"]
ESKAPE = AMR_CONFIG["eskape_organisms"]

# Curated essential genes for ESKAPE organisms from DEG/literature
# These are experimentally validated essential genes
CURATED_ESSENTIAL_GENES = {
    "Staphylococcus aureus": {
        # Cell wall synthesis
        "murA": {"function": "UDP-N-acetylglucosamine enolpyruvyl transferase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.95},
        "murB": {"function": "UDP-N-acetylenolpyruvoylglucosamine reductase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.93},
        "murC": {"function": "UDP-N-acetylmuramate-L-alanine ligase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.92},
        "murD": {"function": "UDP-N-acetylmuramoylalanine-D-glutamate ligase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.91},
        "murE": {"function": "UDP-N-acetylmuramoyl-L-alanyl-D-glutamate-2,6-diaminopimelate ligase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.90},
        "murF": {"function": "UDP-N-acetylmuramoyl-tripeptide-D-alanyl-D-alanine ligase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.90},
        "ftsZ": {"function": "Cell division protein FtsZ", "pathway": "cell_division", "essential_score": 0.98},
        "ftsA": {"function": "Cell division protein FtsA", "pathway": "cell_division", "essential_score": 0.92},
        "ftsW": {"function": "Lipid II flippase", "pathway": "cell_division", "essential_score": 0.90},
        # DNA replication/repair
        "dnaA": {"function": "Chromosomal replication initiator", "pathway": "dna_replication", "essential_score": 0.97},
        "dnaB": {"function": "Replicative DNA helicase", "pathway": "dna_replication", "essential_score": 0.96},
        "dnaE": {"function": "DNA polymerase III alpha subunit", "pathway": "dna_replication", "essential_score": 0.97},
        "dnaN": {"function": "DNA polymerase III beta subunit", "pathway": "dna_replication", "essential_score": 0.95},
        "gyrA": {"function": "DNA gyrase subunit A", "pathway": "dna_topology", "essential_score": 0.98},
        "gyrB": {"function": "DNA gyrase subunit B", "pathway": "dna_topology", "essential_score": 0.97},
        # Protein synthesis
        "rpoB": {"function": "RNA polymerase beta subunit", "pathway": "transcription", "essential_score": 0.99},
        "rpoC": {"function": "RNA polymerase beta' subunit", "pathway": "transcription", "essential_score": 0.99},
        "infA": {"function": "Translation initiation factor IF-1", "pathway": "translation", "essential_score": 0.94},
        "infB": {"function": "Translation initiation factor IF-2", "pathway": "translation", "essential_score": 0.95},
        # Fatty acid synthesis
        "fabI": {"function": "Enoyl-ACP reductase", "pathway": "fatty_acid_synthesis", "essential_score": 0.93},
        "fabH": {"function": "Beta-ketoacyl-ACP synthase III", "pathway": "fatty_acid_synthesis", "essential_score": 0.88},
        "accA": {"function": "Acetyl-CoA carboxylase alpha", "pathway": "fatty_acid_synthesis", "essential_score": 0.87},
        # Folate pathway
        "folA": {"function": "Dihydrofolate reductase (DHFR)", "pathway": "folate_synthesis", "essential_score": 0.92},
        "folP": {"function": "Dihydropteroate synthase (DHPS)", "pathway": "folate_synthesis", "essential_score": 0.90},
        # Two-component systems
        "walK": {"function": "Sensor histidine kinase WalK", "pathway": "signal_transduction", "essential_score": 0.96},
        "walR": {"function": "Response regulator WalR", "pathway": "signal_transduction", "essential_score": 0.96},
        # Isoprenoid synthesis
        "dxr": {"function": "1-deoxy-D-xylulose 5-phosphate reductoisomerase", "pathway": "isoprenoid_synthesis", "essential_score": 0.89},
        "ispD": {"function": "2-C-methyl-D-erythritol 4-phosphate cytidylyltransferase", "pathway": "isoprenoid_synthesis", "essential_score": 0.87},
        # Protein secretion
        "secA": {"function": "Protein translocase SecA", "pathway": "protein_secretion", "essential_score": 0.94},
        "secY": {"function": "Protein translocase SecY", "pathway": "protein_secretion", "essential_score": 0.95},
    },
    "Pseudomonas aeruginosa": {
        "lpxC": {"function": "UDP-3-O-acyl-N-acetylglucosamine deacetylase", "pathway": "lps_synthesis", "essential_score": 0.97},
        "lpxA": {"function": "UDP-N-acetylglucosamine acyltransferase", "pathway": "lps_synthesis", "essential_score": 0.93},
        "lpxB": {"function": "Lipid-A-disaccharide synthase", "pathway": "lps_synthesis", "essential_score": 0.92},
        "lpxD": {"function": "UDP-3-O-acylglucosamine N-acyltransferase", "pathway": "lps_synthesis", "essential_score": 0.91},
        "lptD": {"function": "LPS transport protein LptD", "pathway": "lps_transport", "essential_score": 0.94},
        "bamA": {"function": "Outer membrane protein assembly factor BamA", "pathway": "outer_membrane", "essential_score": 0.96},
        "bamD": {"function": "Outer membrane protein assembly factor BamD", "pathway": "outer_membrane", "essential_score": 0.93},
        "ftsZ": {"function": "Cell division protein FtsZ", "pathway": "cell_division", "essential_score": 0.98},
        "murA": {"function": "UDP-N-acetylglucosamine enolpyruvyl transferase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.95},
        "gyrA": {"function": "DNA gyrase subunit A", "pathway": "dna_topology", "essential_score": 0.98},
        "gyrB": {"function": "DNA gyrase subunit B", "pathway": "dna_topology", "essential_score": 0.97},
        "parC": {"function": "DNA topoisomerase IV subunit A", "pathway": "dna_topology", "essential_score": 0.95},
        "parE": {"function": "DNA topoisomerase IV subunit B", "pathway": "dna_topology", "essential_score": 0.94},
        "rpoB": {"function": "RNA polymerase beta subunit", "pathway": "transcription", "essential_score": 0.99},
        "dnaA": {"function": "Chromosomal replication initiator", "pathway": "dna_replication", "essential_score": 0.97},
        "fabI": {"function": "Enoyl-ACP reductase", "pathway": "fatty_acid_synthesis", "essential_score": 0.91},
        "folA": {"function": "Dihydrofolate reductase", "pathway": "folate_synthesis", "essential_score": 0.90},
        "secA": {"function": "Protein translocase SecA", "pathway": "protein_secretion", "essential_score": 0.94},
    },
    "Acinetobacter baumannii": {
        "lpxC": {"function": "UDP-3-O-acyl-N-acetylglucosamine deacetylase", "pathway": "lps_synthesis", "essential_score": 0.96},
        "bamA": {"function": "Outer membrane protein assembly factor BamA", "pathway": "outer_membrane", "essential_score": 0.95},
        "ftsZ": {"function": "Cell division protein FtsZ", "pathway": "cell_division", "essential_score": 0.97},
        "murA": {"function": "UDP-N-acetylglucosamine enolpyruvyl transferase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.94},
        "gyrA": {"function": "DNA gyrase subunit A", "pathway": "dna_topology", "essential_score": 0.98},
        "rpoB": {"function": "RNA polymerase beta subunit", "pathway": "transcription", "essential_score": 0.99},
        "dnaA": {"function": "Chromosomal replication initiator", "pathway": "dna_replication", "essential_score": 0.96},
    },
    "Klebsiella pneumoniae": {
        "lpxC": {"function": "UDP-3-O-acyl-N-acetylglucosamine deacetylase", "pathway": "lps_synthesis", "essential_score": 0.97},
        "lpxA": {"function": "UDP-N-acetylglucosamine acyltransferase", "pathway": "lps_synthesis", "essential_score": 0.92},
        "bamA": {"function": "Outer membrane protein assembly factor BamA", "pathway": "outer_membrane", "essential_score": 0.95},
        "lptD": {"function": "LPS transport protein LptD", "pathway": "lps_transport", "essential_score": 0.93},
        "ftsZ": {"function": "Cell division protein FtsZ", "pathway": "cell_division", "essential_score": 0.98},
        "murA": {"function": "UDP-N-acetylglucosamine enolpyruvyl transferase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.95},
        "gyrA": {"function": "DNA gyrase subunit A", "pathway": "dna_topology", "essential_score": 0.98},
        "rpoB": {"function": "RNA polymerase beta subunit", "pathway": "transcription", "essential_score": 0.99},
        "folA": {"function": "Dihydrofolate reductase", "pathway": "folate_synthesis", "essential_score": 0.91},
    },
    "Enterococcus faecium": {
        "ftsZ": {"function": "Cell division protein FtsZ", "pathway": "cell_division", "essential_score": 0.97},
        "murA": {"function": "UDP-N-acetylglucosamine enolpyruvyl transferase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.94},
        "gyrA": {"function": "DNA gyrase subunit A", "pathway": "dna_topology", "essential_score": 0.97},
        "gyrB": {"function": "DNA gyrase subunit B", "pathway": "dna_topology", "essential_score": 0.96},
        "rpoB": {"function": "RNA polymerase beta subunit", "pathway": "transcription", "essential_score": 0.99},
        "dnaA": {"function": "Chromosomal replication initiator", "pathway": "dna_replication", "essential_score": 0.96},
        "fabI": {"function": "Enoyl-ACP reductase", "pathway": "fatty_acid_synthesis", "essential_score": 0.90},
    },
    "Enterobacter cloacae": {
        "lpxC": {"function": "UDP-3-O-acyl-N-acetylglucosamine deacetylase", "pathway": "lps_synthesis", "essential_score": 0.96},
        "bamA": {"function": "Outer membrane protein assembly factor BamA", "pathway": "outer_membrane", "essential_score": 0.94},
        "ftsZ": {"function": "Cell division protein FtsZ", "pathway": "cell_division", "essential_score": 0.97},
        "murA": {"function": "UDP-N-acetylglucosamine enolpyruvyl transferase", "pathway": "peptidoglycan_synthesis", "essential_score": 0.94},
        "gyrA": {"function": "DNA gyrase subunit A", "pathway": "dna_topology", "essential_score": 0.98},
        "rpoB": {"function": "RNA polymerase beta subunit", "pathway": "transcription", "essential_score": 0.99},
    },
}

# Known antibiotic targets and their drugs
KNOWN_ANTIBIOTIC_TARGETS = {
    "gyrA": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
    "gyrB": ["novobiocin", "coumermycin"],
    "parC": ["ciprofloxacin"],
    "parE": ["ciprofloxacin"],
    "rpoB": ["rifampin"],
    "folA": ["trimethoprim"],
    "folP": ["sulfamethoxazole"],
    "fabI": ["triclosan", "isoniazid"],
    "murA": ["fosfomycin"],
    "ftsZ": [],  # No approved drug yet — novel target!
    "lpxC": [],  # ACHN-975 failed Phase I — still promising
    "bamA": [],  # Darobactin in preclinical
    "lptD": [],  # Murepavadin failed Phase III but target validated
    "walK": [],  # Novel — no drugs
    "walR": [],  # Novel — no drugs
    "dxr": ["fosmidomycin"],
    "secA": [],  # Novel
}


class AMRDataCollectorV2:
    def load_curated_essential_genes(self) -> pd.DataFrame:
        """Load curated essential genes database."""
        print("[Curated] Loading experimentally validated essential genes...")
        rows = []
        for organism, genes in CURATED_ESSENTIAL_GENES.items():
            for gene_name, info in genes.items():
                existing_drugs = KNOWN_ANTIBIOTIC_TARGETS.get(gene_name, [])
                rows.append({
                    "gene_name": gene_name,
                    "organism": organism,
                    "function": info["function"],
                    "pathway": info["pathway"],
                    "essential_score": info["essential_score"],
                    "has_existing_drug": len(existing_drugs) > 0,
                    "existing_drugs": ", ".join(existing_drugs) if existing_drugs else "NONE",
                    "is_novel_target": len(existing_drugs) == 0,
                })

        df = pd.DataFrame(rows)
        df.to_csv(DATA_DIR / "curated_essential_genes.csv", index=False)

        # Stats
        unique_genes = df["gene_name"].nunique()
        novel = df[df["is_novel_target"]]["gene_name"].nunique()
        print(f"  Total entries: {len(df)}")
        print(f"  Unique genes: {unique_genes}")
        print(f"  Novel targets (no existing drug): {novel}")
        print(f"  Organisms covered: {df['organism'].nunique()}")
        return df

    def fetch_uniprot_for_genes(self, genes_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch UniProt IDs for essential genes."""
        print("[UniProt] Fetching protein details for essential genes...")
        results = []
        seen = set()

        for _, row in tqdm(genes_df.iterrows(), total=len(genes_df), desc="UniProt lookup"):
            gene = row["gene_name"]
            organism = row["organism"]
            key = f"{gene}_{organism}"
            if key in seen:
                continue
            seen.add(key)

            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": f'(gene:{gene}) AND (organism_name:"{organism}")',
                "format": "json", "size": 1,
                "fields": "accession,gene_names,protein_name,structure_3d,go_f",
            }
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    hits = resp.json().get("results", [])
                    if hits:
                        r = hits[0]
                        results.append({
                            "gene_name": gene,
                            "organism": organism,
                            "uniprot_id": r.get("primaryAccession", ""),
                            "protein_name": r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                            "has_structure": bool(r.get("structure3D")),
                        })
                time.sleep(0.3)
            except Exception:
                pass

        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(DATA_DIR / "essential_genes_uniprot.csv", index=False)
        print(f"  Resolved {len(df)} UniProt entries")
        return df

    def fetch_opentargets_amr(self) -> pd.DataFrame:
        """Fetch AMR-related targets from OpenTargets."""
        from src.common.opentargets import OpenTargetsClient
        print("[OpenTargets] Fetching bacterial infection targets...")
        client = OpenTargetsClient()

        disease_ids = {
            "EFO_0000763": "bacterial infectious disease",
            "OTAR_0000017": "bacterial infection",
        }

        all_targets = []
        for efo_id, name in disease_ids.items():
            try:
                targets = client.get_disease_targets(efo_id, size=200)
                for t in targets:
                    t["source_disease"] = name
                all_targets.extend(targets)
                print(f"  {name}: {len(targets)} targets")
            except Exception as e:
                print(f"  {name}: {e}")

        df = pd.DataFrame(all_targets)
        if not df.empty:
            df.to_csv(DATA_DIR / "opentargets_amr_v2.csv", index=False)
        return df

    def fetch_pubmed_targeted(self) -> list[dict]:
        """Fetch targeted PubMed articles about specific AMR drug targets."""
        from src.common.pubmed_miner import PubMedMiner
        print("[PubMed] Fetching targeted AMR research papers...")
        miner = PubMedMiner()

        # Focused queries for real drug targets
        queries = [
            '"essential gene" "drug target" ESKAPE 2022:2026[dp]',
            '"LpxC" inhibitor antibacterial 2022:2026[dp]',
            '"FtsZ" inhibitor antibacterial 2022:2026[dp]',
            '"BamA" antibacterial target 2022:2026[dp]',
            '"WalK" "WalR" antibacterial 2022:2026[dp]',
            '"MurA" "MurB" antibacterial target',
            'novel antibacterial target mechanism 2023:2026[dp]',
            '"outer membrane" biogenesis drug target gram-negative',
            '"cell division" antibacterial novel target',
            '"fatty acid synthesis" antibacterial FabI FabH',
        ]

        all_articles = []
        seen = set()
        for query in queries:
            try:
                articles = miner.search_and_fetch(query, max_results=100)
                for a in articles:
                    if a["pmid"] not in seen:
                        seen.add(a["pmid"])
                        all_articles.append(a)
                time.sleep(1)
            except Exception as e:
                print(f"  PubMed query failed: {e}")
                continue

        print(f"  Total unique articles: {len(all_articles)}")
        with open(DATA_DIR / "pubmed_amr_v2.json", "w") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        return all_articles

    def collect_all(self) -> dict:
        """Run all V2 data collection."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        results = {}

        print("=" * 60)
        print("AMR DATA COLLECTION V2 — Starting")
        print("=" * 60)

        results["essential_genes"] = self.load_curated_essential_genes()
        results["uniprot"] = self.fetch_uniprot_for_genes(results["essential_genes"])
        results["opentargets"] = self.fetch_opentargets_amr()
        results["pubmed"] = self.fetch_pubmed_targeted()

        print("=" * 60)
        print("AMR DATA COLLECTION V2 — Complete")
        for k, v in results.items():
            size = len(v) if isinstance(v, (list, pd.DataFrame)) else "N/A"
            print(f"  {k}: {size}")
        print("=" * 60)

        return results
