#!/usr/bin/env python3
"""Re-run docking with correct UniProt IDs from V2 data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.common.molecular_docking import MolecularDockingPipeline
from config.settings import AMR_CONFIG, PRURITUS_CONFIG


def main():
    pipeline = MolecularDockingPipeline()

    # AMR — use essential_genes_uniprot.csv (has real UniProt IDs)
    amr_uniprot = pd.read_csv(AMR_CONFIG["data_dir"] / "essential_genes_uniprot.csv")
    # Deduplicate by gene_name, keep first (best match)
    amr_unique = amr_uniprot.drop_duplicates(subset="gene_name", keep="first")

    # Pruritus — use alphafold_druggability.csv
    pru_path = PRURITUS_CONFIG["data_dir"] / "alphafold_druggability.csv"
    pru_af = pd.read_csv(pru_path) if pru_path.exists() else pd.DataFrame()

    # Build target list
    targets = []

    # AMR — get canonical SwissProt IDs (not TrEMBL fragments)
    import requests, time

    amr_scored = pd.read_csv(AMR_CONFIG["data_dir"] / "top_targets_scored.csv")
    novel_genes = set(amr_scored[amr_scored.get("is_novel_target", False) == True]["gene_name"].unique())

    print("Resolving canonical UniProt IDs for AMR targets...")
    for gene in list(novel_genes)[:15]:
        # Search for reviewed (SwissProt) entries, E. coli K-12 as reference organism
        # since most essential gene studies use model organisms
        for org in ["Escherichia coli", "Pseudomonas aeruginosa", "Staphylococcus aureus"]:
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": f'(gene:{gene}) AND (organism_name:"{org}") AND (reviewed:true)',
                "format": "json", "size": 1, "fields": "accession",
            }
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    hits = resp.json().get("results", [])
                    if hits:
                        uid = hits[0]["primaryAccession"]
                        targets.append({"gene": gene, "uniprot_id": uid, "track": "amr"})
                        print(f"  {gene} → {uid} ({org})")
                        break
                time.sleep(0.3)
            except Exception:
                pass

    # Pruritus top targets
    for _, row in pru_af.iterrows():
        uid = row.get("uniprot_id", "")
        if uid and str(uid) != "nan":
            targets.append({"gene": row["gene"], "uniprot_id": uid, "track": "pruritus"})

    # Deduplicate
    seen = set()
    unique_targets = []
    for t in targets:
        if t["uniprot_id"] not in seen:
            seen.add(t["uniprot_id"])
            unique_targets.append(t)

    print(f"🔬 Running docking for {len(unique_targets)} targets")
    print(f"   AMR: {sum(1 for t in unique_targets if t['track'] == 'amr')}")
    print(f"   Pruritus: {sum(1 for t in unique_targets if t['track'] == 'pruritus')}")

    results = pipeline.run_full_analysis(unique_targets)

    # Summary
    analyzed = [r for r in results if r.get("status") == "analyzed"]
    print(f"\n{'='*60}")
    print(f"DOCKING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Structures downloaded: {len(analyzed)}/{len(unique_targets)}")

    for r in sorted(analyzed, key=lambda x: x.get("top_pocket_score", 0), reverse=True):
        track = next((t["track"] for t in unique_targets if t["gene"] == r["gene"]), "?")
        print(f"  {r['gene']:12s} [{track:8s}] | Atoms: {r['n_atoms']:5d} | "
              f"Pockets: {r['n_pockets']:2d} | Top score: {r['top_pocket_score']:.3f}")


if __name__ == "__main__":
    main()
