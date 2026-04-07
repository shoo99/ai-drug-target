#!/usr/bin/env python3
"""Fetch AlphaFold structure + druggability data for top targets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.common.alphafold_client import AlphaFoldClient
from config.settings import AMR_CONFIG, PRURITUS_CONFIG, DATA_DIR


def main():
    client = AlphaFoldClient()

    # AMR top targets
    amr_path = AMR_CONFIG["data_dir"] / "top_targets_scored.csv"
    if amr_path.exists():
        amr_df = pd.read_csv(amr_path)
        amr_genes = amr_df["gene_name"].dropna().unique()[:20]
        print(f"\n🧬 AMR — Assessing {len(amr_genes)} targets")
        amr_results = client.batch_assess(list(amr_genes))
        amr_results.to_csv(AMR_CONFIG["data_dir"] / "alphafold_druggability.csv", index=False)

        print("\n  AMR Druggability Results:")
        for _, row in amr_results.iterrows():
            af = "✅" if row.get("has_alphafold") else "❌"
            pdb = "✅" if row.get("has_pdb_experimental") else "❌"
            score = row.get("druggability_score", 0)
            print(f"    {row['gene']:12s} | AF:{af} PDB:{pdb} | "
                  f"Class: {row.get('protein_class', 'N/A'):15s} | "
                  f"Score: {score:.3f} | {row.get('modality_suggestion', '')}")

    # Pruritus top targets
    pru_path = PRURITUS_CONFIG["data_dir"] / "top_targets_scored.csv"
    if pru_path.exists():
        pru_df = pd.read_csv(pru_path)
        pru_genes = pru_df["gene_name"].dropna().unique()[:20]
        print(f"\n🧪 Pruritus — Assessing {len(pru_genes)} targets")
        pru_results = client.batch_assess(list(pru_genes))
        pru_results.to_csv(PRURITUS_CONFIG["data_dir"] / "alphafold_druggability.csv", index=False)

        print("\n  Pruritus Druggability Results:")
        for _, row in pru_results.iterrows():
            af = "✅" if row.get("has_alphafold") else "❌"
            pdb = "✅" if row.get("has_pdb_experimental") else "❌"
            score = row.get("druggability_score", 0)
            print(f"    {row['gene']:12s} | AF:{af} PDB:{pdb} | "
                  f"Class: {row.get('protein_class', 'N/A'):15s} | "
                  f"Score: {score:.3f} | {row.get('modality_suggestion', '')}")

    print("\n✅ AlphaFold assessment complete!")


if __name__ == "__main__":
    main()
