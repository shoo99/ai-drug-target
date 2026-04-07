#!/usr/bin/env python3
"""
Internal Validation — Cross-validate AI predictions against recent literature
Checks: Did our AI predict targets that were recently validated in new papers?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from src.common.pubmed_miner import PubMedMiner
from config.settings import AMR_CONFIG, PRURITUS_CONFIG


def validate_track(track: str, config: dict):
    """Validate predictions for a given track."""
    track_name = "AMR" if track == "amr" else "Pruritus"
    print(f"\n{'='*60}")
    print(f"VALIDATION — {track_name}")
    print(f"{'='*60}")

    # Load our top predictions
    scored_path = config["data_dir"] / "top_targets_scored.csv"
    if not scored_path.exists():
        print(f"  No scored targets for {track_name}")
        return

    df = pd.read_csv(scored_path)
    our_targets = df["gene_name"].dropna().unique()[:20]
    print(f"\n  Our top {len(our_targets)} predicted targets: {list(our_targets)}")

    # Search for very recent validation papers (2025-2026)
    miner = PubMedMiner()

    validation_results = []
    for gene in our_targets:
        if track == "amr":
            query = f'"{gene}" antimicrobial drug target 2025:2026[dp]'
        else:
            query = f'"{gene}" pruritus itch therapeutic target 2025:2026[dp]'

        pmids = miner.search(query, max_results=10)

        if pmids:
            # Fetch first few to check relevance
            articles = miner.fetch_abstracts(pmids[:3])
            titles = [a.get("title", "") for a in articles]

            validation_results.append({
                "gene": gene,
                "recent_papers": len(pmids),
                "validated": len(pmids) >= 2,
                "sample_titles": titles[:2],
            })
            status = "✅ VALIDATED" if len(pmids) >= 2 else f"⚠️ {len(pmids)} paper(s)"
            print(f"\n  {gene}: {status}")
            for t in titles[:2]:
                print(f"    → {t[:100]}")
        else:
            validation_results.append({
                "gene": gene,
                "recent_papers": 0,
                "validated": False,
                "sample_titles": [],
            })
            print(f"\n  {gene}: 🆕 NOVEL — No recent papers (potential first-mover)")

    # Summary
    val_df = pd.DataFrame(validation_results)
    validated_count = len(val_df[val_df["validated"]])
    novel_count = len(val_df[val_df["recent_papers"] == 0])
    partial_count = len(val_df) - validated_count - novel_count

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY — {track_name}")
    print(f"{'='*60}")
    print(f"  ✅ Validated (≥2 recent papers): {validated_count}/{len(val_df)}")
    print(f"  ⚠️ Partially validated (1 paper): {partial_count}/{len(val_df)}")
    print(f"  🆕 Novel (no recent papers): {novel_count}/{len(val_df)}")
    print(f"  Validation rate: {validated_count/len(val_df)*100:.1f}%")

    val_df.to_csv(config["data_dir"] / "validation_results.csv", index=False)
    return val_df


def main():
    print("\n🔬 AI PREDICTION VALIDATION")
    print("Checking if our predictions match recent literature...\n")

    amr_val = validate_track("amr", AMR_CONFIG)
    pru_val = validate_track("pruritus", PRURITUS_CONFIG)

    print(f"\n✅ Validation complete!")


if __name__ == "__main__":
    main()
