#!/usr/bin/env python3
"""
Paper 3 — Phase 1: Collect PubMed articles on chronic stress and brain structure.

Multiple targeted queries to cover:
- Hippocampus + stress (structural changes, volume, neurogenesis)
- Amygdala + stress (hypertrophy, connectivity)
- Prefrontal cortex + stress (atrophy, function)
- HPA axis genes (BDNF, NR3C1, FKBP5, CRHR1, 5-HTT/SLC6A4)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from src.common.pubmed_miner import PubMedMiner
from config.settings import DATA_DIR

OUT_DIR = DATA_DIR / "stress"
OUT_DIR.mkdir(exist_ok=True)

# Targeted queries for comprehensive coverage
QUERIES = [
    # Main structural queries
    '("chronic stress"[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (volume OR atrophy OR neurogenesis OR "structural change")',
    '("chronic stress"[Title/Abstract]) AND (amygdala[Title/Abstract]) AND (volume OR hypertrophy OR "structural change" OR connectivity)',
    '("chronic stress"[Title/Abstract]) AND ("prefrontal cortex"[Title/Abstract]) AND (volume OR atrophy OR "gray matter" OR thickness)',

    # HPA axis + brain structure
    '(cortisol[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (volume OR atrophy OR "brain structure")',
    '("HPA axis"[Title/Abstract]) AND (brain[Title/Abstract]) AND (structural OR morphological OR volume)',

    # Key genes + stress + brain
    '(BDNF[Title/Abstract]) AND (stress[Title/Abstract]) AND (hippocampus OR amygdala OR "prefrontal cortex")',
    '(NR3C1 OR "glucocorticoid receptor"[Title/Abstract]) AND (stress[Title/Abstract]) AND (brain[Title/Abstract])',
    '(FKBP5[Title/Abstract]) AND (stress[Title/Abstract]) AND (brain OR hippocampus OR amygdala)',
    '(CRHR1 OR "CRH receptor"[Title/Abstract]) AND (stress[Title/Abstract]) AND (brain[Title/Abstract])',
    '(SLC6A4 OR "serotonin transporter" OR "5-HTT"[Title/Abstract]) AND (stress[Title/Abstract]) AND (hippocampus OR amygdala)',

    # PTSD + brain structure (major stress-related condition)
    '(PTSD[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (volume OR atrophy)',
    '(PTSD[Title/Abstract]) AND (amygdala[Title/Abstract]) AND (volume OR activity)',

    # Meta-analysis / review coverage
    '("chronic stress"[Title/Abstract]) AND (brain[Title/Abstract]) AND ("meta-analysis" OR "systematic review")',

    # Animal models
    '("chronic unpredictable stress" OR "chronic mild stress"[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (dendritic OR synaptic OR neurogenesis)',

    # Epigenetics + stress + brain
    '(epigenetic[Title/Abstract]) AND (stress[Title/Abstract]) AND (hippocampus OR amygdala) AND (methylation OR acetylation)',
]


def main():
    print("=" * 60)
    print("  PAPER 3 — PUBMED COLLECTION: CHRONIC STRESS & BRAIN")
    print("=" * 60)

    miner = PubMedMiner()
    all_pmids = set()
    query_results = {}

    for i, query in enumerate(QUERIES, 1):
        print(f"\n  Query {i}/{len(QUERIES)}:")
        print(f"    {query[:80]}...")
        try:
            pmids = miner.search(query, max_results=1000)
            new_pmids = set(pmids) - all_pmids
            all_pmids.update(pmids)
            query_results[f"Q{i}"] = {"query": query, "count": len(pmids), "new": len(new_pmids)}
            print(f"    Found: {len(pmids)}, New unique: {len(new_pmids)}, Total: {len(all_pmids)}")
            time.sleep(0.5)
        except Exception as e:
            print(f"    ERROR: {e}")
            query_results[f"Q{i}"] = {"query": query, "count": 0, "error": str(e)}

    print(f"\n{'='*60}")
    print(f"  TOTAL UNIQUE PMIDs: {len(all_pmids)}")
    print(f"{'='*60}")

    # Fetch all abstracts
    print(f"\n  Fetching abstracts for {len(all_pmids)} articles...")
    pmid_list = sorted(all_pmids)
    articles = miner.fetch_abstracts(pmid_list)
    print(f"  Fetched: {len(articles)} articles with abstracts")

    # Filter: must have abstract
    articles_with_abstract = [a for a in articles if a.get("abstract")]
    print(f"  With abstract: {len(articles_with_abstract)}")

    # Save
    out_path = OUT_DIR / "pubmed_stress_brain.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles_with_abstract, f, ensure_ascii=False, indent=2)

    # Save query summary
    summary_path = OUT_DIR / "collection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_unique_pmids": len(all_pmids),
            "articles_with_abstract": len(articles_with_abstract),
            "queries": query_results,
        }, f, indent=2)

    # Basic stats
    years = [a.get("year", 0) for a in articles_with_abstract if a.get("year")]
    if years:
        print(f"\n  Year range: {min(years)} - {max(years)}")
        # Decade distribution
        from collections import Counter
        decades = Counter(y // 10 * 10 for y in years if y > 1900)
        print(f"  Decade distribution:")
        for decade in sorted(decades):
            print(f"    {decade}s: {decades[decade]}")

    print(f"\n  Saved: {out_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
