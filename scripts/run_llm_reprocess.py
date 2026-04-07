#!/usr/bin/env python3
"""Reprocess all PubMed articles with LLM NLP (background job)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.llm_nlp import LLMNLPExtractor
from config.settings import AMR_CONFIG, PRURITUS_CONFIG


def main():
    ext = LLMNLPExtractor(backend="ollama", ollama_model="gemma4:e4b")

    all_articles = []

    # AMR articles
    amr_path = AMR_CONFIG["data_dir"] / "pubmed_amr_v2.json"
    if amr_path.exists():
        with open(amr_path) as f:
            articles = json.load(f)
        print(f"AMR articles: {len(articles)}")
        all_articles.extend(articles)

    # Pruritus articles
    pru_path = PRURITUS_CONFIG["data_dir"] / "pubmed_pruritus_articles.json"
    if pru_path.exists():
        with open(pru_path) as f:
            articles = json.load(f)
        print(f"Pruritus articles: {len(articles)}")
        all_articles.extend(articles)

    print(f"\nTotal articles to process: {len(all_articles)}")
    print(f"Estimated time: ~{len(all_articles) * 50 / 3600:.1f} hours")
    print(f"Starting LLM reprocessing...\n")

    result = ext.process_articles(all_articles, batch_delay=0.3)

    print(f"\n{'='*60}")
    print(f"REPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Articles processed: {result['n_articles']}")
    print(f"  Unique genes: {result['n_unique_genes']}")
    print(f"  Relations: {result['n_relations']}")
    print(f"  Drugs: {result['n_drugs']}")
    print(f"  Top 20 genes: {result['gene_counts'].most_common(20)}")


if __name__ == "__main__":
    main()
