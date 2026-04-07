"""
Patent Competition Analyzer — Google Patents + Lens.org integration
Analyzes patent landscape for drug target candidates
"""
import re
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from config.settings import DATA_DIR

PATENT_DIR = DATA_DIR / "common" / "patents"


class PatentAnalyzer:
    def __init__(self):
        PATENT_DIR.mkdir(parents=True, exist_ok=True)

    def search_google_patents(self, query: str, max_results: int = 20) -> list[dict]:
        """Search Google Patents via SerpAPI-style scraping of public data."""
        # Use Google Patents public search URL parsing
        # Alternative: use Lens.org free API
        results = self._search_lens_org(query, max_results)
        return results

    def _search_lens_org(self, query: str, max_results: int = 20) -> list[dict]:
        """Search patents via Lens.org scholarly API (free tier)."""
        url = "https://api.lens.org/patent/search"

        # Lens.org requires API token for full access
        # Fall back to EPO Open Patent Services (free)
        return self._search_epo(query, max_results)

    def _search_epo(self, query: str, max_results: int = 20) -> list[dict]:
        """Search patents via EPO Open Patent Services (OPS) - free access."""
        # EPO OPS published-data search
        base_url = "https://ops.epo.org/3.2/rest-services/published-data/search"

        # Clean query for CQL
        clean_query = query.replace('"', '').replace("'", "")
        params = {
            "q": f'ta="{clean_query}"',
            "Range": f"1-{min(max_results, 25)}",
        }
        headers = {"Accept": "application/json"}

        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = self._parse_epo_results(data)
                return results
            elif resp.status_code == 403:
                # EPO requires OAuth for some endpoints; use fallback
                return self._search_pubmed_patents(query, max_results)
        except Exception as e:
            print(f"  EPO search error: {e}")

        return self._search_pubmed_patents(query, max_results)

    def _search_pubmed_patents(self, query: str, max_results: int = 20) -> list[dict]:
        """Fallback: Search PubMed for patent-related publications."""
        from src.common.pubmed_miner import PubMedMiner
        miner = PubMedMiner()

        patent_query = f'({query}) AND (patent OR "intellectual property" OR "patent application")'
        articles = miner.search_and_fetch(patent_query, max_results=max_results)

        results = []
        for article in articles:
            results.append({
                "source": "pubmed_patent_ref",
                "title": article.get("title", ""),
                "year": article.get("year"),
                "pmid": article.get("pmid", ""),
                "type": "patent_reference",
            })
        return results

    def _parse_epo_results(self, data: dict) -> list[dict]:
        """Parse EPO OPS search results."""
        results = []
        try:
            search_result = data.get("ops:world-patent-data", {}).get("ops:biblio-search", {})
            search_results = search_result.get("ops:search-result", {}).get("ops:publication-reference", [])

            if isinstance(search_results, dict):
                search_results = [search_results]

            for pub in search_results:
                doc_id = pub.get("document-id", {})
                results.append({
                    "source": "epo",
                    "country": doc_id.get("country", {}).get("$", ""),
                    "doc_number": doc_id.get("doc-number", {}).get("$", ""),
                    "kind": doc_id.get("kind", {}).get("$", ""),
                    "type": "patent",
                })
        except Exception:
            pass
        return results

    def analyze_target_patents(self, gene_symbol: str, disease_context: str) -> dict:
        """Comprehensive patent analysis for a drug target."""
        print(f"  Analyzing patents for: {gene_symbol} + {disease_context}")

        queries = [
            f"{gene_symbol} drug target",
            f"{gene_symbol} inhibitor therapeutic",
            f"{gene_symbol} {disease_context} treatment",
            f"{gene_symbol} antibody pharmaceutical",
        ]

        all_patents = []
        for query in queries:
            patents = self.search_google_patents(query, max_results=10)
            all_patents.extend(patents)
            time.sleep(1)

        # Deduplicate
        seen = set()
        unique_patents = []
        for p in all_patents:
            key = p.get("title", "") or p.get("doc_number", "")
            if key and key not in seen:
                seen.add(key)
                unique_patents.append(p)

        # Analyze
        recent_count = sum(1 for p in unique_patents
                          if p.get("year") and p["year"] >= 2022)

        result = {
            "gene": gene_symbol,
            "disease_context": disease_context,
            "total_patents_found": len(unique_patents),
            "recent_patents_3yr": recent_count,
            "patent_activity": "high" if len(unique_patents) > 10
                              else "moderate" if len(unique_patents) > 3
                              else "low",
            "freedom_to_operate": "needs_review" if len(unique_patents) > 5
                                  else "likely_clear",
            "ip_opportunity": "low" if len(unique_patents) > 10
                             else "moderate" if len(unique_patents) > 3
                             else "high",
            "patents": unique_patents[:10],
        }
        return result

    def batch_analyze(self, targets: list[dict], disease_context: str) -> pd.DataFrame:
        """Analyze patents for multiple targets."""
        print(f"\n{'='*60}")
        print(f"PATENT LANDSCAPE ANALYSIS — {disease_context}")
        print(f"{'='*60}")

        results = []
        for target in tqdm(targets, desc="Patent analysis"):
            gene = target.get("gene_name", target.get("gene", ""))
            if not gene:
                continue
            analysis = self.analyze_target_patents(gene, disease_context)
            results.append(analysis)
            time.sleep(0.5)

        df = pd.DataFrame(results)
        if not df.empty:
            output_path = PATENT_DIR / f"patent_analysis_{disease_context.replace(' ', '_')}.csv"
            df.to_csv(output_path, index=False)
            print(f"\n  Saved: {output_path}")

            # Summary
            high_ip = len(df[df["ip_opportunity"] == "high"])
            print(f"\n  Summary:")
            print(f"    Total targets analyzed: {len(df)}")
            print(f"    High IP opportunity: {high_ip}")
            print(f"    Low competition (clear FTO): {len(df[df['freedom_to_operate'] == 'likely_clear'])}")

        return df
