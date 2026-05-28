#!/usr/bin/env python3
"""
Paper 3 — Fetch abstracts for already-collected PMIDs.
Uses smaller batch size and retry logic to handle NCBI timeouts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import requests
import xml.etree.ElementTree as ET
from config.settings import DATA_DIR, NCBI_EMAIL, NCBI_API_KEY

OUT_DIR = DATA_DIR / "stress"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pmids(query, max_results=1000):
    """Search PubMed."""
    params = {
        "db": "pubmed", "term": query, "retmax": max_results,
        "retmode": "json", "sort": "relevance",
    }
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_batch(pmids, retries=3):
    """Fetch a small batch with retries."""
    params = {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
    }
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    for attempt in range(retries):
        try:
            resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=120)
            resp.raise_for_status()
            return parse_xml(resp.text)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            print(f"    Retry {attempt+1}/{retries}: {e}")
            time.sleep(5 * (attempt + 1))
    return []


def parse_xml(xml_text):
    """Parse PubMed XML."""
    root = ET.fromstring(xml_text)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.find(".//PMID").text
            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None and title_el.text else ""

            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (p.text or "") + (p.tail or "")
                for p in abstract_parts
            ).strip()

            year_el = article.find(".//PubDate/Year")
            year = int(year_el.text) if year_el is not None and year_el.text else 0

            # MeSH terms
            mesh_terms = [
                m.find("DescriptorName").text
                for m in article.findall(".//MeshHeading")
                if m.find("DescriptorName") is not None
            ]

            # Keywords
            keywords = [
                k.text for k in article.findall(".//Keyword")
                if k.text
            ]

            if abstract:
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "mesh_terms": mesh_terms,
                    "keywords": keywords,
                })
        except Exception:
            continue
    return articles


def main():
    print("=" * 60)
    print("  PAPER 3 — FETCH STRESS ABSTRACTS (with retry)")
    print("=" * 60)

    # Re-collect PMIDs (fast, just search)
    QUERIES = [
        '("chronic stress"[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (volume OR atrophy OR neurogenesis OR "structural change")',
        '("chronic stress"[Title/Abstract]) AND (amygdala[Title/Abstract]) AND (volume OR hypertrophy OR "structural change" OR connectivity)',
        '("chronic stress"[Title/Abstract]) AND ("prefrontal cortex"[Title/Abstract]) AND (volume OR atrophy OR "gray matter" OR thickness)',
        '(cortisol[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (volume OR atrophy OR "brain structure")',
        '("HPA axis"[Title/Abstract]) AND (brain[Title/Abstract]) AND (structural OR morphological OR volume)',
        '(BDNF[Title/Abstract]) AND (stress[Title/Abstract]) AND (hippocampus OR amygdala OR "prefrontal cortex")',
        '(NR3C1 OR "glucocorticoid receptor"[Title/Abstract]) AND (stress[Title/Abstract]) AND (brain[Title/Abstract])',
        '(FKBP5[Title/Abstract]) AND (stress[Title/Abstract]) AND (brain OR hippocampus OR amygdala)',
        '(CRHR1 OR "CRH receptor"[Title/Abstract]) AND (stress[Title/Abstract]) AND (brain[Title/Abstract])',
        '(SLC6A4 OR "serotonin transporter" OR "5-HTT"[Title/Abstract]) AND (stress[Title/Abstract]) AND (hippocampus OR amygdala)',
        '(PTSD[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (volume OR atrophy)',
        '(PTSD[Title/Abstract]) AND (amygdala[Title/Abstract]) AND (volume OR activity)',
        '("chronic stress"[Title/Abstract]) AND (brain[Title/Abstract]) AND ("meta-analysis" OR "systematic review")',
        '("chronic unpredictable stress" OR "chronic mild stress"[Title/Abstract]) AND (hippocampus[Title/Abstract]) AND (dendritic OR synaptic OR neurogenesis)',
        '(epigenetic[Title/Abstract]) AND (stress[Title/Abstract]) AND (hippocampus OR amygdala) AND (methylation OR acetylation)',
    ]

    all_pmids = set()
    for i, q in enumerate(QUERIES, 1):
        try:
            pmids = search_pmids(q)
            all_pmids.update(pmids)
            print(f"  Q{i}: +{len(pmids)} (total: {len(all_pmids)})")
            time.sleep(0.4)
        except Exception as e:
            print(f"  Q{i}: ERROR {e}")

    print(f"\n  Total unique PMIDs: {len(all_pmids)}")

    # Fetch in small batches (50 per batch)
    pmid_list = sorted(all_pmids)
    batch_size = 50
    all_articles = []

    print(f"  Fetching abstracts ({len(pmid_list)} PMIDs, batch={batch_size})...")
    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i:i + batch_size]
        articles = fetch_batch(batch)
        all_articles.extend(articles)

        if (i // batch_size + 1) % 10 == 0:
            print(f"    {i + len(batch)}/{len(pmid_list)} fetched ({len(all_articles)} with abstract)")

        # Rate limit
        time.sleep(0.4 if NCBI_API_KEY else 1.0)

    print(f"\n  Total articles with abstract: {len(all_articles)}")

    # Save
    out_path = OUT_DIR / "pubmed_stress_brain.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    # Stats
    years = [a["year"] for a in all_articles if a["year"] > 1900]
    if years:
        print(f"  Year range: {min(years)} - {max(years)}")
        from collections import Counter
        recent = sum(1 for y in years if y >= 2020)
        print(f"  Articles 2020+: {recent}")

    print(f"\n  Saved: {out_path} ({out_path.stat().st_size // 1024}KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
