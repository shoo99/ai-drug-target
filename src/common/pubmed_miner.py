"""
PubMed Literature Mining — Fetch and extract gene-disease relationships from papers
"""
import time
import xml.etree.ElementTree as ET
import requests
from tqdm import tqdm
from config.settings import NCBI_EMAIL, NCBI_API_KEY

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedMiner:
    def __init__(self):
        self.params = {}
        if NCBI_EMAIL:
            self.params["email"] = NCBI_EMAIL
        if NCBI_API_KEY:
            self.params["api_key"] = NCBI_API_KEY

    def search(self, query: str, max_results: int = 1000) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        url = f"{EUTILS_BASE}/esearch.fcgi"
        params = {
            **self.params,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        print(f"[PubMed] Found {len(pmids)} articles for: {query}")
        return pmids

    def fetch_abstracts(self, pmids: list[str], batch_size: int = 200) -> list[dict]:
        """Fetch article details (title, abstract, year, MeSH) for given PMIDs."""
        articles = []
        for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching abstracts"):
            batch = pmids[i:i + batch_size]
            url = f"{EUTILS_BASE}/efetch.fcgi"
            params = {
                **self.params,
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            articles.extend(self._parse_xml(resp.text))
            time.sleep(0.34 if NCBI_API_KEY else 1.0)
        return articles

    def _parse_xml(self, xml_text: str) -> list[dict]:
        """Parse PubMed XML response into structured dicts."""
        root = ET.fromstring(xml_text)
        articles = []
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                medline = article_elem.find("MedlineCitation")
                pmid = medline.findtext("PMID")
                art = medline.find("Article")
                title = art.findtext("ArticleTitle", "")

                # Abstract
                abstract_parts = []
                abs_elem = art.find("Abstract")
                if abs_elem is not None:
                    for at in abs_elem.findall("AbstractText"):
                        label = at.get("Label", "")
                        text = "".join(at.itertext())
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                # Year
                pub_date = art.find(".//PubDate")
                year = None
                if pub_date is not None:
                    y = pub_date.findtext("Year")
                    if y:
                        year = int(y)

                # MeSH terms
                mesh_terms = []
                for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                    mesh_terms.append(mesh.text)

                # Keywords
                keywords = []
                for kw in medline.findall(".//Keyword"):
                    if kw.text:
                        keywords.append(kw.text)

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

    def search_and_fetch(self, query: str, max_results: int = 500) -> list[dict]:
        """Convenience: search + fetch in one call."""
        pmids = self.search(query, max_results)
        if not pmids:
            return []
        return self.fetch_abstracts(pmids)

    def extract_gene_mentions(self, text: str, gene_list: list[str]) -> list[str]:
        """Simple gene mention extraction from text using a known gene list."""
        text_upper = text.upper()
        found = []
        for gene in gene_list:
            if gene.upper() in text_upper:
                found.append(gene)
        return found
