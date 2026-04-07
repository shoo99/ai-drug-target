"""
NLP Gene-Disease Relationship Extractor
Extracts gene mentions and relationships from PubMed abstracts
using pattern matching + co-occurrence analysis
"""
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm


# Common gene name patterns (human genes are typically uppercase, 2-6 chars)
GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,6})\b')

# Filter out common abbreviations that are NOT genes
NON_GENE_WORDS = {
    # Common English words
    "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD", "HIS",
    "HER", "ITS", "OUR", "WHO", "HOW", "ALL", "CAN", "DID", "GET", "HIM",
    "LET", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "DAY", "USE", "MAN",
    "IN", "OF", "TO", "IS", "IT", "AT", "BY", "ON", "AN", "AS", "IF",
    "DO", "NO", "SO", "UP", "WE", "HE", "ME", "MY", "BE",
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
    # Paper section headers / common abstract words (CRITICAL FILTER)
    "RESULTS", "METHODS", "BACKGROUND", "CONCLUSIONS", "CONCLUSION",
    "OBJECTIVE", "OBJECTIVES", "PURPOSE", "DESIGN", "SETTING", "PATIENTS",
    "FINDINGS", "SIGNIFICANCE", "INTRODUCTION", "DISCUSSION", "ABSTRACT",
    "STUDY", "STUDIES", "REVIEW", "ANALYSIS", "DATA", "EVIDENCE",
    "AREAS", "COVERED", "EXPERT", "OPINION", "AIMS", "AIM",
    "TRIAL", "TRIALS", "GROUP", "GROUPS", "CONTROL", "TREATMENT",
    "DOSE", "DOSES", "SAFETY", "EFFICACY", "OUTCOME", "OUTCOMES",
    "RISK", "FACTOR", "FACTORS", "CASE", "CASES", "REPORT",
    "TABLE", "FIGURE", "FIGURES", "TABLES", "MODEL", "MODELS",
    "CELL", "CELLS", "GENE", "GENES", "PROTEIN", "PROTEINS",
    "DRUG", "DRUGS", "THERAPY", "AGENT", "AGENTS",
    "HIGH", "LOW", "LEVEL", "LEVELS", "TYPE", "TYPES",
    "ROLE", "EFFECT", "EFFECTS", "RESPONSE", "ACTIVITY",
    "BASED", "NOVEL", "RECENT", "CURRENT", "TOTAL", "MAJOR",
    "ACUTE", "CHRONIC", "EARLY", "LATE", "FIRST", "SECOND",
    # Lab/medical abbreviations
    "DNA", "RNA", "PCR", "USA", "FDA", "WHO", "CDC", "NIH", "BMI", "ICU",
    "HIV", "AIDS", "COVID", "SARS", "MRSA", "VRE", "MDR", "XDR", "PDR",
    "MIC", "MBC", "CFU", "OD", "PH", "UV", "IR", "NMR", "MS", "GC",
    "HPLC", "SDS", "PAGE", "ELISA", "FACS", "FISH", "ISH", "IHC",
    "WT", "KO", "TG", "OE", "SI", "SD", "SE", "CI", "OR", "HR", "RR",
    "NS", "VS", "ED", "AD", "ER", "PR", "IC", "EC", "IV", "IM", "SC",
    "PO", "BID", "TID", "QID", "PRN", "QD", "HS", "AC", "PC",
    "AUC", "PK", "PD", "ADME", "GLP", "GMP", "GCP", "IRB",
    "SAE", "AE", "DLT", "MTD", "RP2D", "ORR", "CR", "PR", "SD", "PD",
    "OS", "PFS", "DFS", "TTP", "TTR", "DOT",
    "ESKAPE", "AMR", "ABR", "AST", "CLSI", "EUCAST",
    # Common non-gene medical/science abbreviations
    "BP", "HR", "ECG", "EEG", "CT", "MRI", "PET", "CBC",
    "OM", "QS", "MM", "OA", "MD", "RX", "TX", "DX", "SX",
    "AB", "AG", "IG", "EMA", "WHO", "LEV", "VCM", "CUR",
    "TS", "CM", "ML", "MG", "KG", "UG", "NM", "UM",
    "MIN", "MAX", "AVG", "MEAN", "SEM", "STD",
    "CRAB", "QIDP", "RG6006",  # Specific noise from AMR results
    "DRG", "CNS", "PNS", "GI", "UTI", "URI", "COPD",
    "NSAID", "SSRI", "SNRI", "ACE", "ARB", "PPI",
    "PCG", "UDP",
}

# Known gene families for better detection
GENE_FAMILIES = {
    "IL": r'IL-?\d+[A-Z]?[A-Z]?',      # Interleukins
    "TLR": r'TLR\d+',                     # Toll-like receptors
    "CXCL": r'CXCL\d+',                   # Chemokines
    "CCL": r'CCL\d+',
    "CXCR": r'CXCR\d+',
    "CCR": r'CCR\d+',
    "MMP": r'MMP-?\d+',                   # Matrix metalloproteinases
    "JAK": r'JAK\d?',
    "STAT": r'STAT\d[A-Z]?',
    "TRP": r'TRP[ACMV]\d+',              # TRP channels
    "MRGPR": r'MRGPR[A-Z]\d*',           # Mas-related GPCRs
    "HRH": r'HRH\d',                      # Histamine receptors
    "TNF": r'TNF[A-Z]?',
    "IFN": r'IFN[A-Z]?',
    "TGF": r'TGF[A-Z]\d?',
    "FGF": r'FGF\d+',
    "VEGF": r'VEGF[A-Z]?',
    "PDGF": r'PDGF[A-Z]?',
    "EGF": r'EGFR?',
}


class NLPExtractor:
    def __init__(self, known_genes: list[str] = None):
        self.known_genes = set(g.upper() for g in (known_genes or []))
        self.family_patterns = {
            name: re.compile(pattern) for name, pattern in GENE_FAMILIES.items()
        }

    def extract_genes_from_text(self, text: str) -> list[str]:
        """Extract gene mentions from text."""
        found = set()

        # 1. Match known genes directly
        text_upper = text.upper()
        for gene in self.known_genes:
            if gene in text_upper:
                found.add(gene)

        # 2. Match gene family patterns
        for family, pattern in self.family_patterns.items():
            matches = pattern.findall(text)
            for m in matches:
                found.add(m.upper().replace("-", ""))

        # 3. General gene pattern matching — strict filtering
        candidates = GENE_PATTERN.findall(text)
        for candidate in candidates:
            if candidate in NON_GENE_WORDS:
                continue
            if len(candidate) < 3:
                continue  # 2-letter codes too noisy
            # Must look like a gene: starts with letter, contains digits or
            # is a known gene family prefix
            has_digit = any(c.isdigit() for c in candidate)
            known_prefix = any(candidate.startswith(p) for p in
                              ["IL", "TLR", "CCL", "CCR", "CXC", "MMP", "JAK",
                               "STAT", "TRP", "TNF", "IFN", "TGF", "FGF", "SCN",
                               "HRH", "MUR", "FTS", "LPX", "BAM", "GYR", "RPO",
                               "FAB", "FOL", "DNA", "SEC", "WAL", "ISP", "ACC"])
            if has_digit or known_prefix or candidate in self.known_genes:
                found.add(candidate)

        return sorted(found)

    def extract_relationships(self, text: str, genes: list[str]) -> list[dict]:
        """Extract gene-gene and gene-function relationships from text."""
        relationships = []

        # Relationship keywords
        interaction_terms = {
            "inhibit": "INHIBITS",
            "activate": "ACTIVATES",
            "bind": "BINDS_TO",
            "regulate": "REGULATES",
            "induce": "INDUCES",
            "suppress": "SUPPRESSES",
            "target": "TARGETS",
            "modulate": "MODULATES",
            "phosphorylat": "PHOSPHORYLATES",
            "signal": "SIGNALS_THROUGH",
            "mediate": "MEDIATES",
            "express": "EXPRESSED_IN",
        }

        text_lower = text.lower()
        sentences = text.split(". ")

        for sentence in sentences:
            sentence_lower = sentence.lower()
            genes_in_sentence = [g for g in genes if g.upper() in sentence.upper()]

            if len(genes_in_sentence) >= 2:
                # Co-occurrence of 2+ genes in same sentence
                for i, g1 in enumerate(genes_in_sentence):
                    for g2 in genes_in_sentence[i+1:]:
                        rel_type = "CO_MENTIONED"
                        for term, rtype in interaction_terms.items():
                            if term in sentence_lower:
                                rel_type = rtype
                                break
                        relationships.append({
                            "gene1": g1,
                            "gene2": g2,
                            "relationship": rel_type,
                            "context": sentence[:200],
                        })

        return relationships

    def process_articles(self, articles: list[dict]) -> dict:
        """Process a batch of PubMed articles and extract all gene mentions."""
        gene_counts = Counter()
        gene_cooccurrence = defaultdict(Counter)
        gene_articles = defaultdict(list)
        all_relationships = []

        for article in tqdm(articles, desc="NLP processing"):
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            pmid = article.get("pmid", "")

            genes = self.extract_genes_from_text(text)
            for gene in genes:
                gene_counts[gene] += 1
                gene_articles[gene].append(pmid)

            # Co-occurrence
            for i, g1 in enumerate(genes):
                for g2 in genes[i+1:]:
                    gene_cooccurrence[g1][g2] += 1
                    gene_cooccurrence[g2][g1] += 1

            # Relationships
            rels = self.extract_relationships(text, genes)
            all_relationships.extend(rels)

        return {
            "gene_counts": gene_counts,
            "gene_cooccurrence": dict(gene_cooccurrence),
            "gene_articles": dict(gene_articles),
            "relationships": all_relationships,
        }

    def get_trending_genes(self, articles: list[dict], top_n: int = 50) -> pd.DataFrame:
        """Identify genes with increasing publication frequency (trending)."""
        year_gene_counts = defaultdict(Counter)

        for article in articles:
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            year = article.get("year")
            if not year:
                continue

            genes = self.extract_genes_from_text(text)
            for gene in genes:
                year_gene_counts[year][gene] += 1

        # Calculate trend scores
        years = sorted(year_gene_counts.keys())
        if len(years) < 2:
            return pd.DataFrame()

        recent_years = years[-3:] if len(years) >= 3 else years
        older_years = years[:-3] if len(years) > 3 else []

        trend_scores = {}
        all_genes = set()
        for year_counts in year_gene_counts.values():
            all_genes.update(year_counts.keys())

        for gene in all_genes:
            recent_count = sum(year_gene_counts[y].get(gene, 0) for y in recent_years)
            older_count = sum(year_gene_counts[y].get(gene, 0) for y in older_years) if older_years else 0
            total = recent_count + older_count

            if total >= 2:
                trend = recent_count / max(older_count, 1)
                trend_scores[gene] = {
                    "gene": gene,
                    "total_mentions": total,
                    "recent_mentions": recent_count,
                    "older_mentions": older_count,
                    "trend_score": round(trend, 2),
                }

        df = pd.DataFrame(trend_scores.values())
        if not df.empty:
            df = df.sort_values("trend_score", ascending=False).head(top_n)
        return df
