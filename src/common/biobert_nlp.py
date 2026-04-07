"""
BioBERT-powered NLP Pipeline — Professional-grade biomedical NER and relation extraction.
Replaces keyword matching with transformer-based named entity recognition.
"""
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    pipeline as hf_pipeline
)
from config.settings import DATA_DIR

MODELS_CACHE = DATA_DIR / "models_cache"


class BioBERTExtractor:
    """BioBERT-based Named Entity Recognition for biomedical text."""

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.2"):
        MODELS_CACHE.mkdir(parents=True, exist_ok=True)
        self.ner_pipeline = None
        self.model_name = model_name
        self._load_ner()

    def _load_ner(self):
        """Load BioBERT NER model (uses general NER fine-tuned on biomedical text)."""
        try:
            # Use a biomedical NER model fine-tuned for gene/protein recognition
            ner_model = "alvaroalon2/biobert_genetic_ner"
            self.ner_pipeline = hf_pipeline(
                "ner",
                model=ner_model,
                tokenizer=ner_model,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
            print(f"  BioBERT NER loaded: {ner_model}")
        except Exception as e:
            print(f"  BioBERT NER model failed ({e}), trying fallback...")
            try:
                # Fallback: general biomedical NER
                ner_model = "d4data/biomedical-ner-all"
                self.ner_pipeline = hf_pipeline(
                    "ner",
                    model=ner_model,
                    tokenizer=ner_model,
                    aggregation_strategy="simple",
                    device=-1,
                )
                print(f"  Fallback NER loaded: {ner_model}")
            except Exception as e2:
                print(f"  All NER models failed: {e2}")
                self.ner_pipeline = None

    def extract_entities(self, text: str) -> list[dict]:
        """Extract biomedical named entities from text using BioBERT."""
        if not self.ner_pipeline:
            return []

        # Truncate to model max length
        text = text[:512]

        try:
            entities = self.ner_pipeline(text)
            results = []
            for ent in entities:
                results.append({
                    "text": ent["word"].strip(),
                    "type": ent["entity_group"],
                    "score": round(ent["score"], 4),
                    "start": ent["start"],
                    "end": ent["end"],
                })
            return results
        except Exception as e:
            return []

    def extract_genes(self, text: str) -> list[dict]:
        """Extract gene/protein mentions specifically."""
        entities = self.extract_entities(text)
        # Support multiple NER model entity tag formats
        gene_types = {
            "GENE", "PROTEIN", "Gene", "Protein", "gene", "protein",
            "GENE_OR_GENE_PRODUCT", "DNA", "RNA",
            "GENETIC",  # alvaroalon2/biobert_genetic_ner uses this tag
            "Gene_or_gene_product",
        }
        genes = []
        seen = set()
        for ent in entities:
            if ent["type"] in gene_types and ent["score"] > 0.5:
                name = ent["text"].upper().strip()
                # Clean up BioBERT artifacts
                name = name.replace("##", "").strip()
                # Skip descriptive phrases (keep only actual gene symbols)
                if len(name) > 15:
                    continue  # Gene symbols are short
                if " " in name and not name.startswith("IL"):
                    continue  # Multi-word = probably description, not gene
                if len(name) < 2:
                    continue
                if name not in seen:
                    seen.add(name)
                    genes.append({
                        "name": name,
                        "type": ent["type"],
                        "confidence": ent["score"],
                    })
        return genes

    def extract_relations(self, text: str, genes: list[str]) -> list[dict]:
        """
        Extract gene-gene relations from text using co-occurrence + context analysis.
        More sophisticated than keyword matching — uses sentence structure.
        """
        relations = []
        sentences = re.split(r'[.!?]\s+', text)

        # Relation patterns (regex-based for speed, BioBERT for entity detection)
        RELATION_PATTERNS = {
            "INHIBITS": re.compile(r'inhibit[s|ed|ing]?\b|suppress[es|ed]?\b|block[s|ed]?\b|antagoni[sz]', re.I),
            "ACTIVATES": re.compile(r'activat[es|ed|ing]?\b|stimulat[es|ed]?\b|induc[es|ed]?\b|promot[es|ed]?\b', re.I),
            "REGULATES": re.compile(r'regulat[es|ed|ing]?\b|modulat[es|ed|ing]?\b|control', re.I),
            "BINDS_TO": re.compile(r'bind[s]?\b|interact[s]?\b|associat[es|ed]?\b|complex', re.I),
            "PHOSPHORYLATES": re.compile(r'phosphorylat[es|ed|ing]?\b', re.I),
            "UPREGULATES": re.compile(r'upregulat[es|ed]?\b|overexpress[es|ed]?\b|elevat[es|ed]?\b', re.I),
            "DOWNREGULATES": re.compile(r'downregulat[es|ed]?\b|underexpress[es|ed]?\b|reduc[es|ed]?\b', re.I),
            "TARGETS": re.compile(r'target[s|ed|ing]?\b|direct[s|ed]?\b', re.I),
        }

        for sentence in sentences:
            # Find which genes appear in this sentence
            sent_upper = sentence.upper()
            genes_in_sent = [g for g in genes if g.upper() in sent_upper]

            if len(genes_in_sent) < 2:
                continue

            # Determine relationship type from context
            rel_type = "CO_EXPRESSED"
            rel_confidence = 0.5
            for rtype, pattern in RELATION_PATTERNS.items():
                if pattern.search(sentence):
                    rel_type = rtype
                    rel_confidence = 0.8
                    break

            # Create pairwise relations
            for i, g1 in enumerate(genes_in_sent):
                for g2 in genes_in_sent[i+1:]:
                    relations.append({
                        "gene1": g1,
                        "gene2": g2,
                        "relation": rel_type,
                        "confidence": rel_confidence,
                        "context": sentence[:200],
                    })

        return relations

    def process_articles(self, articles: list[dict]) -> dict:
        """Process PubMed articles with BioBERT NER."""
        gene_counts = Counter()
        all_genes = []
        all_relations = []
        gene_articles = defaultdict(list)

        for article in tqdm(articles, desc="BioBERT NLP"):
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            pmid = article.get("pmid", "")

            # BioBERT gene extraction
            genes = self.extract_genes(text)
            gene_names = [g["name"] for g in genes]

            for gene in genes:
                gene_counts[gene["name"]] += 1
                gene_articles[gene["name"]].append(pmid)
                all_genes.append({**gene, "pmid": pmid})

            # Relation extraction
            rels = self.extract_relations(text, gene_names)
            for rel in rels:
                rel["pmid"] = pmid
            all_relations.extend(rels)

        return {
            "gene_counts": gene_counts,
            "genes": all_genes,
            "relations": all_relations,
            "gene_articles": dict(gene_articles),
            "n_articles": len(articles),
            "n_unique_genes": len(gene_counts),
            "n_relations": len(all_relations),
        }
