"""
Pruritus Graph Loader — Load collected pruritus data into Neo4j knowledge graph
"""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.common.knowledge_graph import KnowledgeGraph
from config.settings import PRURITUS_CONFIG

DATA_DIR = PRURITUS_CONFIG["data_dir"]


class PruritusGraphLoader:
    def __init__(self):
        self.kg = KnowledgeGraph()

    def close(self):
        self.kg.close()

    def load_diseases(self):
        """Load pruritus-related disease nodes."""
        print("[Graph] Loading pruritus disease nodes...")
        diseases = [
            ("pruritus", "Pruritus (Chronic Itch)"),
            ("atopic_dermatitis", "Atopic Dermatitis"),
            ("uremic_pruritus", "Uremic Pruritus"),
            ("cholestatic_pruritus", "Cholestatic Pruritus"),
            ("neuropathic_itch", "Neuropathic Itch"),
            ("psoriatic_pruritus", "Psoriatic Pruritus"),
        ]
        for did, name in diseases:
            self.kg.add_disease(did, name)
        print(f"  Loaded {len(diseases)} disease subtypes")

    def load_known_targets(self):
        """Load known pruritus targets with details."""
        csv_path = DATA_DIR / "known_targets_details.csv"
        if not csv_path.exists():
            print("[Graph] No known targets data, skipping")
            return

        df = pd.read_csv(csv_path)
        print(f"[Graph] Loading {len(df)} known pruritus targets...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Known targets"):
            gene_symbol = str(row.get("gene_symbol", ""))
            uniprot_id = str(row.get("uniprot_id", ""))
            protein_name = str(row.get("protein_name", ""))
            function = str(row.get("function", ""))
            has_structure = bool(row.get("has_structure", False))

            if not gene_symbol or gene_symbol == "nan":
                continue

            gene_id = f"HS_{gene_symbol}"
            self.kg.add_gene(gene_id, gene_symbol,
                             organism="Homo sapiens",
                             is_known_target=True)

            if uniprot_id and uniprot_id != "nan":
                self.kg.add_protein(uniprot_id, protein_name,
                                    function=function[:300],
                                    has_structure=has_structure)
                self.kg.add_relationship(
                    "Gene", "gene_id", gene_id,
                    "ENCODES",
                    "Protein", "uniprot_id", uniprot_id
                )

            # Link to pruritus disease
            self.kg.add_relationship(
                "Gene", "gene_id", gene_id,
                "ASSOCIATED_WITH",
                "Disease", "disease_id", "pruritus",
                evidence="known_target", confidence="high"
            )

    def load_opentargets_data(self):
        """Load OpenTargets associations."""
        csv_path = DATA_DIR / "opentargets_pruritus.csv"
        if not csv_path.exists():
            print("[Graph] No OpenTargets data, skipping")
            return

        df = pd.read_csv(csv_path)
        print(f"[Graph] Loading {len(df)} OpenTargets pruritus targets...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="OT targets"):
            ensembl_id = str(row.get("ensembl_id", ""))
            symbol = str(row.get("symbol", ""))
            score = float(row.get("overall_score", 0))
            genetic = float(row.get("genetic_association", 0))
            source_disease = str(row.get("source_disease", ""))

            if not ensembl_id or ensembl_id == "nan":
                continue

            gene_id = f"HS_{symbol}"
            self.kg.add_gene(gene_id, symbol,
                             ensembl_id=ensembl_id,
                             organism="Homo sapiens",
                             opentargets_score=score,
                             genetic_association=genetic)

            # Link to relevant disease
            disease_map = {
                "atopic dermatitis": "atopic_dermatitis",
                "pruritus": "pruritus",
                "eczema": "atopic_dermatitis",
            }
            disease_id = disease_map.get(source_disease, "pruritus")
            self.kg.add_relationship(
                "Gene", "gene_id", gene_id,
                "ASSOCIATED_WITH",
                "Disease", "disease_id", disease_id,
                source="opentargets", score=score
            )

    def load_gwas_data(self):
        """Load GWAS associations."""
        csv_path = DATA_DIR / "gwas_pruritus.csv"
        if not csv_path.exists():
            print("[Graph] No GWAS data, skipping")
            return

        df = pd.read_csv(csv_path)
        print(f"[Graph] Loading {len(df)} GWAS associations...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="GWAS"):
            gene_name = str(row.get("gene", ""))
            pvalue = row.get("pvalue", 1)
            trait = str(row.get("trait", ""))

            if not gene_name or gene_name == "nan":
                continue

            gene_id = f"HS_{gene_name}"
            self.kg.add_gene(gene_id, gene_name,
                             organism="Homo sapiens",
                             gwas_pvalue=float(pvalue) if pvalue else None)

    def load_pubmed_data(self):
        """Load PubMed articles."""
        json_path = DATA_DIR / "pubmed_pruritus_articles.json"
        if not json_path.exists():
            print("[Graph] No PubMed data, skipping")
            return

        with open(json_path) as f:
            articles = json.load(f)

        print(f"[Graph] Loading {len(articles)} PubMed articles...")
        for article in tqdm(articles[:500], desc="Loading papers"):
            pmid = article.get("pmid", "")
            title = article.get("title", "")
            year = article.get("year")

            if not pmid:
                continue

            self.kg.add_paper(pmid, title, year=year,
                              track="pruritus")

    def load_all(self):
        """Load all pruritus data into knowledge graph."""
        print("=" * 60)
        print("PRURITUS GRAPH LOADING — Starting")
        print("=" * 60)

        self.load_diseases()
        self.load_known_targets()
        self.load_opentargets_data()
        self.load_gwas_data()
        self.load_pubmed_data()

        stats = self.kg.get_graph_stats()
        print("=" * 60)
        print("PRURITUS GRAPH LOADING — Complete")
        print(f"  Graph stats: {stats}")
        print("=" * 60)

        self.close()
