"""
AMR Graph Loader — Load collected AMR data into Neo4j knowledge graph
"""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.common.knowledge_graph import KnowledgeGraph
from config.settings import AMR_CONFIG

DATA_DIR = AMR_CONFIG["data_dir"]


class AMRGraphLoader:
    def __init__(self):
        self.kg = KnowledgeGraph()

    def close(self):
        self.kg.close()

    def load_eskape_organisms(self):
        """Load ESKAPE organisms as Bacterium nodes."""
        print("[Graph] Loading ESKAPE organisms...")
        gram_types = {
            "Enterococcus faecium": "positive",
            "Staphylococcus aureus": "positive",
            "Klebsiella pneumoniae": "negative",
            "Acinetobacter baumannii": "negative",
            "Pseudomonas aeruginosa": "negative",
            "Enterobacter cloacae": "negative",
        }
        for org in AMR_CONFIG["eskape_organisms"]:
            taxon_id = org.replace(" ", "_").lower()
            self.kg.add_bacterium(taxon_id, org, gram=gram_types.get(org))
        print(f"  Loaded {len(AMR_CONFIG['eskape_organisms'])} organisms")

    def load_uniprot_data(self):
        """Load UniProt essential proteins into graph."""
        csv_path = DATA_DIR / "uniprot_essential_proteins.csv"
        if not csv_path.exists():
            print("[Graph] No UniProt data found, skipping")
            return

        df = pd.read_csv(csv_path)
        print(f"[Graph] Loading {len(df)} UniProt proteins...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading proteins"):
            uniprot_id = str(row.get("uniprot_id", ""))
            gene_name = str(row.get("gene_name", ""))
            protein_name = str(row.get("protein_name", ""))
            organism = str(row.get("organism", ""))
            has_structure = bool(row.get("has_structure", False))

            if not uniprot_id:
                continue

            # Add protein node
            self.kg.add_protein(uniprot_id, protein_name,
                                has_structure=has_structure)

            # Add gene node if gene name exists
            if gene_name and gene_name != "nan":
                gene_id = f"{gene_name}_{organism.replace(' ', '_')}"
                self.kg.add_gene(gene_id, gene_name,
                                 essential=True, organism=organism)

                # Gene -> Protein
                self.kg.add_relationship(
                    "Gene", "gene_id", gene_id,
                    "ENCODES",
                    "Protein", "uniprot_id", uniprot_id
                )

                # Gene -> Bacterium
                taxon_id = organism.replace(" ", "_").lower()
                self.kg.add_relationship(
                    "Gene", "gene_id", gene_id,
                    "BELONGS_TO",
                    "Bacterium", "taxon_id", taxon_id
                )

    def load_chembl_data(self):
        """Load ChEMBL drug-target mechanisms."""
        csv_path = DATA_DIR / "chembl_antibiotics.csv"
        if not csv_path.exists():
            print("[Graph] No ChEMBL data found, skipping")
            return

        df = pd.read_csv(csv_path)
        print(f"[Graph] Loading {len(df)} ChEMBL drug-target mechanisms...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading drugs"):
            drug_id = str(row.get("drug_chembl_id", ""))
            drug_name = str(row.get("drug_name", ""))
            target_id = str(row.get("target_chembl_id", ""))
            target_name = str(row.get("target_name", ""))
            mechanism = str(row.get("mechanism", ""))

            if not drug_id or drug_id == "nan":
                continue

            self.kg.add_drug(drug_id, drug_name, source="chembl")

            if target_id and target_id != "nan":
                self.kg.add_protein(target_id, target_name, source="chembl")
                self.kg.add_relationship(
                    "Drug", "drug_id", drug_id,
                    "TARGETS",
                    "Protein", "uniprot_id", target_id,
                    mechanism=mechanism
                )

    def load_opentargets_data(self):
        """Load OpenTargets associations."""
        csv_path = DATA_DIR / "opentargets_amr.csv"
        if not csv_path.exists():
            print("[Graph] No OpenTargets data found, skipping")
            return

        df = pd.read_csv(csv_path)
        print(f"[Graph] Loading {len(df)} OpenTargets associations...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading targets"):
            ensembl_id = str(row.get("ensembl_id", ""))
            symbol = str(row.get("symbol", ""))
            score = float(row.get("overall_score", 0))
            genetic = float(row.get("genetic_association", 0))
            known_drug = float(row.get("known_drug", 0))

            if not ensembl_id or ensembl_id == "nan":
                continue

            gene_id = f"OT_{ensembl_id}"
            self.kg.add_gene(gene_id, symbol,
                             ensembl_id=ensembl_id,
                             opentargets_score=score,
                             genetic_association=genetic,
                             known_drug_score=known_drug)

    def load_pubmed_data(self):
        """Load PubMed articles and link to genes mentioned."""
        json_path = DATA_DIR / "pubmed_amr_articles.json"
        if not json_path.exists():
            print("[Graph] No PubMed data found, skipping")
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
                              mesh_terms=",".join(article.get("mesh_terms", [])))

    def load_all(self):
        """Load all AMR data into knowledge graph."""
        print("=" * 60)
        print("AMR GRAPH LOADING — Starting")
        print("=" * 60)

        self.load_eskape_organisms()
        self.load_uniprot_data()
        self.load_chembl_data()
        self.load_opentargets_data()
        self.load_pubmed_data()

        stats = self.kg.get_graph_stats()
        print("=" * 60)
        print("AMR GRAPH LOADING — Complete")
        print(f"  Graph stats: {stats}")
        print("=" * 60)

        self.close()
