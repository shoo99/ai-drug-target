"""
Knowledge Graph Manager — Neo4j interface for drug target discovery
Supports multi-database: AMR and Pruritus tracks
"""
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class KnowledgeGraph:
    def __init__(self, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=30,
            max_connection_lifetime=300,
            max_connection_pool_size=50,
        )
        self.database = database

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: dict = None, max_retries: int = 2):
        import time
        for attempt in range(max_retries + 1):
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(query, parameters or {})
                    return [record.data() for record in result]
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue
                raise

    def setup_amr_schema(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.gene_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Bacterium) REQUIRE b.taxon_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein) REQUIRE p.uniprot_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.pathway_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:ResistanceMechanism) REQUIRE r.mech_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (paper:Paper) REQUIRE paper.pmid IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.name)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Bacterium) ON (b.name)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.essential)",
        ]
        for q in constraints + indexes:
            self.run_query(q)
        print("[AMR] Schema created successfully")

    def setup_pruritus_schema(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.gene_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein) REQUIRE p.uniprot_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (dis:Disease) REQUIRE dis.disease_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tissue) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.pathway_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ct:ClinicalTrial) REQUIRE ct.nct_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (paper:Paper) REQUIRE paper.pmid IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.name)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX IF NOT EXISTS FOR (dis:Disease) ON (dis.name)",
            "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.target_score)",
        ]
        for q in constraints + indexes:
            self.run_query(q)
        print("[Pruritus] Schema created successfully")

    # --- AMR Node/Relationship creation ---

    def add_bacterium(self, taxon_id: str, name: str, gram: str = None, **props):
        query = """
        MERGE (b:Bacterium {taxon_id: $taxon_id})
        SET b.name = $name, b.gram = $gram
        SET b += $props
        RETURN b
        """
        return self.run_query(query, {
            "taxon_id": taxon_id, "name": name, "gram": gram, "props": props
        })

    def add_gene(self, gene_id: str, name: str, **props):
        query = """
        MERGE (g:Gene {gene_id: $gene_id})
        SET g.name = $name
        SET g += $props
        RETURN g
        """
        return self.run_query(query, {"gene_id": gene_id, "name": name, "props": props})

    def add_protein(self, uniprot_id: str, name: str, **props):
        query = """
        MERGE (p:Protein {uniprot_id: $uniprot_id})
        SET p.name = $name
        SET p += $props
        RETURN p
        """
        return self.run_query(query, {"uniprot_id": uniprot_id, "name": name, "props": props})

    def add_drug(self, drug_id: str, name: str, **props):
        query = """
        MERGE (d:Drug {drug_id: $drug_id})
        SET d.name = $name
        SET d += $props
        RETURN d
        """
        return self.run_query(query, {"drug_id": drug_id, "name": name, "props": props})

    def add_disease(self, disease_id: str, name: str, **props):
        query = """
        MERGE (dis:Disease {disease_id: $disease_id})
        SET dis.name = $name
        SET dis += $props
        RETURN dis
        """
        return self.run_query(query, {"disease_id": disease_id, "name": name, "props": props})

    def add_paper(self, pmid: str, title: str, year: int = None, **props):
        query = """
        MERGE (paper:Paper {pmid: $pmid})
        SET paper.title = $title, paper.year = $year
        SET paper += $props
        RETURN paper
        """
        return self.run_query(query, {
            "pmid": pmid, "title": title, "year": year, "props": props
        })

    # Whitelist of allowed labels and relationship types
    VALID_LABELS = {"Gene", "Protein", "Drug", "Disease", "Bacterium", "Pathway", "Paper",
                    "ResistanceMechanism", "Tissue", "ClinicalTrial"}
    VALID_REL_TYPES = {"ENCODES", "TARGETS", "ASSOCIATED_WITH", "BELONGS_TO", "ESSENTIAL_IN",
                       "IN_PATHWAY", "MENTIONED_IN", "CO_MENTIONED", "INHIBITS", "ACTIVATES",
                       "BINDS_TO", "REGULATES", "INDUCES", "SUPPRESSES", "MODULATES",
                       "PHOSPHORYLATES", "SIGNALS_THROUGH", "MEDIATES", "EXPRESSED_IN"}

    def add_relationship(self, from_label: str, from_key: str, from_value: str,
                         rel_type: str,
                         to_label: str, to_key: str, to_value: str,
                         **props):
        # Validate labels and relationship types to prevent injection
        import re
        for label in [from_label, to_label]:
            if label not in self.VALID_LABELS:
                if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label):
                    raise ValueError(f"Invalid label: {label}")
        if rel_type not in self.VALID_REL_TYPES:
            if not re.match(r'^[A-Z_][A-Z0-9_]*$', rel_type):
                raise ValueError(f"Invalid relationship type: {rel_type}")
        if not re.match(r'^[a-z_][a-z0-9_]*$', from_key):
            raise ValueError(f"Invalid key: {from_key}")
        if not re.match(r'^[a-z_][a-z0-9_]*$', to_key):
            raise ValueError(f"Invalid key: {to_key}")

        query = f"""
        MATCH (a:{from_label} {{{from_key}: $from_value}})
        MATCH (b:{to_label} {{{to_key}: $to_value}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        RETURN type(r)
        """
        return self.run_query(query, {
            "from_value": from_value, "to_value": to_value, "props": props
        })

    # --- Query helpers ---

    def get_gene_connections(self, gene_name: str):
        query = """
        MATCH (g:Gene {name: $name})-[r]-(connected)
        RETURN type(r) as relationship, labels(connected) as node_type,
               connected.name as connected_name
        """
        return self.run_query(query, {"name": gene_name})

    def get_essential_genes_without_drugs(self):
        query = """
        MATCH (g:Gene {essential: true})
        WHERE NOT (g)-[:ENCODES]->(:Protein)<-[:TARGETS]-(:Drug)
        RETURN g.gene_id, g.name, g.essentiality_score
        ORDER BY g.essentiality_score DESC
        """
        return self.run_query(query)

    def get_target_candidates(self, min_score: float = 0.5):
        query = """
        MATCH (g:Gene)
        WHERE g.target_score >= $min_score
        RETURN g.gene_id, g.name, g.target_score, g.druggability, g.novelty
        ORDER BY g.target_score DESC
        LIMIT 50
        """
        return self.run_query(query, {"min_score": min_score})

    def get_graph_stats(self):
        query = """
        CALL {
            MATCH (n) RETURN count(n) as nodes
        }
        CALL {
            MATCH ()-[r]->() RETURN count(r) as relationships
        }
        RETURN nodes, relationships
        """
        return self.run_query(query)
