"""
AI Target Prediction Model
Uses Node2Vec embeddings + Link Prediction on the knowledge graph
to discover novel drug target candidates
"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
from collections import defaultdict
from tqdm import tqdm

from src.common.knowledge_graph import KnowledgeGraph


class Node2Vec:
    """Lightweight Node2Vec implementation for graph embeddings."""

    def __init__(self, graph: nx.Graph, dimensions: int = 64,
                 walk_length: int = 30, num_walks: int = 200,
                 p: float = 1.0, q: float = 1.0):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.embeddings = {}

    def _random_walk(self, start_node) -> list:
        """Perform a biased random walk from start_node."""
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))
            if not neighbors:
                break
            if len(walk) == 1:
                walk.append(np.random.choice(neighbors))
            else:
                prev = walk[-2]
                probabilities = []
                for neighbor in neighbors:
                    if neighbor == prev:
                        probabilities.append(1.0 / self.p)
                    elif self.graph.has_edge(neighbor, prev):
                        probabilities.append(1.0)
                    else:
                        probabilities.append(1.0 / self.q)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                walk.append(np.random.choice(neighbors, p=probabilities))
        return walk

    def generate_walks(self) -> list[list]:
        """Generate random walks for all nodes."""
        walks = []
        nodes = list(self.graph.nodes())
        for _ in tqdm(range(self.num_walks), desc="Generating walks"):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(node)
                walks.append(walk)
        return walks

    def train(self):
        """Train Node2Vec embeddings using skip-gram-like approach."""
        print(f"[Node2Vec] Training on {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

        walks = self.generate_walks()

        # Build co-occurrence matrix from walks
        node_list = list(self.graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)

        # Simple SVD-based embedding (efficient for moderate graphs)
        window = 5
        cooccurrence = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        for walk in tqdm(walks, desc="Building co-occurrence"):
            for i, node in enumerate(walk):
                idx_i = node_to_idx[node]
                for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                    if i != j:
                        idx_j = node_to_idx[walk[j]]
                        cooccurrence[idx_i, idx_j] += 1

        # Apply PPMI (Positive Pointwise Mutual Information)
        total = cooccurrence.sum()
        if total > 0:
            row_sums = cooccurrence.sum(axis=1, keepdims=True)
            col_sums = cooccurrence.sum(axis=0, keepdims=True)
            expected = (row_sums * col_sums) / total
            with np.errstate(divide='ignore', invalid='ignore'):
                pmi = np.log2(cooccurrence / (expected + 1e-10) + 1e-10)
            ppmi = np.maximum(pmi, 0)
        else:
            ppmi = cooccurrence

        # SVD for dimensionality reduction
        print("[Node2Vec] Computing SVD embeddings...")
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        embeddings_matrix = U[:, :self.dimensions] * np.sqrt(S[:self.dimensions])

        self.embeddings = {
            node_list[i]: embeddings_matrix[i]
            for i in range(n_nodes)
        }
        print(f"[Node2Vec] Generated {len(self.embeddings)} embeddings "
              f"(dim={self.dimensions})")
        return self.embeddings

    def get_embedding(self, node) -> np.ndarray:
        return self.embeddings.get(node, np.zeros(self.dimensions))


class TargetPredictor:
    """Link prediction model for drug target discovery."""

    def __init__(self):
        self.kg = KnowledgeGraph()
        self.model = None
        self.scaler = StandardScaler()
        self.node2vec = None
        self.nx_graph = None

    def build_networkx_graph(self) -> nx.Graph:
        """Export Neo4j graph to NetworkX for analysis."""
        print("[Predictor] Building NetworkX graph from Neo4j...")

        # Get all nodes and relationships
        nodes_query = """
        MATCH (n)
        RETURN id(n) as id, labels(n) as labels, n.name as name,
               n.gene_id as gene_id, n.uniprot_id as uniprot_id
        """
        rels_query = """
        MATCH (a)-[r]->(b)
        RETURN id(a) as source, id(b) as target, type(r) as rel_type
        """

        nodes = self.kg.run_query(nodes_query)
        rels = self.kg.run_query(rels_query)

        G = nx.Graph()

        for node in nodes:
            node_id = node["id"]
            attrs = {
                "labels": node["labels"],
                "name": node.get("name", ""),
                "gene_id": node.get("gene_id", ""),
            }
            G.add_node(node_id, **attrs)

        for rel in rels:
            G.add_edge(rel["source"], rel["target"], rel_type=rel["rel_type"])

        print(f"[Predictor] NetworkX graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
        self.nx_graph = G
        return G

    def compute_features(self, node1, node2) -> np.ndarray:
        """Compute edge features for link prediction."""
        features = []

        # 1. Node2Vec embedding similarity (with zero-vector guard)
        emb1 = self.node2vec.get_embedding(node1)
        emb2 = self.node2vec.get_embedding(node2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(emb1, emb2) / (norm1 * norm2))
        features.append(cos_sim)

        # Hadamard product features (element-wise)
        hadamard = emb1 * emb2
        features.extend(hadamard[:8])  # First 8 dims of hadamard

        # 2. Graph topology features
        G = self.nx_graph

        # Common neighbors
        if G.has_node(node1) and G.has_node(node2):
            common_neighbors = len(list(nx.common_neighbors(G, node1, node2)))
            features.append(common_neighbors)

            # Jaccard coefficient
            preds = list(nx.jaccard_coefficient(G, [(node1, node2)]))
            features.append(preds[0][2] if preds else 0)

            # Adamic-Adar
            preds = list(nx.adamic_adar_index(G, [(node1, node2)]))
            features.append(preds[0][2] if preds else 0)

            # Preferential attachment
            preds = list(nx.preferential_attachment(G, [(node1, node2)]))
            features.append(preds[0][2] if preds else 0)

            # Degree
            features.append(G.degree(node1))
            features.append(G.degree(node2))
        else:
            features.extend([0] * 6)

        return np.array(features)

    def prepare_training_data(self) -> tuple:
        """Prepare positive and negative examples for link prediction."""
        G = self.nx_graph
        edges = list(G.edges())
        nodes = list(G.nodes())

        print(f"[Predictor] Preparing training data from {len(edges)} edges...")

        # Positive examples: random sample of existing edges (avoid ordering bias)
        np.random.shuffle(edges)
        positive_pairs = edges[:min(len(edges), 2000)]

        # Negative examples: random non-edges
        negative_pairs = []
        max_attempts = len(positive_pairs) * 10
        attempts = 0
        while len(negative_pairs) < len(positive_pairs) and attempts < max_attempts:
            n1 = np.random.choice(nodes)
            n2 = np.random.choice(nodes)
            if n1 != n2 and not G.has_edge(n1, n2):
                negative_pairs.append((n1, n2))
            attempts += 1

        print(f"  Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)}")

        # Compute features
        X, y = [], []
        for n1, n2 in tqdm(positive_pairs, desc="Positive features"):
            X.append(self.compute_features(n1, n2))
            y.append(1)
        for n1, n2 in tqdm(negative_pairs, desc="Negative features"):
            X.append(self.compute_features(n1, n2))
            y.append(0)

        return np.array(X), np.array(y)

    def train(self):
        """Train the full prediction pipeline."""
        # Step 1: Build graph
        self.build_networkx_graph()

        if self.nx_graph.number_of_nodes() < 10:
            print("[Predictor] Graph too small for training")
            return

        # Step 2: Train Node2Vec
        dims = min(64, self.nx_graph.number_of_nodes() - 1)
        self.node2vec = Node2Vec(
            self.nx_graph,
            dimensions=dims,
            walk_length=20,
            num_walks=50,
            p=1.0, q=0.5  # Bias towards exploring
        )
        self.node2vec.train()

        # Step 3: Prepare data & train classifier
        X, y = self.prepare_training_data()

        if len(X) < 20:
            print("[Predictor] Not enough training data")
            return

        X_scaled = self.scaler.fit_transform(X)

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )

        # Cross-validation
        scores = cross_val_score(self.model, X_scaled, y, cv=3, scoring="roc_auc")
        print(f"[Predictor] CV AUC-ROC: {scores.mean():.3f} (+/- {scores.std():.3f})")

        # Final training
        self.model.fit(X_scaled, y)
        print("[Predictor] Model trained successfully")

    def predict_novel_targets(self, track: str = "amr") -> pd.DataFrame:
        """Predict novel drug targets by scoring unconnected gene-drug pairs."""
        G = self.nx_graph

        # Find gene nodes without drug connections
        gene_nodes = []
        drug_nodes = []
        for node, attrs in G.nodes(data=True):
            labels = attrs.get("labels", [])
            if "Gene" in labels:
                gene_nodes.append(node)
            elif "Drug" in labels:
                drug_nodes.append(node)

        print(f"[Predictor] Scoring {len(gene_nodes)} genes against "
              f"{len(drug_nodes)} drugs...")

        if not drug_nodes:
            # If no drugs, score gene-gene novel connections
            print("[Predictor] No drug nodes, scoring gene-gene predictions...")
            return self._predict_gene_associations(gene_nodes)

        predictions = []
        for gene_node in tqdm(gene_nodes, desc="Predicting targets"):
            gene_attrs = G.nodes[gene_node]
            gene_name = gene_attrs.get("name", "")

            # Check if already targeted
            already_targeted = any(
                G.has_edge(gene_node, drug) for drug in drug_nodes
            )

            # Score against all drugs
            scores = []
            for drug_node in drug_nodes[:50]:  # Sample for efficiency
                if not G.has_edge(gene_node, drug_node):
                    features = self.compute_features(gene_node, drug_node)
                    features_scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(features_scaled)[0][1]
                    scores.append(prob)

            if scores:
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                predictions.append({
                    "node_id": gene_node,
                    "gene_name": gene_name,
                    "gene_id": gene_attrs.get("gene_id", ""),
                    "avg_target_score": round(avg_score, 4),
                    "max_target_score": round(max_score, 4),
                    "already_targeted": already_targeted,
                    "degree": G.degree(gene_node),
                })

        df = pd.DataFrame(predictions)
        if not df.empty:
            df = df.sort_values("avg_target_score", ascending=False)
        return df

    def _predict_gene_associations(self, gene_nodes: list) -> pd.DataFrame:
        """For graphs without drugs, predict novel gene-gene associations."""
        G = self.nx_graph
        predictions = []

        for gene_node in tqdm(gene_nodes, desc="Scoring genes"):
            gene_attrs = G.nodes[gene_node]
            gene_name = gene_attrs.get("name", "")

            # Score connectivity potential
            emb = self.node2vec.get_embedding(gene_node)
            emb_norm = np.linalg.norm(emb)

            # Centrality-based novelty
            degree = G.degree(gene_node)
            neighbors = list(G.neighbors(gene_node))

            # Average similarity to all other gene nodes
            similarities = []
            sample = np.random.choice(gene_nodes, min(100, len(gene_nodes)), replace=False)
            for other in sample:
                if other != gene_node:
                    other_emb = self.node2vec.get_embedding(other)
                    sim = np.dot(emb, other_emb) / (emb_norm * np.linalg.norm(other_emb) + 1e-8)
                    similarities.append(sim)

            predictions.append({
                "node_id": gene_node,
                "gene_name": gene_name,
                "gene_id": gene_attrs.get("gene_id", ""),
                "embedding_centrality": round(np.mean(similarities) if similarities else 0, 4),
                "degree": degree,
                "avg_neighbor_degree": round(
                    np.mean([G.degree(n) for n in neighbors]) if neighbors else 0, 2
                ),
            })

        df = pd.DataFrame(predictions)
        if not df.empty:
            df = df.sort_values("embedding_centrality", ascending=False)
        return df

    def close(self):
        self.kg.close()
