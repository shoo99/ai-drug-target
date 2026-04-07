"""
AI Drug Target Discovery Platform — Configuration
Dual Track: AMR (Antimicrobial Resistance) + Chronic Pruritus
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# PubMed / NCBI
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# API endpoints
OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
DISGENET_API = "https://www.disgenet.org/api"
CARD_RGI_URL = "https://card.mcmaster.ca/download"
STRING_API = "https://string-db.org/api"

# Track-specific settings
AMR_CONFIG = {
    "eskape_organisms": [
        "Enterococcus faecium",
        "Staphylococcus aureus",
        "Klebsiella pneumoniae",
        "Acinetobacter baumannii",
        "Pseudomonas aeruginosa",
        "Enterobacter cloacae",
    ],
    "data_dir": DATA_DIR / "amr",
    "min_essentiality_score": 0.7,
}

PRURITUS_CONFIG = {
    "disease_terms": [
        "pruritus",
        "chronic itch",
        "atopic dermatitis",
        "uremic pruritus",
        "cholestatic pruritus",
        "neuropathic itch",
    ],
    "known_targets": [
        "IL31", "IL31RA", "OSMR",  # IL-31 axis
        "JAK1", "JAK2", "TYK2",    # JAK-STAT
        "TRPV1", "TRPA1", "TRPV4", # TRP channels
        "TACR1",                     # NK1R
        "MRGPRX1", "MRGPRX2", "MRGPRX4",  # Mrgpr family
        "HRH1", "HRH4",            # Histamine receptors
        "S1PR1",                     # S1P receptor
        "GRPR", "NPPB",            # GRP/BNP itch circuit
    ],
    "data_dir": DATA_DIR / "pruritus",
}

# Scoring weights
SCORING_WEIGHTS = {
    "genetic_evidence": 0.25,
    "expression_specificity": 0.20,
    "druggability": 0.20,
    "novelty": 0.15,
    "competition": 0.10,
    "literature_trend": 0.10,
}
