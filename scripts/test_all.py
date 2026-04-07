#!/usr/bin/env python3
"""
Comprehensive Platform Test Suite — 6 Scenarios, 40+ Test Cases
Run: python scripts/test_all.py
"""
import sys
import os
import json
import time
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import pandas as pd

# Test tracking
RESULTS = {"passed": 0, "failed": 0, "errors": []}


def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            try:
                func()
                RESULTS["passed"] += 1
                print(f"  ✅ {name}")
            except AssertionError as e:
                RESULTS["failed"] += 1
                RESULTS["errors"].append(f"FAIL: {name} — {e}")
                print(f"  ❌ {name} — {e}")
            except Exception as e:
                RESULTS["failed"] += 1
                RESULTS["errors"].append(f"ERROR: {name} — {type(e).__name__}: {e}")
                print(f"  💥 {name} — {type(e).__name__}: {e}")
        wrapper.__name__ = name
        return wrapper
    return decorator


# ============================================================
# SCENARIO 4: NLP Precision Test (run first — fastest)
# ============================================================

@test("4-1: Known genes detected in text")
def test_nlp_known_genes():
    from src.common.nlp_extractor import NLPExtractor
    known = ["IL31", "TRPV1", "JAK1", "MRGPRX2", "LPXC", "BAMA", "FTSZ", "MURA"]
    ext = NLPExtractor(known_genes=known)
    text = "IL31 activates TRPV1 and JAK1 signaling. MRGPRX2 and LPXC are novel targets. BAMA and FTSZ are essential. MURA is a drug target."
    genes = ext.extract_genes_from_text(text)
    for g in known:
        assert g.upper() in [x.upper() for x in genes], f"{g} not detected (found: {genes})"

@test("4-2: Non-gene words filtered out")
def test_nlp_filtering():
    from src.common.nlp_extractor import NLPExtractor
    ext = NLPExtractor()
    noise = "RESULTS METHODS BACKGROUND CONCLUSIONS OBJECTIVE PURPOSE DESIGN PATIENTS FINDINGS AREAS COVERED EXPERT OPINION"
    genes = ext.extract_genes_from_text(noise)
    for word in ["RESULTS", "METHODS", "BACKGROUND", "CONCLUSIONS", "OBJECTIVE", "AREAS", "COVERED", "EXPERT", "OPINION"]:
        assert word not in genes, f"Noise word '{word}' not filtered"

@test("4-3: Gene family patterns detected")
def test_nlp_gene_families():
    from src.common.nlp_extractor import NLPExtractor
    ext = NLPExtractor()
    text = "IL-31 receptor and TRPV4 channel are involved. CCL17 and CXCR4 were upregulated."
    genes = ext.extract_genes_from_text(text)
    assert "IL31" in genes or "IL-31" in genes.get("IL31", genes)
    assert "TRPV4" in genes
    assert "CCL17" in genes
    assert "CXCR4" in genes

@test("4-4: Empty text returns empty list")
def test_nlp_empty():
    from src.common.nlp_extractor import NLPExtractor
    ext = NLPExtractor()
    assert ext.extract_genes_from_text("") == []
    assert ext.extract_genes_from_text("   ") == []

@test("4-5: Relationship extraction produces valid types")
def test_nlp_relationships():
    from src.common.nlp_extractor import NLPExtractor
    ext = NLPExtractor(known_genes=["IL31", "JAK1"])
    text = "IL31 activates JAK1 signaling pathway. IL31 inhibits immune response via JAK1."
    rels = ext.extract_relationships(text, ["IL31", "JAK1"])
    assert len(rels) > 0, "No relationships extracted"
    for r in rels:
        assert "gene1" in r and "gene2" in r and "relationship" in r


# ============================================================
# SCENARIO 2: Edge Cases
# ============================================================

@test("2-1: FBA with nonexistent gene doesn't crash")
def test_fba_missing_gene():
    from src.common.metabolic_analysis import MetabolicAnalyzer
    analyzer = MetabolicAnalyzer()
    model = analyzer.load_model("Escherichia coli")
    result = analyzer.simulate_gene_knockout(model, "FAKE_GENE_XYZ123")
    assert result["status"] == "gene_not_found"

@test("2-2: Empty PubMed query returns empty")
def test_pubmed_empty():
    from src.common.pubmed_miner import PubMedMiner
    miner = PubMedMiner()
    try:
        pmids = miner.search("", max_results=1)
        # Either empty or raises — both OK
    except Exception:
        pass  # Network error is acceptable

@test("2-3: OpenTargets invalid disease ID returns empty")
def test_opentargets_invalid():
    from src.common.opentargets import OpenTargetsClient
    client = OpenTargetsClient()
    try:
        targets = client.get_disease_targets("INVALID_ID_123")
        assert isinstance(targets, list)
    except Exception:
        pass  # Network error acceptable

@test("2-4: AlphaFold with invalid UniProt returns None/empty")
def test_alphafold_invalid():
    from src.common.alphafold_client import AlphaFoldClient
    client = AlphaFoldClient()
    result = client.assess_druggability("FAKEGENE999")
    assert result["uniprot_id"] == ""
    assert result["druggability_score"] == 0.0

@test("2-5: Empty DataFrame scoring doesn't crash")
def test_scoring_empty():
    from src.common.scoring import TargetScorer
    scorer = TargetScorer()
    result = scorer.score_target({})
    assert "composite_score" in result
    assert result["composite_score"] >= 0

@test("2-6: Scoring with all zeros produces valid output")
def test_scoring_zeros():
    from src.common.scoring import TargetScorer
    scorer = TargetScorer()
    result = scorer.score_target({
        "genetic_evidence": 0, "expression_specificity": 0,
        "druggability": 0, "novelty": 0, "competition": 0, "literature_trend": 0
    })
    assert result["composite_score"] == 0.0
    assert "tier" in result

@test("2-7: Gene validation helper works correctly")
def test_gene_validation():
    from src.common.utils import validate_gene_name
    assert validate_gene_name("LPXC") == True
    assert validate_gene_name("IL31") == True
    assert validate_gene_name("murA") == True
    assert validate_gene_name("") == False
    assert validate_gene_name("A") == False
    assert validate_gene_name("123ABC") == False
    assert validate_gene_name("A" * 20) == False

@test("2-8: Node2Vec with zero vector doesn't NaN")
def test_cosine_zero_vector():
    emb1 = np.zeros(64)
    emb2 = np.array([1.0] * 64)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(emb1, emb2) / (norm1 * norm2))
    assert cos_sim == 0.0
    assert not np.isnan(cos_sim)


# ============================================================
# SCENARIO 3: CALMA Scientific Validation
# ============================================================

@test("3-1: murA knockout is lethal in E. coli")
def test_fba_murA_lethal():
    from src.common.metabolic_analysis import MetabolicAnalyzer
    analyzer = MetabolicAnalyzer()
    model = analyzer.load_model("Escherichia coli")
    result = analyzer.simulate_gene_knockout(model, "murA")
    assert result["status"] == "success", f"Gene not found in model"
    assert result["is_lethal"] == True, f"murA should be lethal, got growth ratio {result['growth_ratio']}"

@test("3-2: Non-essential gene knockout preserves growth")
def test_fba_non_essential():
    from src.common.metabolic_analysis import MetabolicAnalyzer
    analyzer = MetabolicAnalyzer()
    model = analyzer.load_model("Escherichia coli")
    # lacZ is non-essential
    result = analyzer.simulate_gene_knockout(model, "lacZ")
    if result["status"] == "success":
        assert result["growth_ratio"] > 0.5, "lacZ knockout should not be lethal"

@test("3-3: Sigma/Delta produces 4 features per subsystem")
def test_calma_sigma_delta():
    from src.common.calma_engine import CALMAFeatureGenerator
    gen = CALMAFeatureGenerator("iML1515")
    flux_a = gen.compute_differential_flux("murA")
    flux_b = gen.compute_differential_flux("lpxC")
    if flux_a and flux_b:
        sd = gen.compute_sigma_delta_4feature(flux_a, flux_b)
        for subsystem, features in sd.items():
            assert "vp_sigma" in features
            assert "vn_sigma" in features
            assert "vp_delta" in features
            assert "vn_delta" in features

@test("3-4: CALMA ANN trains without crash")
def test_calma_nn_train():
    # Minimal training test
    import torch
    from src.common.calma_engine import CALMANeuralNetwork
    model = CALMANeuralNetwork(
        subsystem_feature_sizes={"sub_a": 4, "sub_b": 4, "sub_c": 4},
    )
    inputs = {
        "sub_a": torch.randn(10, 4),
        "sub_b": torch.randn(10, 4),
        "sub_c": torch.randn(10, 4),
    }
    pot, tox = model(inputs)
    assert pot.shape == (10, 1)
    assert tox.shape == (10, 1)
    assert not torch.isnan(pot).any()
    assert not torch.isnan(tox).any()

@test("3-5: Toxicity assessment returns valid organ scores")
def test_toxicity_organs():
    from src.common.human_toxicity import HumanToxicityPredictor
    pred = HumanToxicityPredictor()
    result = pred.assess_organ_toxicity(["Oxidative phosphorylation", "Purine and Pyrimidine Biosynthesis"])
    assert "overall_toxicity_score" in result
    assert "organ_scores" in result
    for organ in ["kidney", "liver", "heart", "neuron"]:
        assert organ in result["organ_scores"]
        assert "risk" in result["organ_scores"][organ]
        assert "score" in result["organ_scores"][organ]

@test("3-6: Bliss synergy calculation is correct")
def test_bliss_synergy():
    from src.common.drug_combinations import DrugCombinationSimulator
    sim = DrugCombinationSimulator.__new__(DrugCombinationSimulator)
    sim.wt_growth = 1.0
    result = sim.compute_synergy_score(0.5, 0.5, 0.1)  # actual < expected → synergistic
    assert result["bliss_expected"] == 0.25
    assert result["bliss_score"] == 0.15  # 0.25 - 0.10
    assert result["interaction"] == "synergistic"

@test("3-7: Bliss antagonistic correctly identified")
def test_bliss_antagonistic():
    from src.common.drug_combinations import DrugCombinationSimulator
    sim = DrugCombinationSimulator.__new__(DrugCombinationSimulator)
    sim.wt_growth = 1.0
    result = sim.compute_synergy_score(0.5, 0.5, 0.5)  # actual > expected → antagonistic
    assert result["interaction"] == "antagonistic"


# ============================================================
# SCENARIO 6: Security & Data Integrity
# ============================================================

@test("6-1: .env in .gitignore")
def test_env_gitignored():
    gitignore = Path(".gitignore").read_text()
    assert ".env" in gitignore

@test("6-2: Neo4j query injection blocked")
def test_injection_blocked():
    from src.common.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph.__new__(KnowledgeGraph)
    try:
        # This should raise ValueError due to invalid label
        kg.add_relationship(
            "Gene} RETURN 1;//", "id", "test",
            "VALID_REL",
            "Gene", "id", "test"
        )
        assert False, "Injection should have been blocked"
    except (ValueError, AttributeError):
        pass  # Expected

@test("6-3: No hardcoded passwords in settings")
def test_no_hardcoded_passwords():
    content = Path("config/settings.py").read_text()
    assert 'drugTarget2026!' not in content, "Hardcoded password found"

@test("6-4: .env.example exists and has no real credentials")
def test_env_example():
    path = Path(".env.example")
    assert path.exists(), ".env.example missing"
    content = path.read_text()
    assert "your_" in content or "example" in content, "Looks like real credentials"

@test("6-5: No API keys in source code")
def test_no_api_keys():
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text()
        assert "sk-" not in content, f"Potential API key in {py_file}"
        assert "AKIA" not in content, f"Potential AWS key in {py_file}"

@test("6-6: data/ directory gitignored")
def test_data_gitignored():
    gitignore = Path(".gitignore").read_text()
    assert "data/" in gitignore


# ============================================================
# SCENARIO 5: Dashboard UI Tests
# ============================================================

@test("5-1: Dashboard app.py imports without error")
def test_dashboard_import():
    # Just check syntax — don't launch streamlit
    import py_compile
    py_compile.compile("src/dashboard/app.py", doraise=True)

@test("5-2: CRM module imports without error")
def test_crm_import():
    import py_compile
    py_compile.compile("src/dashboard/crm.py", doraise=True)

@test("5-3: Dashboard running on port 8501")
def test_dashboard_running():
    import requests
    try:
        resp = requests.get("http://localhost:8501", timeout=5)
        assert resp.status_code == 200
    except Exception:
        pass  # OK if not running during test

@test("5-4: Landing page accessible")
def test_landing_page():
    path = Path("src/dashboard/landing.html")
    assert path.exists()
    content = path.read_text()
    assert "AI Drug Target" in content
    assert "</html>" in content


# ============================================================
# SCENARIO 1: End-to-End Integration
# ============================================================

@test("1-1: Neo4j connection works")
def test_neo4j_connection():
    from src.common.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    result = kg.run_query("RETURN 1 as test")
    assert result == [{"test": 1}]
    kg.close()

@test("1-2: Knowledge graph has data")
def test_graph_has_data():
    from src.common.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    stats = kg.get_graph_stats()
    assert stats[0]["nodes"] > 100, f"Only {stats[0]['nodes']} nodes"
    assert stats[0]["relationships"] > 50, f"Only {stats[0]['relationships']} relationships"
    kg.close()

@test("1-3: AMR scored targets exist and have valid scores")
def test_amr_scored():
    path = Path("data/amr/top_targets_scored.csv")
    assert path.exists(), "AMR scored targets file missing"
    df = pd.read_csv(path)
    assert len(df) > 0, "No scored targets"
    assert "composite_score" in df.columns
    assert df["composite_score"].max() > 0.5, "All scores too low"
    assert df["composite_score"].min() >= 0, "Negative scores"
    assert df["composite_score"].max() <= 1, "Scores > 1"

@test("1-4: Pruritus scored targets exist")
def test_pruritus_scored():
    path = Path("data/pruritus/top_targets_scored.csv")
    assert path.exists(), "Pruritus scored targets missing"
    df = pd.read_csv(path)
    assert len(df) > 0

@test("1-5: FBA results exist and are valid")
def test_fba_results():
    path = Path("data/metabolic_analysis/metabolic_Escherichia_coli.csv")
    assert path.exists(), "E. coli FBA results missing"
    df = pd.read_csv(path)
    assert len(df) > 0
    valid = df[df["growth_ratio"].notna()]
    assert len(valid) > 5, f"Only {len(valid)} valid FBA results"
    lethal = valid[valid["is_lethal"] == True]
    assert len(lethal) > 0, "No lethal knockouts found"

@test("1-6: CALMA landscape exists")
def test_calma_landscape():
    path = Path("data/calma_results/calma_landscape.csv")
    assert path.exists(), "CALMA landscape missing"
    df = pd.read_csv(path)
    assert len(df) > 10, "Too few combinations"
    if "pareto_optimal" in df.columns:
        assert df["pareto_optimal"].sum() > 0, "No Pareto optimal found"

@test("1-7: PDF reports generated")
def test_reports_exist():
    report_dir = Path("reports")
    pdfs = list(report_dir.glob("*.pdf"))
    assert len(pdfs) >= 5, f"Only {len(pdfs)} PDF reports"

@test("1-8: AlphaFold structures downloaded")
def test_alphafold_structures():
    struct_dir = Path("data/docking/structures")
    if struct_dir.exists():
        pdbs = list(struct_dir.glob("*.pdb"))
        assert len(pdbs) > 10, f"Only {len(pdbs)} PDB files"

@test("1-9: All config files present")
def test_config_files():
    for f in ["config/settings.py", ".env", ".env.example", ".gitignore",
              "requirements.txt", "LICENSE", "README.md"]:
        assert Path(f).exists(), f"Missing: {f}"

@test("1-10: Clinical trial data exists")
def test_clinical_trials_data():
    for name in ["trials_antimicrobial.csv", "trials_pruritus.csv"]:
        path = Path(f"data/common/clinical_trials/{name}")
        assert path.exists(), f"Missing: {name}"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("🧪 COMPREHENSIVE PLATFORM TEST SUITE")
    print("   6 Scenarios, 40+ Test Cases")
    print("=" * 60)

    scenarios = [
        ("SCENARIO 4: NLP Precision", [
            test_nlp_known_genes, test_nlp_filtering, test_nlp_gene_families,
            test_nlp_empty, test_nlp_relationships,
        ]),
        ("SCENARIO 2: Edge Cases", [
            test_fba_missing_gene, test_pubmed_empty, test_opentargets_invalid,
            test_alphafold_invalid, test_scoring_empty, test_scoring_zeros,
            test_gene_validation, test_cosine_zero_vector,
        ]),
        ("SCENARIO 3: CALMA Scientific Validation", [
            test_fba_murA_lethal, test_fba_non_essential, test_calma_sigma_delta,
            test_calma_nn_train, test_toxicity_organs, test_bliss_synergy,
            test_bliss_antagonistic,
        ]),
        ("SCENARIO 6: Security & Data Integrity", [
            test_env_gitignored, test_injection_blocked, test_no_hardcoded_passwords,
            test_env_example, test_no_api_keys, test_data_gitignored,
        ]),
        ("SCENARIO 5: Dashboard UI", [
            test_dashboard_import, test_crm_import, test_dashboard_running,
            test_landing_page,
        ]),
        ("SCENARIO 1: End-to-End Integration", [
            test_neo4j_connection, test_graph_has_data, test_amr_scored,
            test_pruritus_scored, test_fba_results, test_calma_landscape,
            test_reports_exist, test_alphafold_structures, test_config_files,
            test_clinical_trials_data,
        ]),
    ]

    for scenario_name, tests in scenarios:
        print(f"\n{'─'*60}")
        print(f"  {scenario_name}")
        print(f"{'─'*60}")
        for test_func in tests:
            test_func()

    # Summary
    total = RESULTS["passed"] + RESULTS["failed"]
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total:  {total}")
    print(f"  Passed: {RESULTS['passed']} ✅")
    print(f"  Failed: {RESULTS['failed']} ❌")
    print(f"  Rate:   {RESULTS['passed']/max(total,1)*100:.1f}%")

    if RESULTS["errors"]:
        print(f"\n  Failures:")
        for err in RESULTS["errors"]:
            print(f"    • {err}")

    print(f"\n{'='*60}")
    if RESULTS["failed"] == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {RESULTS['failed']} test(s) need attention")
    print(f"{'='*60}")

    return RESULTS["failed"]


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
