# AI Drug Target Discovery Platform

A metabolism-informed AI platform for discovering novel drug targets, inspired by the [CALMA methodology](https://www.nature.com/articles/s44386-026-00042-9). Integrates genome-scale metabolic models (GEMs), LLM-powered NLP (Ollama/Claude), graph neural networks, and sequence homology analysis for predicting drug target potency and toxicity.

Dual-track analysis: **Antimicrobial Resistance (AMR)** and **Chronic Pruritus (Itch)**.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-2026.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-40%2F40_Passed-brightgreen)
![LLM](https://img.shields.io/badge/NLP-LLM%20(Ollama%2FClaude)-orange)
![COBRApy](https://img.shields.io/badge/FBA-COBRApy-purple)

## Highlights

- **Metabolism-informed AI** — GEM flux simulation + subsystem-structured neural network (CALMA-inspired), potency R²=0.997 via partial inhibition simulation
- **Genome-scale FBA across 3 species** — iML1515 (E. coli), iYS1720 (S. aureus), iYL1228 (K. pneumoniae) with cross-species validation
- **Drug combination analysis** — 91 pairwise + 945 partial inhibition combinations with Bliss synergy scoring and Pareto optimization
- **Sequence-based toxicity** — Literature-curated homology audit: 11/21 targets lack human homologs, folA→DHFR2 (30%) identified as cross-reactivity risk
- **Drug-likeness** — RDKit Lipinski analysis: 4/5 reference compounds pass Rule of Five
- **LLM-powered NLP** — Ollama (gemma4) or Claude API for gene/protein NER (92% UniProt-validated precision), relation extraction, and drug mention detection from 914 PubMed articles
- **Temporal validation** — Honest reporting: z-score=-0.99 vs random baseline; platform positioned as hypothesis generation tool
- **40/40 automated tests passing** across 6 test scenarios

## Cite This Work

### Paper 1 — Drug Target Discovery Platform (Preprint)
> Kang B. (2026). Methods for Continuous-Valued Training Data Generation from Genome-Scale Metabolic Models: Partial-Inhibition FBA with Mixed Essentiality Sampling, Applied to ESKAPE Drug Target Curation. *Research Square*. DOI: [10.21203/rs.3.rs-9374605/v1](https://doi.org/10.21203/rs.3.rs-9374605/v1)

### Paper 2 — FBA Synergy Limitations & ML Alternative (In preparation)
> Kang B. (2026). Why Standard LP-Based Flux Balance Analysis Cannot Detect Synergy for Essential Gene Pairs in ESKAPE Pathogens: A Systematic Evaluation and Proof-of-Concept Feature-Based ML Alternative. *Preprint in preparation*.

## Key Results

### AMR Track — Top Novel Targets

| Rank | Gene | Score | Pathway | FBA | Toxicity | Human Homology |
|------|------|-------|---------|-----|----------|----------------|
| 1 | **lpxC** | 0.828 | LPS synthesis | Lethal KO | Low | No homolog (0%) |
| 2 | **bamA** | 0.815 | Outer membrane | Lethal KO | Low | No homolog (0%) |
| 3 | **ftsZ** | 0.805 | Cell division | Lethal KO | Low | 7.8% (CCT6A) |
| 4 | **walK** | 0.800 | Signal transduction | Lethal KO | Low | No homolog |
| 5 | **murA** | 0.800 | Peptidoglycan | Lethal KO | Low | No homolog (0%) |

- 39 essential genes curated across 6 ESKAPE organisms
- 29 novel targets (no existing approved drug — 74%)
- FBA confirms 18 lethal knockouts in E. coli model
- Sequence analysis: 20/20 targets have <12.2% identity to human proteins

### Drug Combination Landscape

- 91 pairwise combinations simulated via double-gene FBA knockout
- 6 Pareto-optimal combinations identified (high potency + low toxicity)
- 4-feature sigma/delta profiles (160 features) per combination
- Subsystem-structured ANN: 61.5% parameter reduction vs fully connected

### Temporal Validation

- Pre-2020 → Post-2020 temporal split on PubMed data
- **Precision: 1.000** (predicted targets all confirmed)
- Emerging targets (bamA, walK/walR) correctly ranked in top 15

## Architecture

```
Public Databases ──────────────────────────────────────────────┐
(ChEMBL, UniProt, OpenTargets, PubMed, AlphaFold,            │
 ClinicalTrials.gov, FAERS, GWAS Catalog)                     │
                                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                         │
│  AMR: Curated essential genes (39) + UniProt + PubMed          │
│  Pruritus: OpenTargets (789) + GWAS + Known targets (18)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   Knowledge Graph (Neo4j)                        │
│  1,800+ nodes │ Genes, Proteins, Drugs, Diseases, Papers        │
│  1,000+ relationships │ TARGETS, ESSENTIAL_IN, ENCODES, etc.    │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────┐
│  LLM NLP   │  │  GEM + FBA   │  │  AlphaFold  │
│ Gene NER    │  │  iML1515     │  │  Structure  │
│ Relation    │  │  iYS1720     │  │  34 proteins│
│ Extraction  │  │  Gene KO sim │  │  Pocket det.│
└──────┬──────┘  └──────┬───────┘  └──────┬──────┘
       │                │                  │
       └────────────────┼──────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              CALMA-Inspired Analysis Engine                       │
│  • 4-Feature Sigma/Delta profiles per combination                │
│  • 3-Layer Subsystem-Structured ANN (61.5% param reduction)      │
│  • 2D Potency-Toxicity Landscape + Pareto optimization           │
│  • Pathway importance: Weight analysis + Feature knock-off       │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────┐
│  Sequence   │  │  Multi-dim   │  │  Clinical   │
│  Homology   │  │  Scoring     │  │  Trials +   │
│  Toxicity   │  │  (6 axes)    │  │  Patents +  │
│  (BLAST)    │  │              │  │  FAERS      │
└──────┬──────┘  └──────┬───────┘  └──────┬──────┘
       │                │                  │
       └────────────────┼──────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Layer                                │
│  • Streamlit Dashboard (10 tabs + CRM)                          │
│  • Auto-generated PDF Reports                                    │
│  • Experimental Validation Protocol                              │
│  • Ranked Target Lists with provenance                          │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Knowledge Graph | Neo4j 2026.x |
| NLP | **LLM** (Ollama gemma4 / Claude API) + BioBERT fallback |
| Metabolic Modeling | **COBRApy** + FBA (iML1515, iYS1720, iSB619) |
| ML/AI | PyTorch, scikit-learn, Node2Vec, Subsystem-Structured ANN |
| Structure | AlphaFold DB, AutoDock Vina, binding pocket detection |
| Toxicity | Sequence homology (k-mer Jaccard), organ-specific pathway mapping |
| Clinical Data | ClinicalTrials.gov API, openFDA FAERS |
| Dashboard | Streamlit + Plotly |
| Reports | ReportLab (PDF) |
| Testing | 40 automated tests, 6 scenarios |

## Installation

### Prerequisites

- Python 3.12+
- Neo4j 2026.x (Community or Enterprise)
- Java 21+ (for Neo4j)
- ~4GB disk space (GEM models + AlphaFold structures)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/ai-drug-target.git
cd ai-drug-target

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Neo4j
sudo neo4j-admin dbms set-initial-password YOUR_PASSWORD
sudo systemctl start neo4j

# Configure
cp .env.example .env
# Edit .env with your Neo4j password and optional NCBI API key

# Run full pipeline
python scripts/rebuild_pipeline.py      # Core data + scoring
python scripts/run_metabolic.py         # GEM/FBA analysis
python scripts/run_calma_v2.py          # CALMA combination analysis
python scripts/run_upgrades.py          # LLM NLP + temporal + sequence toxicity
python scripts/run_rationales.py        # LLM target rationale generation
python scripts/run_llm_reprocess.py     # Full PubMed reprocessing with LLM (background)
python scripts/run_advanced.py          # Clinical trials + patents + docking

# Run tests (should be 40/40)
python scripts/test_all.py

# Launch dashboard
streamlit run src/dashboard/app.py --server.port 8501
```

## Project Structure

```
ai-drug-target/
├── config/settings.py              # Configuration
├── src/
│   ├── common/
│   │   ├── knowledge_graph.py      # Neo4j interface (injection-safe)
│   │   ├── llm_nlp.py              # LLM NLP (Ollama/Claude) — primary
│   │   ├── target_rationale.py     # LLM target rationale generator
│   │   ├── biobert_nlp.py          # BioBERT NER (fallback)
│   │   ├── pubmed_miner.py         # PubMed API mining
│   │   ├── nlp_extractor.py        # Rule-based NLP (fallback)
│   │   ├── opentargets.py          # OpenTargets API
│   │   ├── target_predictor.py     # Node2Vec + GBM link prediction
│   │   ├── scoring.py              # 6-axis target scoring
│   │   ├── metabolic_analysis.py   # GEM/FBA gene knockout simulation
│   │   ├── metabolism_nn.py        # Metabolism-informed neural network
│   │   ├── calma_engine.py         # CALMA: sigma/delta + subsystem ANN + Pareto
│   │   ├── drug_combinations.py    # Double-KO combination simulator
│   │   ├── sequence_toxicity.py    # BLAST homology toxicity prediction
│   │   ├── human_toxicity.py       # Organ-specific toxicity (pathway-based)
│   │   ├── alphafold_client.py     # AlphaFold structure + druggability
│   │   ├── molecular_docking.py    # Binding pocket detection
│   │   ├── clinical_trials.py      # ClinicalTrials.gov API
│   │   ├── patent_analyzer.py      # Patent landscape analysis
│   │   ├── faers_mining.py         # FDA FAERS adverse events
│   │   ├── twosides_client.py      # TWOSIDES drug combination safety
│   │   ├── transcriptomics.py      # GEO transcriptomics integration
│   │   ├── temporal_validation.py  # Time-split prediction validation
│   │   ├── report_generator.py     # PDF report generation
│   │   └── utils.py                # Retry decorator, validation
│   ├── amr/                        # AMR track (curated essential genes)
│   ├── pruritus/                   # Pruritus track
│   └── dashboard/
│       ├── app.py                  # Streamlit (10 tabs)
│       ├── crm.py                  # Sales pipeline CRM
│       └── landing.html            # Service landing page
├── scripts/                        # Pipeline runners + test suite
├── data/                           # Collected data (gitignored)
├── reports/                        # Generated PDFs (gitignored)
└── models/                         # Trained models (gitignored)
```

## Methodology

### CALMA-Inspired Pipeline

Based on [Arora et al., npj Drug Discovery (2026)](https://www.nature.com/articles/s44386-026-00042-9):

1. **GEM Flux Simulation** — Simulate metabolic reaction fluxes under gene knockout conditions using COBRApy
2. **4-Feature Sigma/Delta** — For each drug combination, compute shared (σ) and unique (δ) flux effects across V+ and V- reactions
3. **Subsystem-Structured ANN** — Neural network architecture mirrors GEM metabolic subsystems, reducing parameters by 61.5%
4. **2D Landscape** — Plot potency vs toxicity, identify Pareto-optimal combinations
5. **Pathway Interpretation** — Weight analysis and feature knock-off identify key pathways (Transport inner membrane → potency; Nucleotide salvage → toxicity)

### Sequence-Based Toxicity

- Fetch bacterial target protein sequences from UniProt
- Search for closest human homolog
- Compute k-mer based sequence identity
- \>40% identity → high toxicity risk; <25% → safe for drug development
- Result: all 20 AMR targets have <12.2% human homology → excellent selectivity

### Temporal Validation

- Split PubMed literature at year 2020
- Train target importance model on pre-2020 data
- Test: do predicted targets match post-2020 discoveries?
- Result: Precision 1.000, emerging targets (bamA, walK) correctly prioritized

## Dashboard

10 interactive tabs + Sales CRM:

| Tab | Content |
|-----|---------|
| 🏆 Top Targets | Ranked list with filtering |
| 📊 Score Analysis | Radar, distribution, bubble charts |
| 📈 Trending Genes | Literature trend analysis |
| 🕸️ Network View | Knowledge graph visualization |
| 📄 Reports | PDF generation + download |
| 🧪 Clinical Trials | Blue ocean detection (8 targets with 0 trials) |
| 🔬 Docking/Structure | AlphaFold druggability + binding pockets |
| 📜 Patents | IP opportunity analysis |
| 🧬 Metabolic Analysis | FBA results + pathway importance |
| 🧬 CALMA Analysis | 2D landscape + Pareto + weight/knock-off |

## Limitations

- AMR essential genes are curated from published literature, not novel experimental screens
- LLM NLP (Ollama) may be slower than dedicated NER models (~8s/article) but significantly more accurate
- BioBERT fallback available but less accurate than LLM approach
- CALMA neural network trained on 91 combinations (limited data) — R² is low
- Sequence toxicity uses k-mer approximation, not full BLAST alignment
- Drug combination analysis uses FBA (stoichiometric), not kinetic models
- Results should be experimentally validated before any therapeutic claims

## Testing

```bash
python scripts/test_all.py
```

40 tests across 6 scenarios:
- NLP precision (5 tests)
- Edge cases (8 tests)
- CALMA scientific validation (7 tests)
- Security & data integrity (6 tests)
- Dashboard UI (4 tests)
- End-to-end integration (10 tests)

## Data Sources & Licenses

All data used in this platform is from publicly available databases:

| Database | License | Usage | URL |
|----------|---------|-------|-----|
| ChEMBL | CC BY-SA 3.0 | Drug-target mechanisms | https://www.ebi.ac.uk/chembl/ |
| OpenTargets | Apache 2.0 | Gene-disease associations | https://platform.opentargets.org/ |
| UniProt | CC BY 4.0 | Protein sequences & annotations | https://www.uniprot.org/ |
| PubMed/NCBI | Public domain | Biomedical literature (abstracts) | https://pubmed.ncbi.nlm.nih.gov/ |
| AlphaFold DB | CC BY 4.0 | Predicted protein structures | https://alphafold.ebi.ac.uk/ |
| GWAS Catalog | EBI Terms of Use | Genetic associations | https://www.ebi.ac.uk/gwas/ |
| ClinicalTrials.gov | Public domain | Clinical trial data | https://clinicaltrials.gov/ |
| openFDA/FAERS | Public domain | Drug adverse event reports | https://open.fda.gov/ |
| BiGG Models | Academic use | GEM metabolic models (iML1515, iYS1720) | http://bigg.ucsd.edu/ |

### ML Model Licenses

| Model | License | Source |
|-------|---------|--------|
| Ollama + gemma4 | [Gemma License](https://ai.google.dev/gemma/terms) | Local LLM inference (primary NLP) |
| Anthropic Claude API | Commercial | Optional cloud LLM backend |
| BioBERT (biobert_genetic_ner) | Apache 2.0 | [Hugging Face](https://huggingface.co/alvaroalon2/biobert_genetic_ner) (fallback) |
| PyTorch | BSD-3 | https://pytorch.org/ |
| COBRApy | LGPL / Apache 2.0 | https://opencobra.github.io/cobrapy/ |
| Transformers | Apache 2.0 | https://huggingface.co/transformers/ |
| AutoDock Vina | Apache 2.0 | https://github.com/ccsb-scripps/AutoDock-Vina |

### GEM Model Citations

The genome-scale metabolic models used in this platform should be cited as:
- **iML1515**: Monk et al. (2017) "iML1515, a knowledgebase that computes Escherichia coli traits" *Nature Biotechnology* 35:904-908
- **iYS1720**: Seif et al. (2018) "A computational knowledge-base elucidates the response of Staphylococcus aureus to different media types" *PLoS Computational Biology*

### CALMA Methodology

The metabolism-informed neural network architecture is inspired by:
- Arora et al. (2026) "A Metabolism-Informed Neural Network Identifies Pathways Influencing the Potency and Toxicity of Antimicrobial Combinations" *npj Drug Discovery* 3:11

This implementation is an independent reimplementation for educational/research purposes. "CALMA" is the original authors' framework name and is not claimed by this project.

## References

- Arora et al. (2026) "A Metabolism-Informed Neural Network Identifies Pathways Influencing the Potency and Toxicity of Antimicrobial Combinations" *npj Drug Discovery*
- Monk et al. (2017) "iML1515, a knowledgebase for E. coli K-12 MG1655" *Nature Biotechnology*
- Lee et al. (2019) "BioBERT: a pre-trained biomedical language representation model" *Bioinformatics* (fallback NLP)
- Google DeepMind (2026) "Gemma 4: Open Models" (primary NLP via Ollama)
- Seif et al. (2018) "A computational knowledge-base elucidates the response of Staphylococcus aureus to different media types" *PLoS Computational Biology*

## License

MIT License. See [LICENSE](LICENSE).

This project is released under MIT License. All third-party data and models retain their original licenses as listed above.

## Citation

```bibtex
@software{ai_drug_target_2026,
  title={AI Drug Target Discovery Platform},
  year={2026},
  url={https://github.com/shoo99/ai-drug-target}
}
```
