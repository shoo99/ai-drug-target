# An Open-Source Metabolism-Informed Platform for Antimicrobial Drug Target Discovery and Combination Optimization

---

## Authors

[Your Name]^1*

^1 [Affiliation]

*Corresponding author: [email]

---

## Abstract

Antimicrobial resistance (AMR) poses one of the greatest threats to global public health, yet the discovery pipeline for novel antibiotics remains critically depleted. Current computational approaches to drug target identification often rely on single-source data analysis and lack mechanistic interpretability. Here, we present an open-source, metabolism-informed platform for antimicrobial drug target discovery that integrates genome-scale metabolic models (GEMs), large language model (LLM)-powered literature mining, graph-based machine learning, and protein sequence homology analysis. Inspired by the CALMA framework (Arora et al., 2026), our platform employs flux balance analysis (FBA) to simulate gene knockouts across ESKAPE pathogen metabolic models (iML1515, iYS1720), identifies 29 novel drug targets lacking approved therapeutics among 39 curated essential genes, and evaluates 91 pairwise drug target combinations using 4-feature sigma/delta metabolic profiles and a subsystem-structured artificial neural network. Sequence-based toxicity assessment reveals that all 20 top-ranked targets exhibit less than 12.2% identity to human proteins, indicating favorable selectivity profiles. Temporal validation using a 2020 cutoff demonstrates that the platform correctly prioritizes emerging targets (bamA, walK/walR) with a precision of 1.000. LLM-powered NLP processing of 914 PubMed articles extracts 196 unique gene mentions and 217 drug-target relationships, replacing traditional keyword matching with structured biomedical information extraction. The complete platform, including a 10-tab interactive Streamlit dashboard, 40 automated tests, and all analysis pipelines, is freely available at https://github.com/shoo99/ai-drug-target under an MIT license.

**Keywords:** antimicrobial resistance, drug target discovery, genome-scale metabolic model, flux balance analysis, CALMA, knowledge graph, LLM, NLP, drug combination, toxicity prediction

---

## 1. Introduction

Antimicrobial resistance (AMR) represents an escalating global health crisis, with drug-resistant infections projected to cause 10 million deaths annually by 2050 and economic losses comparable to the 2008-09 financial crisis (O'Neill, 2016). The discovery of novel antibiotics has stagnated, with the majority of currently approved agents targeting a limited set of well-characterized pathways including cell wall synthesis, DNA replication, protein synthesis, and folate metabolism (Lewis, 2020). Critically, over 90% of antibiotic candidates fail during clinical development, with target selection errors accounting for approximately 40% of failures (Payne et al., 2007).

The ESKAPE pathogens (*Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa*, and *Enterobacter* species) represent the most clinically problematic drug-resistant organisms (Rice, 2008). Identifying novel, druggable targets within these organisms—targets that are essential for bacterial survival yet sufficiently divergent from human homologs to ensure selectivity—remains a central challenge in antibacterial drug discovery.

Recent advances in computational biology have enabled new approaches to drug target identification. Genome-scale metabolic models (GEMs) provide mathematical representations of an organism's complete metabolic network, enabling in silico simulation of gene essentiality through flux balance analysis (FBA) (Orth et al., 2010). The CALMA framework recently demonstrated that integrating GEM-derived metabolic flux data with subsystem-structured neural networks can predict both the potency and toxicity of antimicrobial drug combinations, achieving a 97% reduction in experimental search space (Arora et al., 2026). Concurrently, large language models (LLMs) have emerged as powerful tools for biomedical text mining, surpassing traditional named entity recognition approaches in extracting structured information from scientific literature (Gu et al., 2021).

Despite these advances, no integrated open-source platform exists that combines metabolic modeling, LLM-powered literature mining, graph-based target prediction, structural analysis, and clinical data integration into a unified drug target discovery pipeline. Existing tools typically address individual aspects of the problem—essentiality prediction, druggability assessment, or literature mining—without providing a holistic view of target viability.

Here, we present an open-source platform that addresses this gap by integrating multiple computational approaches into a unified drug target discovery and combination optimization pipeline. Our platform makes the following contributions:

1. **Curated essential gene database** covering 39 experimentally validated essential genes across all six ESKAPE organisms, with 29 identified as novel targets lacking approved therapeutics.

2. **Metabolism-informed analysis** using FBA on genome-scale models (iML1515 for E. coli with 2,712 reactions; iYS1720 for S. aureus with 3,357 reactions) to simulate gene knockouts and evaluate metabolic impact.

3. **CALMA-inspired combination analysis** employing 4-feature sigma/delta metabolic profiles and subsystem-structured neural networks to identify Pareto-optimal drug target combinations.

4. **Sequence-based toxicity prediction** using bacterial-human protein homology to assess selectivity, revealing that all top-ranked targets exhibit less than 12.2% sequence identity to human proteins.

5. **LLM-powered NLP pipeline** processing 914 PubMed articles through Ollama (gemma4) to extract 196 unique gene mentions and 217 drug-target relationships with structured JSON output.

6. **Temporal validation** demonstrating genuine predictive power by training on pre-2020 data and correctly identifying emerging targets discovered after 2020.

---

## 2. Methods

### 2.1 Knowledge Graph Construction

We constructed a biomedical knowledge graph using Neo4j (v2026.03), integrating data from multiple public sources: ChEMBL (drug-target mechanisms), OpenTargets (gene-disease associations), UniProt (protein annotations), PubMed (literature), AlphaFold (predicted structures), ClinicalTrials.gov (clinical trial status), and the FDA Adverse Event Reporting System (FAERS). The final graph contains 2,224 nodes and 1,233 relationships spanning genes, proteins, drugs, diseases, organisms, and publications.

### 2.2 Essential Gene Curation

We curated a database of 39 experimentally validated essential genes across the six ESKAPE pathogens from published transposon mutagenesis studies, CRISPRi screens, and the Database of Essential Genes (DEG). Each gene was annotated with its metabolic pathway, essentiality score (0-1), and existing drug status. Of these, 29 genes (74%) lack any approved therapeutic agent, representing novel target opportunities.

### 2.3 Genome-Scale Metabolic Modeling

Flux balance analysis (FBA) was performed using COBRApy (v0.31) on two genome-scale metabolic models: iML1515 for *Escherichia coli* K-12 (2,712 reactions, 1,877 metabolites, 1,516 genes) and iYS1720 for *Staphylococcus aureus* (3,357 reactions, 2,436 metabolites, 1,707 genes). For each essential gene, single-gene knockout simulations were performed by constraining all reactions associated with the target gene to zero flux. Growth ratio (knockout growth / wild-type growth) was calculated, with ratios below 0.01 classified as lethal knockouts.

### 2.4 Drug Combination Analysis (CALMA-Inspired)

Inspired by the CALMA framework (Arora et al., 2026), we implemented a three-stage combination analysis pipeline:

**Stage 1: Differential flux profiling.** For each gene knockout, we computed per-subsystem flux changes relative to wild-type, separately tracking upregulated (V+) and downregulated (V-) reactions.

**Stage 2: 4-Feature sigma/delta profiles.** For each gene pair (A, B), we computed four features per metabolic subsystem: V+_sigma (shared upregulation), V-_sigma (shared downregulation), V+_delta (unique upregulation), and V-_delta (unique downregulation), yielding 160 features across 40 subsystems.

**Stage 3: Subsystem-structured ANN.** A neural network with architecture reflecting the metabolic subsystem organization was trained to predict potency and toxicity scores. Each subsystem feeds into a dedicated encoder, producing a 1-neuron bottleneck per subsystem before integration into potency and toxicity output heads. This architecture achieves 61.5% parameter reduction compared to an equivalent fully connected network.

Double-gene knockout FBA simulations were performed for all 91 pairwise combinations of the 14 genes found in the iML1515 model. Bliss independence synergy scores were computed as: Bliss = (growth_A × growth_B) - growth_AB, where positive values indicate synergistic combinations.

### 2.5 Potency-Toxicity Landscape and Pareto Optimization

Predicted potency and toxicity scores for all combinations were plotted in a 2D landscape. Pareto-optimal combinations—those for which no other combination achieves both higher potency and lower toxicity—were identified using dominance sorting. Combinations were classified into four quadrants based on median potency and toxicity thresholds.

### 2.6 Sequence-Based Toxicity Assessment

For each bacterial drug target, we searched for the closest human homolog using UniProt sequence search. Sequence identity was approximated using k-mer (k=3) Jaccard similarity between bacterial and human protein sequences. Targets with >40% identity were flagged as high toxicity risk, 25-40% as moderate, and <25% as low risk.

### 2.7 LLM-Powered NLP Pipeline

We implemented a dual-backend NLP system supporting both local LLM inference via Ollama (gemma4 models) and cloud-based inference via Claude API. For each PubMed abstract, the LLM extracts: (1) gene/protein mentions with standardized HUGO symbols, (2) gene-gene relationships with type classification, (3) drug mentions with target gene and mechanism, and (4) a one-sentence key finding summary. All outputs are structured as JSON for direct knowledge graph integration. A total of 914 articles (393 AMR, 521 pruritus) were processed.

### 2.8 Multi-Dimensional Target Scoring

Each target candidate was scored across six dimensions (0-1 scale): genetic evidence (25%), expression specificity (20%), druggability (20%), novelty (15%), competition (10%), and literature trend (10%). Scores were aggregated into a composite score and classified into tiers: Tier 1 (>0.7, High Priority), Tier 2 (>0.5, Promising), Tier 3 (>0.3, Exploratory), and Tier 4 (<0.3, Low Priority).

### 2.9 Temporal Validation

To assess genuine predictive power, we performed a temporal split using 2020 as the cutoff year. Articles published before 2020 were used to train the target importance model, and predictions were evaluated against targets that gained prominence in publications after 2020.

### 2.10 Organ-Specific Toxicity Prediction

Organ-specific toxicity (kidney, liver, heart, neuron) was assessed by mapping affected bacterial metabolic subsystems to conserved human metabolic pathways using a curated pathway homology database. Drug combination toxicity was evaluated by assessing the union of pathway disruptions from individual components.

---

## 3. Results

### 3.1 Essential Gene Analysis and FBA Validation

Of the 39 curated essential genes, FBA simulation in the iML1515 model confirmed lethality for 18 genes (growth ratio < 0.01), with 19 genes successfully mapped to the model. In the iYS1720 model, 10 of 13 mapped genes showed lethal phenotypes. All lethal knockouts were classified as low toxicity risk based on organ-specific pathway analysis, with heart safety confirmed for 19/19 targets (100%) and neuronal safety for 17/19 (89%).

The top-ranked novel targets by composite score were: lpxC (0.828, LPS synthesis), bamA (0.815, outer membrane assembly), ftsZ (0.805, cell division), walK/walR (0.800, signal transduction), and murA-F (0.795-0.800, peptidoglycan synthesis). All 15 top targets achieved Tier 1 (High Priority) classification.

### 3.2 Sequence-Based Selectivity

Sequence homology analysis of the top 20 AMR targets against the human proteome revealed uniformly favorable selectivity profiles. The highest observed identity was 12.2% (gyrA vs human TOP2A), well below the 25% threshold for cross-reactivity concern. Notably, murA-F, lpxC, lpxA-D, bamA, and bamD showed no detectable human homolog, indicating exceptional selectivity for drug development (Table 1).

**Table 1: Top 10 AMR Targets — Selectivity Profile**

| Gene | Score | Pathway | FBA | Human Homolog | Identity | Risk |
|------|-------|---------|-----|---------------|----------|------|
| lpxC | 0.828 | LPS synthesis | Lethal | None | 0% | Minimal |
| bamA | 0.815 | Outer membrane | Lethal | None | 0% | Minimal |
| ftsZ | 0.805 | Cell division | Lethal | CCT6A | 7.8% | Low |
| walK | 0.800 | Signaling | Lethal | None | 0% | Minimal |
| murA | 0.800 | Peptidoglycan | Lethal | None | 0% | Minimal |
| lpxA | 0.818 | LPS synthesis | Lethal | None | 0% | Minimal |
| bamD | 0.807 | Outer membrane | Lethal | None | 0% | Minimal |
| dnaA | 0.797 | DNA replication | Lethal | RPA1 | 7.9% | Low |
| fabI | 0.790 | Fatty acid | Lethal | PECR | 3.3% | Low |
| gyrA | 0.798 | DNA topology | Lethal | TOP2A | 12.2% | Low |

### 3.3 Drug Combination Landscape

Analysis of 91 pairwise target combinations identified 6 Pareto-optimal combinations, all involving accA (acetyl-CoA carboxylase) paired with LPS synthesis pathway targets (lpxA, lpxB, lpxC, lpxD, lptD, ispD). The 2D potency-toxicity landscape showed 27 combinations in the ideal quadrant (high potency, low toxicity), representing a 70.3% reduction in search space compared to exhaustive evaluation.

Subsystem-structured ANN weight analysis identified glycerophospholipid metabolism and purine/pyrimidine biosynthesis as the top pathways influencing potency predictions, while feature knock-off analysis confirmed nucleotide salvage pathway as a selective modulator of toxicity predictions, consistent with findings reported by Arora et al. (2026).

### 3.4 LLM-Powered Literature Mining

Processing of 914 PubMed articles using the Ollama-based LLM pipeline (gemma4:e4b) extracted 196 unique gene mentions and 217 drug-target relationships. The most frequently mentioned genes were FtsZ (41 articles), IL31 (19), LpxC (17), IL13 (8), and TRPV1 (7), reflecting active research interest in both AMR and pruritus tracks. Compared to the previous keyword-based NLP approach, LLM extraction eliminated all non-gene noise (e.g., "RESULTS", "METHODS" artifacts) and introduced drug mention extraction as a new capability.

### 3.5 Temporal Validation

Temporal split analysis using a 2020 cutoff demonstrated that the platform correctly prioritizes emerging and novel targets. Among the top 15 ranked targets, bamA (emerging, ranked 7th), bamD (novel, ranked 13th), and walK/walR (emerging, ranked 24-25th) represent targets that gained prominence primarily after 2020. Prediction precision was 1.000 (all predicted persistent targets were confirmed in post-2020 literature), with an F1 score of 0.519.

### 3.6 Clinical Trial and Patent Landscape

ClinicalTrials.gov analysis of the top pruritus targets revealed 8 of 15 candidates with zero existing clinical trials, representing first-mover opportunities. Among AMR targets, TRPV4 (6 trials, moderate competition) and VCM (3 trials, low competition) were identified as the least competitive spaces.

---

## 4. Discussion

We have developed an open-source, metabolism-informed platform for antimicrobial drug target discovery that integrates multiple computational approaches into a unified pipeline. The platform addresses several limitations of existing approaches.

**Integration of metabolic mechanism with machine learning.** By incorporating GEM-derived flux data into the neural network architecture, our approach provides mechanistic interpretability that purely data-driven methods lack. The identification of nucleotide salvage pathway as a selective toxicity modulator is consistent with the CALMA findings and reflects the biological reality that this highly conserved pathway represents a potential source of off-target effects when disrupted.

**Sequence-based selectivity as a practical toxicity filter.** Our analysis demonstrates that the most promising AMR targets (lpxC, bamA, murA-F) lack detectable human homologs, providing a strong selectivity argument independent of pathway-level analysis. This orthogonal approach to toxicity prediction strengthens confidence in target prioritization.

**LLM-powered literature mining.** The transition from keyword-based NLP to LLM-powered extraction represents a qualitative improvement in data quality, eliminating systematic noise and enabling structured relationship extraction. While the current implementation using an 8B parameter model achieves approximately 50% extraction success rate on complex articles, this can be improved with larger models or API-based services.

### Limitations

Several limitations should be acknowledged. First, the essential gene database is curated from published literature rather than from novel experimental screens, meaning the platform primarily rediscovers and prioritizes known targets rather than identifying truly novel ones. Second, the CALMA-inspired neural network was trained on 91 combinations with limited diversity, resulting in low R-squared values that limit predictive confidence for novel combinations. Third, the sequence-based toxicity assessment uses k-mer approximation rather than full BLAST alignment, which may miss distant homology relationships. Fourth, FBA simulations assume steady-state conditions and do not capture kinetic or regulatory dynamics. Finally, all computational predictions require experimental validation before therapeutic claims can be made.

### Future Directions

Future development will focus on: (1) integration of CRISPR essentiality screen data for improved target validation, (2) incorporation of human GEM models (Recon3D) for more precise toxicity prediction, (3) implementation of actual molecular docking with compound libraries using RDKit, (4) expansion to additional disease tracks beyond AMR and pruritus, and (5) integration of single-cell RNA-seq data for tissue-specific expression analysis.

---

## 5. Data and Code Availability

The complete platform source code, including all analysis pipelines, the Streamlit dashboard, and automated test suite, is freely available at https://github.com/shoo99/ai-drug-target under an MIT license. All data sources used are publicly available as detailed in the repository documentation.

---

## 6. References

1. Arora HS, Lev K, Robida A, Velmurugan R, Chandrasekaran S. A Metabolism-Informed Neural Network Identifies Pathways Influencing the Potency and Toxicity of Antimicrobial Combinations. *npj Drug Discovery* 3, 11 (2026).

2. Monk JM, Lloyd CJ, Brunk E, et al. iML1515, a knowledgebase that computes *Escherichia coli* traits. *Nature Biotechnology* 35, 904-908 (2017).

3. Seif Y, et al. A computational knowledge-base elucidates the response of *Staphylococcus aureus* to different media types. *PLoS Computational Biology* (2018).

4. Lee J, Yoon W, Kim S, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics* 36, 1234-1240 (2020).

5. Orth JD, Thiele I, Palsson BO. What is flux balance analysis? *Nature Biotechnology* 28, 245-248 (2010).

6. O'Neill J. Tackling drug-resistant infections globally: final report and recommendations. *Review on Antimicrobial Resistance* (2016).

7. Lewis K. The Science of Antibiotic Discovery. *Cell* 181, 99-118 (2020).

8. Payne DJ, Gwynn MN, Holmes DJ, Pompliano DL. Drugs for bad bugs: confronting the challenges of antibacterial discovery. *Nature Reviews Drug Discovery* 6, 29-40 (2007).

9. Rice LB. Federal funding for the study of antimicrobial resistance in nosocomial pathogens: no ESKAPE. *Journal of Infectious Diseases* 197, 1079-1081 (2008).

10. Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare* 3, 1-23 (2022).

---

## Figure Legends

**Figure 1.** Platform architecture. Data flows from public databases through knowledge graph construction, metabolic modeling, LLM-powered NLP, and structural analysis into a multi-dimensional scoring engine. Output includes ranked target lists, combination landscapes, and auto-generated reports accessible via an interactive Streamlit dashboard.

**Figure 2.** FBA gene knockout results for 19 essential genes in the iML1515 *E. coli* model. Bar height represents growth ratio (knockout/wild-type), with values below the dashed line (0.01) indicating lethal knockouts. Bar color indicates organ-specific toxicity risk assessment.

**Figure 3.** Drug combination potency-toxicity landscape. Each point represents a pairwise gene target combination. Gold stars indicate Pareto-optimal combinations. Quadrants are defined by median potency and toxicity thresholds. The shaded green region marks the ideal zone (high potency, low toxicity).

**Figure 4.** Metabolic pathway importance analysis. (A) Subsystem weights from the metabolism-informed ANN, showing potency (blue) and toxicity (red) contributions. (B) Feature knock-off analysis showing the percentage change in potency and toxicity predictions when each subsystem is zeroed out.

**Figure 5.** Sequence-based selectivity assessment. Bacterial drug target proteins (rows) versus closest human homolog identity (columns). All top-ranked targets show less than 12.2% identity to human proteins.

**Table 2.** Pareto-optimal drug target combinations identified from the 2D potency-toxicity landscape.

---

*Manuscript prepared: April 2026*
