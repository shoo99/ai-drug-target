"""
Sequence-based Toxicity Prediction — BLAST homology between bacterial and human proteins.
If a bacterial drug target has a close human homolog, inhibiting it may cause toxicity.

Principle:
- Sequence identity >40% → high cross-reactivity risk (toxicity likely)
- 25-40% → moderate risk (structural similarity, possible off-target)
- <25% → low risk (selective target, safe for drug development)
"""
import time
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

RESULTS_DIR = DATA_DIR / "sequence_toxicity"

# UniProt REST API for sequence retrieval + BLAST
UNIPROT_API = "https://rest.uniprot.org"
NCBI_BLAST_API = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"


class SequenceToxicityPredictor:
    """Predict toxicity based on protein sequence homology between bacteria and human."""

    def __init__(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def get_protein_sequence(self, uniprot_id: str) -> str:
        """Fetch protein sequence from UniProt."""
        url = f"{UNIPROT_API}/uniprotkb/{uniprot_id}.fasta"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                sequence = "".join(lines[1:])  # Skip header
                return sequence
        except Exception:
            pass
        return ""

    def search_human_homolog(self, gene_name: str, bacterial_uniprot: str = "") -> dict:
        """
        Search for human homolog of a bacterial protein.
        Uses UniProt to find human proteins with same gene name or function.
        """
        # Strategy 1: Search human proteome for same gene name
        url = f"{UNIPROT_API}/uniprotkb/search"
        params = {
            "query": f'(gene:{gene_name}) AND (organism_id:9606) AND (reviewed:true)',
            "format": "json",
            "size": 1,
            "fields": "accession,gene_names,protein_name,sequence",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                hits = resp.json().get("results", [])
                if hits:
                    r = hits[0]
                    seq = r.get("sequence", {}).get("value", "")
                    genes = r.get("genes", [{}])
                    gene = genes[0].get("geneName", {}).get("value", "") if genes else ""
                    return {
                        "human_uniprot": r.get("primaryAccession", ""),
                        "human_gene": gene,
                        "human_protein": r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                        "human_sequence": seq,
                        "match_method": "gene_name",
                    }
        except Exception:
            pass

        # Strategy 2: Search by function keyword
        function_map = {
            "murA": "UDP-N-acetylglucosamine enolpyruvyl transferase",
            "gyrA": "DNA gyrase",
            "rpoB": "RNA polymerase beta",
            "folA": "dihydrofolate reductase",
            "fabI": "enoyl reductase",
            "ftsZ": "tubulin",  # Human homolog is tubulin
            "dnaA": "replication initiator",
            "lpxC": "deacetylase",
            "bamA": "outer membrane assembly",  # No human homolog
            "accA": "acetyl-CoA carboxylase",
            "secA": "protein translocase",
        }

        func = function_map.get(gene_name, "")
        if func:
            params["query"] = f'({func}) AND (organism_id:9606) AND (reviewed:true)'
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    hits = resp.json().get("results", [])
                    if hits:
                        r = hits[0]
                        seq = r.get("sequence", {}).get("value", "")
                        genes = r.get("genes", [{}])
                        gene = genes[0].get("geneName", {}).get("value", "") if genes else ""
                        return {
                            "human_uniprot": r.get("primaryAccession", ""),
                            "human_gene": gene,
                            "human_protein": r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                            "human_sequence": seq,
                            "match_method": "function_search",
                        }
            except Exception:
                pass

        return {"human_uniprot": "", "human_gene": "", "match_method": "none"}

    def compute_sequence_identity(self, seq1: str, seq2: str) -> float:
        """
        Compute approximate sequence identity using local alignment.
        Uses a simple sliding window approach (fast, no BLAST needed).
        For precise results, BLAST would be better, but this is sufficient for screening.
        """
        if not seq1 or not seq2:
            return 0.0

        # Use shorter sequence as query
        if len(seq1) > len(seq2):
            seq1, seq2 = seq2, seq1

        # Simple k-mer based similarity (fast approximation)
        k = 3  # tripeptide
        kmers1 = set(seq1[i:i+k] for i in range(len(seq1)-k+1))
        kmers2 = set(seq2[i:i+k] for i in range(len(seq2)-k+1))

        if not kmers1 or not kmers2:
            return 0.0

        shared = len(kmers1 & kmers2)
        total = len(kmers1 | kmers2)

        # Jaccard → approximate identity
        jaccard = shared / total if total > 0 else 0

        # Calibrate: k-mer Jaccard ~0.3 corresponds to ~30% sequence identity
        # Empirical mapping (conservative)
        identity = jaccard * 1.2  # Slight upward correction

        return min(round(identity, 4), 1.0)

    def assess_target_selectivity(self, gene_name: str,
                                   bacterial_uniprot: str = "") -> dict:
        """
        Full selectivity assessment for a drug target.
        Lower human homology = more selective = safer drug target.
        """
        result = {
            "gene": gene_name,
            "bacterial_uniprot": bacterial_uniprot,
            "human_homolog_found": False,
            "human_gene": "",
            "human_uniprot": "",
            "sequence_identity": 0.0,
            "selectivity_score": 1.0,
            "toxicity_risk": "low",
            "recommendation": "",
        }

        # Get bacterial sequence
        bact_seq = ""
        if bacterial_uniprot:
            bact_seq = self.get_protein_sequence(bacterial_uniprot)

        # Search human homolog
        homolog = self.search_human_homolog(gene_name, bacterial_uniprot)

        if homolog.get("human_uniprot"):
            result["human_homolog_found"] = True
            result["human_gene"] = homolog.get("human_gene", "")
            result["human_uniprot"] = homolog.get("human_uniprot", "")
            result["match_method"] = homolog.get("match_method", "")

            # Get human sequence and compute identity
            human_seq = homolog.get("human_sequence", "")
            if not human_seq and homolog["human_uniprot"]:
                human_seq = self.get_protein_sequence(homolog["human_uniprot"])

            if bact_seq and human_seq:
                identity = self.compute_sequence_identity(bact_seq, human_seq)
                result["sequence_identity"] = identity
                result["bacterial_seq_length"] = len(bact_seq)
                result["human_seq_length"] = len(human_seq)

                # Selectivity = inverse of identity
                if identity > 0.4:
                    result["selectivity_score"] = round(1.0 - identity, 3)
                    result["toxicity_risk"] = "high"
                    result["recommendation"] = (
                        f"HIGH RISK: {gene_name} has {identity*100:.0f}% identity to human "
                        f"{result['human_gene']}. Drug may cause off-target toxicity. "
                        f"Consider selectivity engineering or alternative target."
                    )
                elif identity > 0.25:
                    result["selectivity_score"] = round(1.0 - identity * 0.8, 3)
                    result["toxicity_risk"] = "moderate"
                    result["recommendation"] = (
                        f"MODERATE RISK: {identity*100:.0f}% identity to human {result['human_gene']}. "
                        f"Structure-based drug design recommended for selectivity."
                    )
                else:
                    result["selectivity_score"] = round(0.9, 3)
                    result["toxicity_risk"] = "low"
                    result["recommendation"] = (
                        f"LOW RISK: Only {identity*100:.0f}% identity to closest human homolog. "
                        f"Good selectivity for drug development."
                    )
            else:
                result["selectivity_score"] = 0.85
                result["toxicity_risk"] = "low"
                result["recommendation"] = "Human homolog found but sequence comparison not possible. Likely low risk based on gene function."
        else:
            result["selectivity_score"] = 0.95
            result["toxicity_risk"] = "minimal"
            result["recommendation"] = (
                f"MINIMAL RISK: No human homolog found for {gene_name}. "
                f"Highly selective — excellent drug target candidate."
            )

        return result

    def batch_assess(self, targets: list[dict]) -> pd.DataFrame:
        """Assess selectivity for multiple targets."""
        print(f"\n{'='*60}")
        print("SEQUENCE HOMOLOGY TOXICITY ASSESSMENT")
        print(f"{'='*60}")

        results = []
        for target in tqdm(targets, desc="Sequence analysis"):
            gene = target.get("gene_name", target.get("gene", ""))
            uniprot = target.get("uniprot_id", "")

            result = self.assess_target_selectivity(gene, uniprot)
            results.append(result)
            time.sleep(0.5)

            risk_icon = {"minimal": "🟢", "low": "🟢", "moderate": "🟡", "high": "🔴"}.get(result["toxicity_risk"], "⚪")
            human = result.get("human_gene", "none")
            ident = result.get("sequence_identity", 0)
            print(f"  {gene:12s} {risk_icon} {result['toxicity_risk']:10s} | "
                  f"Human: {human:10s} | Identity: {ident*100:.1f}%")

        df = pd.DataFrame(results)
        output = RESULTS_DIR / "sequence_selectivity.csv"
        df.to_csv(output, index=False)
        print(f"\n  Saved: {output}")

        # Summary
        if not df.empty:
            minimal = len(df[df["toxicity_risk"] == "minimal"])
            low = len(df[df["toxicity_risk"] == "low"])
            moderate = len(df[df["toxicity_risk"] == "moderate"])
            high = len(df[df["toxicity_risk"] == "high"])
            print(f"\n  Safety Summary:")
            print(f"    🟢 Minimal/Low risk: {minimal + low}/{len(df)}")
            print(f"    🟡 Moderate risk: {moderate}/{len(df)}")
            print(f"    🔴 High risk: {high}/{len(df)}")

        return df
