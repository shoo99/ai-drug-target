"""
AlphaFold + UniProt Structure Client
Fetch protein structure availability and druggability data for target candidates
"""
import time
import requests
import pandas as pd
from tqdm import tqdm


class AlphaFoldClient:
    ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
    UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

    def check_structure(self, uniprot_id: str) -> dict:
        """Check if AlphaFold structure exists for a UniProt ID."""
        url = f"{self.ALPHAFOLD_API}/prediction/{uniprot_id}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    entry = data[0] if isinstance(data, list) else data
                    return {
                        "uniprot_id": uniprot_id,
                        "has_alphafold": True,
                        "pdb_url": entry.get("pdbUrl", ""),
                        "model_url": entry.get("cifUrl", ""),
                        "pae_url": entry.get("paeImageUrl", ""),
                        "avg_plddt": entry.get("averagePlddt"),
                    }
            return {"uniprot_id": uniprot_id, "has_alphafold": False}
        except Exception:
            return {"uniprot_id": uniprot_id, "has_alphafold": False}

    def get_gene_uniprot_id(self, gene_symbol: str, organism: str = "human") -> str:
        """Look up UniProt ID from gene symbol."""
        org_id = "9606" if organism == "human" else ""
        query = f"(gene:{gene_symbol})"
        if org_id:
            query += f" AND (organism_id:{org_id})"

        url = f"{self.UNIPROT_API}/search"
        params = {
            "query": query,
            "format": "json",
            "size": 1,
            "fields": "accession,gene_names,protein_name",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0].get("primaryAccession", "")
        except Exception:
            pass
        return ""

    def assess_druggability(self, gene_symbol: str) -> dict:
        """Comprehensive druggability assessment for a gene target."""
        result = {
            "gene": gene_symbol,
            "uniprot_id": "",
            "has_alphafold": False,
            "avg_plddt": None,
            "protein_class": "",
            "has_pdb_experimental": False,
            "druggability_score": 0.0,
            "modality_suggestion": "",
        }

        # Get UniProt ID
        uniprot_id = self.get_gene_uniprot_id(gene_symbol)
        if not uniprot_id:
            return result
        result["uniprot_id"] = uniprot_id

        # Check AlphaFold
        af_data = self.check_structure(uniprot_id)
        result["has_alphafold"] = af_data.get("has_alphafold", False)
        result["avg_plddt"] = af_data.get("avg_plddt")

        # Get protein details from UniProt
        try:
            url = f"{self.UNIPROT_API}/{uniprot_id}"
            params = {"format": "json"}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()

                # Check for experimental structures
                xrefs = data.get("uniProtKBCrossReferences", [])
                pdb_refs = [x for x in xrefs if x.get("database") == "PDB"]
                result["has_pdb_experimental"] = len(pdb_refs) > 0

                # Protein family/class
                keywords = [kw.get("name", "") for kw in data.get("keywords", [])]
                for kw in keywords:
                    kw_lower = kw.lower()
                    if "kinase" in kw_lower:
                        result["protein_class"] = "kinase"
                        break
                    elif "receptor" in kw_lower:
                        result["protein_class"] = "receptor"
                        break
                    elif "ion channel" in kw_lower or "channel" in kw_lower:
                        result["protein_class"] = "ion_channel"
                        break
                    elif "protease" in kw_lower or "peptidase" in kw_lower:
                        result["protein_class"] = "protease"
                        break
                    elif "transferase" in kw_lower or "oxidoreductase" in kw_lower:
                        result["protein_class"] = "enzyme"
                        break
                    elif "g-protein coupled" in kw_lower:
                        result["protein_class"] = "gpcr"
                        break
                    elif "nuclear receptor" in kw_lower:
                        result["protein_class"] = "nuclear_receptor"
                        break
                    elif "transporter" in kw_lower:
                        result["protein_class"] = "transporter"
                        break

                # Subcellular location
                comments = data.get("comments", [])
                for c in comments:
                    if c.get("commentType") == "SUBCELLULAR LOCATION":
                        locs = c.get("subcellularLocations", [])
                        loc_names = [l.get("location", {}).get("value", "") for l in locs]
                        result["subcellular"] = ", ".join(loc_names[:3])

        except Exception:
            pass

        # Calculate druggability score
        # AlphaFold quality check: pLDDT < 70 for >30% of residues = likely disordered
        score = 0.0
        if result["has_pdb_experimental"]:
            score += 0.3
        elif result["has_alphafold"]:
            plddt = result.get("avg_plddt")
            if plddt and plddt >= 80:
                score += 0.25  # High confidence structure
            elif plddt and plddt >= 70:
                score += 0.15  # Moderate confidence
            elif plddt and plddt < 70:
                score += 0.05  # Low confidence — likely disordered, poor drug target
                result["structure_warning"] = "Low pLDDT (<70) — possibly disordered"

        class_scores = {
            "kinase": 0.35, "gpcr": 0.4, "ion_channel": 0.35,
            "nuclear_receptor": 0.35, "protease": 0.3,
            "enzyme": 0.25, "receptor": 0.3, "transporter": 0.2,
        }
        score += class_scores.get(result["protein_class"], 0.1)

        # Modality suggestion
        if result["protein_class"] in ("kinase", "enzyme", "protease"):
            result["modality_suggestion"] = "Small molecule inhibitor"
        elif result["protein_class"] in ("gpcr", "ion_channel"):
            result["modality_suggestion"] = "Small molecule modulator"
        elif result["protein_class"] in ("receptor",):
            result["modality_suggestion"] = "Monoclonal antibody or small molecule"
        elif score < 0.3:
            result["modality_suggestion"] = "Consider PROTAC, ASO, or siRNA"
        else:
            result["modality_suggestion"] = "Multiple modalities possible"

        result["druggability_score"] = round(min(score, 1.0), 3)
        return result

    def batch_assess(self, gene_list: list[str], organism: str = "human") -> pd.DataFrame:
        """Assess druggability for a list of genes."""
        results = []
        for gene in tqdm(gene_list, desc="AlphaFold assessment"):
            result = self.assess_druggability(gene)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        return pd.DataFrame(results)
