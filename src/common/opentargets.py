"""
OpenTargets Platform API client — fetch gene-disease associations
"""
import requests
from config.settings import OPENTARGETS_API


class OpenTargetsClient:
    def __init__(self):
        self.api_url = OPENTARGETS_API

    def _query(self, query: str, variables: dict = None) -> dict:
        resp = requests.post(
            self.api_url,
            json={"query": query, "variables": variables or {}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    def get_disease_targets(self, disease_id: str, size: int = 500) -> list[dict]:
        """Get all targets associated with a disease (EFO ID)."""
        query = """
        query diseaseTargets($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            name
            associatedTargets(page: {size: $size, index: 0}) {
              count
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                  biotype
                }
                score
                datatypeScores {
                  id
                  score
                }
              }
            }
          }
        }
        """
        data = self._query(query, {"diseaseId": disease_id, "size": size})
        disease = data.get("disease", {})
        if not disease:
            return []
        rows = disease.get("associatedTargets", {}).get("rows", [])
        results = []
        for row in rows:
            target = row.get("target")
            if not target or not target.get("id"):
                continue
            datatype_scores = {d["id"]: d["score"] for d in row.get("datatypeScores", []) if d.get("id")}
            results.append({
                "ensembl_id": target.get("id", ""),
                "symbol": target.get("approvedSymbol", ""),
                "name": target.get("approvedName", ""),
                "biotype": target.get("biotype", ""),
                "overall_score": row.get("score", 0),
                "genetic_association": datatype_scores.get("genetic_association", 0),
                "known_drug": datatype_scores.get("known_drug", 0),
                "literature": datatype_scores.get("literature", 0),
                "rna_expression": datatype_scores.get("expression", 0),
                "animal_model": datatype_scores.get("animal_model", 0),
            })
        print(f"[OpenTargets] Found {len(results)} targets for {disease.get('name', disease_id)}")
        return results

    def get_target_info(self, ensembl_id: str) -> dict:
        """Get detailed info about a specific target."""
        query = """
        query targetInfo($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            biotype
            functionDescriptions
            subcellularLocations {
              location
            }
            tractability {
              label
              modality
              value
            }
            pathways {
              pathway
              pathwayId
            }
          }
        }
        """
        data = self._query(query, {"ensemblId": ensembl_id})
        return data.get("target", {})

    def search_diseases(self, search_term: str, size: int = 20) -> list[dict]:
        """Search for disease EFO IDs by name."""
        query = """
        query searchDisease($term: String!, $size: Int!) {
          search(queryString: $term, entityNames: ["disease"], page: {size: $size, index: 0}) {
            hits {
              id
              name
              entity
              description
            }
          }
        }
        """
        data = self._query(query, {"term": search_term, "size": size})
        return data.get("search", {}).get("hits", [])

    def get_known_drugs(self, ensembl_id: str) -> list[dict]:
        """Get known drugs for a target."""
        query = """
        query knownDrugs($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            knownDrugs(size: 100) {
              rows {
                drug {
                  id
                  name
                  drugType
                  maximumClinicalTrialPhase
                }
                disease {
                  id
                  name
                }
                phase
                status
              }
            }
          }
        }
        """
        data = self._query(query, {"ensemblId": ensembl_id})
        target = data.get("target", {})
        return target.get("knownDrugs", {}).get("rows", [])
