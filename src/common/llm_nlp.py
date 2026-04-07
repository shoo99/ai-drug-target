"""
LLM-powered NLP Pipeline — Supports Claude API and Ollama (local/remote).
State-of-the-art NER + relation extraction + structured output.

Usage:
  # Ollama (free, local/remote)
  extractor = LLMNLPExtractor(backend="ollama", ollama_url="http://localhost:11434")

  # Claude API (paid, highest quality)
  extractor = LLMNLPExtractor(backend="claude")
"""
import os
import json
import time
import requests
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from config.settings import DATA_DIR

LLM_RESULTS_DIR = DATA_DIR / "llm_nlp"

EXTRACTION_PROMPT = """You are a biomedical NLP expert. Extract all gene/protein mentions and their relationships from the following biomedical abstract.

Return ONLY valid JSON in this exact format (no other text):
{
  "genes": [
    {"name": "GENE_SYMBOL", "type": "gene|protein|enzyme|receptor|kinase", "role": "target|biomarker|pathway_member|drug_target"}
  ],
  "relations": [
    {"gene1": "GENE1", "relation": "inhibits|activates|binds|regulates|phosphorylates|upregulates|downregulates|targets|interacts_with", "gene2": "GENE2", "confidence": "high|medium|low", "evidence": "brief quote"}
  ],
  "drugs_mentioned": [
    {"name": "DRUG_NAME", "target_gene": "GENE_SYMBOL", "mechanism": "inhibitor|agonist|antagonist|antibody|other"}
  ],
  "key_finding": "one sentence summary"
}

Rules:
- Use standard HUGO gene symbols (uppercase, e.g., LPXC not lpxC)
- Only include genes/proteins explicitly mentioned in the text
- Do NOT hallucinate genes not present in the text
- If no genes found, return empty arrays
- Return ONLY the JSON, no markdown formatting

Abstract:
"""


class LLMNLPExtractor:
    """LLM-powered biomedical NLP — supports Ollama and Claude API."""

    def __init__(self, backend: str = "ollama",
                 ollama_url: str = None,
                 ollama_model: str = None,
                 ollama_api_key: str = None,
                 claude_api_key: str = None,
                 claude_model: str = "claude-sonnet-4-20250514"):
        self.backend = backend
        LLM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        if backend == "ollama":
            self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
            self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "gemma4:26b")
            self.ollama_api_key = ollama_api_key or os.getenv("OLLAMA_API_KEY", "")
            self.ollama_headers = {"X-API-Key": self.ollama_api_key}
            print(f"  LLM NLP initialized: Ollama ({self.ollama_model}) @ {self.ollama_url}")

        elif backend == "claude":
            from anthropic import Anthropic
            api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY", "")
            self.claude_client = Anthropic(api_key=api_key) if api_key else None
            self.claude_model = claude_model
            if self.claude_client:
                print(f"  LLM NLP initialized: Claude ({claude_model})")
            else:
                print("  ⚠️ No ANTHROPIC_API_KEY. Claude NLP disabled.")

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            }
        }
        try:
            resp = requests.post(url, json=payload,
                                headers=self.ollama_headers, timeout=120)
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                return f'{{"error": "Ollama returned {resp.status_code}"}}'
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'

    def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        if not self.claude_client:
            return '{"error": "Claude not configured"}'
        try:
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'

    def _call_llm(self, prompt: str) -> str:
        """Route to appropriate backend."""
        if self.backend == "ollama":
            return self._call_ollama(prompt)
        elif self.backend == "claude":
            return self._call_claude(prompt)
        return '{"error": "Unknown backend"}'

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response, handling various formats."""
        content = content.strip()

        # Remove markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON object
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

        return {"genes": [], "relations": [], "drugs_mentioned": [], "error": "json_parse_failed"}

    def extract_from_text(self, text: str) -> dict:
        """Extract genes, relations, drugs from text using LLM."""
        prompt = EXTRACTION_PROMPT + text[:3000]
        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        # Normalize gene names
        for gene in result.get("genes", []):
            if isinstance(gene, dict) and "name" in gene:
                gene["name"] = gene["name"].upper().replace("-", "").strip()

        for rel in result.get("relations", []):
            if isinstance(rel, dict):
                if "gene1" in rel:
                    rel["gene1"] = rel["gene1"].upper().replace("-", "").strip()
                if "gene2" in rel:
                    rel["gene2"] = rel["gene2"].upper().replace("-", "").strip()

        return result

    def process_articles(self, articles: list[dict],
                          batch_delay: float = 0.5) -> dict:
        """Process multiple PubMed articles with LLM NLP."""
        gene_counts = Counter()
        all_genes = []
        all_relations = []
        all_drugs = []
        gene_articles = defaultdict(list)
        findings = []

        for article in tqdm(articles, desc=f"LLM NLP ({self.backend})"):
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            pmid = article.get("pmid", "")

            if not text.strip():
                continue

            result = self.extract_from_text(text)

            if "error" in result and not result.get("genes"):
                continue

            for gene in result.get("genes", []):
                if not isinstance(gene, dict) or "name" not in gene:
                    continue
                name = gene["name"]
                if len(name) < 2 or len(name) > 15:
                    continue
                gene_counts[name] += 1
                gene_articles[name].append(pmid)
                all_genes.append({**gene, "pmid": pmid})

            for rel in result.get("relations", []):
                if isinstance(rel, dict):
                    rel["pmid"] = pmid
                    all_relations.append(rel)

            for drug in result.get("drugs_mentioned", []):
                if isinstance(drug, dict):
                    drug["pmid"] = pmid
                    all_drugs.append(drug)

            if result.get("key_finding"):
                findings.append({"pmid": pmid, "finding": result["key_finding"]})

            time.sleep(batch_delay)

        output = {
            "gene_counts": gene_counts,
            "genes": all_genes,
            "relations": all_relations,
            "drugs": all_drugs,
            "findings": findings,
            "gene_articles": dict(gene_articles),
            "n_articles": len(articles),
            "n_unique_genes": len(gene_counts),
            "n_relations": len(all_relations),
            "n_drugs": len(all_drugs),
            "backend": self.backend,
            "model": self.ollama_model if self.backend == "ollama" else "claude",
        }

        self._save_results(output)
        return output

    def _save_results(self, output: dict):
        """Save extraction results."""
        import pandas as pd

        pd.DataFrame(
            output["gene_counts"].most_common(100),
            columns=["gene", "count"]
        ).to_csv(LLM_RESULTS_DIR / "llm_gene_counts.csv", index=False)

        if output["relations"]:
            pd.DataFrame(output["relations"]).to_csv(
                LLM_RESULTS_DIR / "llm_relations.csv", index=False)

        if output["drugs"]:
            pd.DataFrame(output["drugs"]).to_csv(
                LLM_RESULTS_DIR / "llm_drugs.csv", index=False)

        if output["findings"]:
            pd.DataFrame(output["findings"]).to_csv(
                LLM_RESULTS_DIR / "llm_findings.csv", index=False)

        serializable = {k: v for k, v in output.items() if k != "gene_counts"}
        serializable["gene_counts_top50"] = output["gene_counts"].most_common(50)
        with open(LLM_RESULTS_DIR / "llm_extraction_full.json", "w") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n  Results saved to: {LLM_RESULTS_DIR}")
