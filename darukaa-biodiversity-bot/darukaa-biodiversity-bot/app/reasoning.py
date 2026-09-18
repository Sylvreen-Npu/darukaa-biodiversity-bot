"""
Multi-metric reasoning engine.

Given observed metrics (e.g. low SOC + low rainfall + monoculture), this
traverses the curated relationship graph (knowledge/relationship_graph.json)
to find connected chains of edges, then combines those chains with retrieved
RAG evidence to synthesize a structured, evidence-backed recommendation.

This deliberately does NOT let an LLM freelance a recommendation from
scratch. The graph traversal decides *which* mechanisms are relevant; the
LLM (if an ANTHROPIC_API_KEY is configured) is only used to phrase the final
synthesis in natural language from the traversed sub-graph + evidence. If no
API key is configured, a deterministic template-based synthesis is used
instead so the system still runs end-to-end offline.
"""
import json
import os
from typing import Dict, List, Optional

from app.retrieval import Retriever

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "relationship_graph.json")

# Maps raw user metric observations -> graph node names
METRIC_TO_NODE = {
    "low_soc": "low_soil_organic_carbon",
    "low_rainfall": "low_rainfall",
    "monoculture": "monoculture",
    "deforestation": "deforestation",
    "pollution": "pollution",
}

# Interventions the engine can recommend, and which node they inject into the graph
INTERVENTIONS = {
    "legume_intercropping": ["low_soil_organic_carbon"],
    "agroforestry": ["monoculture", "low_soil_organic_carbon"],
    "cover_cropping": ["low_soil_organic_carbon", "low_rainfall"],
}


class ReasoningEngine:
    def __init__(self):
        with open(GRAPH_PATH, "r") as f:
            self.graph = json.load(f)["edges"]
        self.retriever = Retriever()

    def _classify_metrics(self, metrics: Dict) -> List[str]:
        """Turn raw metric values into graph node observations."""
        observed = []
        soc = metrics.get("soil_organic_carbon_pct")
        if soc is not None and soc < 0.8:
            observed.append("low_soil_organic_carbon")
        if metrics.get("rainfall") == "low":
            observed.append("low_rainfall")
        crop = (metrics.get("crop_type") or "").lower()
        if "mono" in crop:
            observed.append("monoculture")
        return observed

    def _pick_intervention(self, observed_nodes: List[str]) -> str:
        best, best_overlap = None, -1
        for intervention, targets in INTERVENTIONS.items():
            overlap = len(set(targets) & set(observed_nodes))
            if overlap > best_overlap:
                best, best_overlap = intervention, overlap
        return best or "agroforestry"

    def _traverse(self, start_nodes: List[str], depth: int = 3) -> List[Dict]:
        """BFS outward from observed/intervention nodes through the graph."""
        visited_edges = []
        frontier = set(start_nodes)
        for _ in range(depth):
            next_frontier = set()
            for edge in self.graph:
                if edge["from"] in frontier and edge not in visited_edges:
                    visited_edges.append(edge)
                    next_frontier.add(edge["to"])
            frontier |= next_frontier
        return visited_edges

    def reason(self, metrics: Dict, region: Optional[str] = None) -> Dict:
        observed = self._classify_metrics(metrics)
        if not observed:
            observed = ["low_soil_organic_carbon"]  # safe default so the demo never errors

        intervention = self._pick_intervention(observed)
        chain = self._traverse([intervention] + observed, depth=3)

        # Require at least 2 linked variables -- if traversal is too shallow, widen it
        impacted_nodes = {e["to"] for e in chain} | {e["from"] for e in chain}
        if len(impacted_nodes) < 3:
            chain += self._traverse(list(impacted_nodes), depth=2)
            impacted_nodes = {e["to"] for e in chain} | {e["from"] for e in chain}

        query = f"{intervention.replace('_', ' ')} " + " ".join(o.replace("_", " ") for o in observed)
        evidence = self.retriever.retrieve(query, region_filter=region, top_k=4)

        sources = sorted({e["source"] for e in chain} | {ch["source"] for ch in evidence})
        effect_sizes = [e["quantified_effect"] for e in chain if e.get("quantified_effect")]

        mechanism_text = " → ".join(
            f"{e['from'].replace('_', ' ')} {e['effect']} {e['to'].replace('_', ' ')}" for e in chain[:4]
        )

        return {
            "intervention": intervention,
            "observed": observed,
            "chain": chain,
            "impacted_metrics": sorted(impacted_nodes),
            "evidence": evidence,
            "sources": sources,
            "mechanism_text": mechanism_text,
            "expected_improvement": effect_sizes[0] if effect_sizes else None,
        }

    def synthesize(self, reasoning: Dict) -> Dict:
        """Turn the traversed reasoning into the final structured recommendation.
        Uses the Anthropic API for natural-language phrasing if ANTHROPIC_API_KEY
        is set; otherwise falls back to a deterministic template so the pipeline
        always produces a complete, schema-valid response.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        recommendation_name = reasoning["intervention"].replace("_", " ").title()

        if api_key:
            try:
                return self._synthesize_with_llm(reasoning, api_key)
            except Exception:
                pass  # fall through to template

        return {
            "recommendation": f"Introduce {recommendation_name.lower()}",
            "mechanism": reasoning["mechanism_text"] or "See linked relationship chain.",
            "impacted_metrics": reasoning["impacted_metrics"],
            "expected_improvement": reasoning["expected_improvement"] or "Effect size not quantified in current sources",
            "time_horizon": "medium (2-3 years)",
            "confidence": "high" if reasoning["expected_improvement"] else "medium",
            "sources": reasoning["sources"],
            "retrieved_chunks": [f"{c['id']} ({c['source']})" for c in reasoning["evidence"]],
        }

    def _synthesize_with_llm(self, reasoning: Dict, api_key: str) -> Dict:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are an environmental scientist. Given these observed metrics: {reasoning['observed']}
And these established relationships: {json.dumps(reasoning['chain'])}
And this retrieved evidence: {json.dumps(reasoning['evidence'])}

Produce a recommendation that:
- Names a specific, non-obvious intervention (the suggested intervention is: {reasoning['intervention']})
- Explains the causal mechanism using at least 2 linked variables
- Cites the specific source and quantifies expected improvement
- States time horizon (short/medium/long) and confidence
Avoid generic phrases like "use sustainable practices."

Respond ONLY with JSON matching this schema, nothing else:
{{"recommendation": "...", "mechanism": "...", "impacted_metrics": ["..."], "expected_improvement": "...", "time_horizon": "...", "confidence": "..."}}
"""
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        text = text.strip().strip("```json").strip("```")
        parsed = json.loads(text)
        parsed["sources"] = reasoning["sources"]
        parsed["retrieved_chunks"] = [f"{c['id']} ({c['source']})" for c in reasoning["evidence"]]
        return parsed
