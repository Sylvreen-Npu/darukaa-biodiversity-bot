# Darukaa Biodiversity Intelligence Chatbot

An AI system that reasons like an environmental scientist across soil, rainfall,
land-use, and biodiversity metrics — not a generic RAG chatbot.

## Architecture

```
User I/O (text/JSON) → Conversation Manager (slot-filling + session memory)
   → Reasoning Engine (relationship-graph traversal + recommendation synthesis)
   → Retrieval Layer (TF-IDF vector search over curated knowledge chunks)
```

1. User message or JSON payload is parsed for known metrics (SOC%, rainfall, crop type, region).
2. If fewer than 3 of the 4 required fields are known, the system asks a targeted clarifying question instead of guessing.
3. Once enough context exists, the **reasoning engine** classifies observed conditions (e.g. `low_soil_organic_carbon`, `monoculture`) and traverses a curated **relationship graph** (`knowledge/relationship_graph.json`, 17 cited edges) to find connected causal chains — this is what forces every recommendation to link 2+ variables instead of giving single-variable advice.
4. In parallel, the **retrieval layer** does TF-IDF vector search (scikit-learn) over 12 curated knowledge chunks (`knowledge/sources.json`), each tagged with topic/region/year/source, filtered toward the traversed graph nodes.
5. The traversed sub-graph + retrieved evidence are combined into a structured recommendation. If `ANTHROPIC_API_KEY` is set, an LLM call phrases the final synthesis naturally; otherwise a deterministic template does it, so the pipeline always runs end-to-end with zero external dependencies.

## Database / Schema

- **Knowledge base**: `knowledge/sources.json` — 12 curated chunks (soil, land use, biodiversity, climate, human impact), each `{id, topic, region, year, source, text}`.
- **Relationship graph**: `knowledge/relationship_graph.json` — 17 metric-to-metric edges, each `{from, to, effect, mechanism, source, quantified_effect}`. This is the structured layer that makes recommendations non-obvious and defensible rather than generic RAG paraphrase.
- **Session state**: in-memory dict keyed by `session_id`, holding `{soil_organic_carbon_pct, rainfall, crop_type, region}`. Swap `SESSIONS` in `app/conversation.py` for Redis/SQLite for persistence across restarts.

## Local Setup

```bash
git clone <this repo>
cd darukaa-biodiversity-bot
pip install -r requirements.txt

# optional: enable LLM-phrased synthesis
export ANTHROPIC_API_KEY=sk-...

# run the API
uvicorn app.main:app --reload

# or run the chat UI
streamlit run streamlit_app.py
```

### API endpoints
- `POST /chat` — `{session_id, message}` free-text turn, with slot-filling.
- `POST /analyze` — `{session_id, metrics: {...}}` structured JSON input (see example below).

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` + `flake8` on every push to `main`.

## Live Demo

_[Add your deployed URL here, e.g. Render/Railway/Fly.io, if deployed]_

## Example Query

Input (matches the hackathon's example):
```json
{
  "soil_organic_carbon_pct": 0.3,
  "rainfall": "low",
  "crop_type": "monoculture_wheat",
  "region": "semi-arid"
}
```

Output:
```json
{
  "recommendation": "Introduce Agroforestry",
  "mechanism": "monoculture narrows land use type → land use type increases habitat fragmentation → habitat fragmentation decreases species richness → agroforestry decreases habitat fragmentation",
  "impacted_metrics": ["agroforestry", "habitat_fragmentation", "land_use_type", "low_soil_organic_carbon", "monoculture", "species_richness", "..."],
  "expected_improvement": "Agroforestry buffers reduce local fragmentation index by 15-30% over 5 years",
  "time_horizon": "medium (2-3 years)",
  "confidence": "high",
  "sources": ["ESA WorldCover documentation, 2021", "FAO Soil Organic Carbon report, 2021", "IPBES Global Assessment Summary, 2019", "IPCC Land-Use Change report, 2019", "IUCN Habitat-Species Correlation Papers, 2020"]
}
```

## What's intentionally minimal

Per the challenge brief ("not looking for UI-heavy applications"), the UI is a
single-file Streamlit chat — the effort is in the relationship graph, the
retrieval pipeline, and the multi-metric reasoning, per the scoring weights
(reasoning 30%, grounding 25%, knowledge system 20%, conversation 15%, output
clarity 10%).
