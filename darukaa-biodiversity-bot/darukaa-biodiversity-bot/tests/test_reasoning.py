from app.reasoning import ReasoningEngine


def test_reasoning_produces_multi_variable_chain():
    engine = ReasoningEngine()
    metrics = {
        "soil_organic_carbon_pct": 0.3,
        "rainfall": "low",
        "crop_type": "monoculture_wheat",
        "region": "semi-arid",
    }
    reasoning = engine.reason(metrics, region="semi-arid")
    assert len(reasoning["impacted_metrics"]) >= 3, "must connect at least 3 metrics"
    assert reasoning["sources"], "recommendation must be source-backed"


def test_synthesize_returns_complete_schema():
    engine = ReasoningEngine()
    metrics = {
        "soil_organic_carbon_pct": 0.3,
        "rainfall": "low",
        "crop_type": "monoculture_wheat",
        "region": "semi-arid",
    }
    reasoning = engine.reason(metrics, region="semi-arid")
    rec = engine.synthesize(reasoning)
    for key in ["recommendation", "mechanism", "impacted_metrics", "time_horizon", "confidence", "sources"]:
        assert key in rec


def test_not_generic_output():
    engine = ReasoningEngine()
    metrics = {"soil_organic_carbon_pct": 0.3, "rainfall": "low", "crop_type": "monoculture_wheat", "region": "semi-arid"}
    reasoning = engine.reason(metrics, region="semi-arid")
    rec = engine.synthesize(reasoning)
    assert rec["recommendation"].lower() != "use sustainable practices"
