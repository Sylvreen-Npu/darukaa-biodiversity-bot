from typing import Optional, List
from pydantic import BaseModel, Field


class MetricsInput(BaseModel):
    """Structured input matching the hackathon's example schema."""
    soil_organic_carbon_pct: Optional[float] = None
    rainfall: Optional[str] = None  # "low" | "medium" | "high"
    crop_type: Optional[str] = None
    region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session id for multi-turn memory")
    message: str


class AnalyzeRequest(BaseModel):
    session_id: str
    metrics: MetricsInput


class Recommendation(BaseModel):
    recommendation: str
    mechanism: str
    impacted_metrics: List[str]
    expected_improvement: Optional[str] = None
    time_horizon: str
    confidence: str
    sources: List[str]
    retrieved_chunks: List[str] = Field(default_factory=list, description="Which knowledge chunks were used, for transparency")


class ChatResponse(BaseModel):
    reply: str
    missing_fields: List[str] = Field(default_factory=list)
    recommendation: Optional[Recommendation] = None
