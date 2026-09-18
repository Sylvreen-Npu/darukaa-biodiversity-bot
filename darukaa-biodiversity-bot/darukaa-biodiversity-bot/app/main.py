from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, AnalyzeRequest, Recommendation
from app.conversation import handle_message, next_clarifying_question, get_session
from app.reasoning import ReasoningEngine

app = FastAPI(title="Darukaa Biodiversity Intelligence Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = ReasoningEngine()


@app.get("/")
def root():
    return {"status": "ok", "service": "darukaa-biodiversity-chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = handle_message(req.session_id, req.message)
    session = result["session"]

    if not result["ready"]:
        question = next_clarifying_question(session)
        return ChatResponse(
            reply=question or "Could you share a bit more about the land — soil, rainfall, and crop type?",
            missing_fields=result["missing"],
        )

    reasoning = engine.reason(session, region=session.get("region"))
    rec = engine.synthesize(reasoning)
    recommendation = Recommendation(**rec)

    reply = (
        f"Recommendation: {recommendation.recommendation}. "
        f"{recommendation.mechanism} "
        f"Expected improvement: {recommendation.expected_improvement}. "
        f"Time horizon: {recommendation.time_horizon}, confidence: {recommendation.confidence}."
    )
    return ChatResponse(reply=reply, missing_fields=[], recommendation=recommendation)


@app.post("/analyze", response_model=ChatResponse)
def analyze(req: AnalyzeRequest):
    """Structured JSON input path (bypasses slot-filling if fields are complete)."""
    metrics = req.metrics.dict()
    from app.conversation import update_session, has_enough_context, missing_fields as mf, next_clarifying_question as ncq

    session = update_session(req.session_id, metrics)
    if not has_enough_context(session):
        return ChatResponse(
            reply=ncq(session) or "Please provide more metrics.",
            missing_fields=mf(session),
        )

    reasoning = engine.reason(session, region=session.get("region"))
    rec = engine.synthesize(reasoning)
    recommendation = Recommendation(**rec)
    reply = f"Recommendation: {recommendation.recommendation}. {recommendation.mechanism}"
    return ChatResponse(reply=reply, missing_fields=[], recommendation=recommendation)
