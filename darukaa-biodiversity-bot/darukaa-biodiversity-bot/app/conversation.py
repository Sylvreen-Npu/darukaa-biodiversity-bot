"""
Conversation manager: slot-filling + in-memory session state.

Required fields (minimum 3, per hackathon constraints):
  - soil_organic_carbon_pct
  - rainfall
  - crop_type / land use
  - region

If any critical field is missing, the system asks a targeted clarifying
question instead of guessing. Session state persists across turns for the
lifetime of the process (swap SESSIONS for Redis/SQLite for cross-restart
persistence, per the design doc's optional bonus).
"""
import re
from typing import Dict, List, Optional

REQUIRED_FIELDS = ["soil_organic_carbon_pct", "rainfall", "crop_type", "region"]
MIN_REQUIRED = 3

CLARIFYING_QUESTIONS = {
    "soil_organic_carbon_pct": "What's the soil organic carbon percentage (SOC%) for your land, if known?",
    "rainfall": "How would you describe the rainfall pattern — low, medium, or high?",
    "crop_type": "What crop or land use type is on this land (e.g. monoculture wheat, mixed cropping)?",
    "region": "What region or climate zone is this in (e.g. semi-arid, tropical)?",
}

SESSIONS: Dict[str, Dict] = {}


def _extract_metrics_from_text(text: str) -> Dict:
    """Very lightweight slot extraction from free text. In production this
    step would be an LLM call; kept regex-based here to run with zero API
    dependency."""
    extracted = {}
    text_l = text.lower()

    soc_match = re.search(r"(\d+(\.\d+)?)\s*%.{0,20}(organic carbon|soc)|(?:soc|organic carbon).{0,10}(\d+(\.\d+)?)\s*%", text_l)
    if soc_match:
        num = soc_match.group(1) or soc_match.group(4)
        if num:
            extracted["soil_organic_carbon_pct"] = float(num)

    if "low rainfall" in text_l or "low rain" in text_l or "drought" in text_l:
        extracted["rainfall"] = "low"
    elif "high rainfall" in text_l or "heavy rain" in text_l:
        extracted["rainfall"] = "high"
    elif "medium rainfall" in text_l or "moderate rain" in text_l:
        extracted["rainfall"] = "medium"

    if "monoculture" in text_l:
        crop_match = re.search(r"monoculture\s+(\w+)", text_l)
        extracted["crop_type"] = f"monoculture_{crop_match.group(1)}" if crop_match else "monoculture"
    elif "mixed crop" in text_l or "polyculture" in text_l:
        extracted["crop_type"] = "mixed_cropping"

    for region in ["semi-arid", "arid", "tropical", "temperate", "coastal"]:
        if region in text_l:
            extracted["region"] = region
            break

    return extracted


def get_session(session_id: str) -> Dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {f: None for f in REQUIRED_FIELDS}
    return SESSIONS[session_id]


def update_session(session_id: str, updates: Dict) -> Dict:
    session = get_session(session_id)
    for k, v in updates.items():
        if v is not None:
            session[k] = v
    return session


def missing_fields(session: Dict) -> List[str]:
    return [f for f in REQUIRED_FIELDS if session.get(f) is None]


def has_enough_context(session: Dict) -> bool:
    filled = [f for f in REQUIRED_FIELDS if session.get(f) is not None]
    return len(filled) >= MIN_REQUIRED


def next_clarifying_question(session: Dict) -> Optional[str]:
    missing = missing_fields(session)
    if not missing:
        return None
    return CLARIFYING_QUESTIONS[missing[0]]


def handle_message(session_id: str, message: str) -> Dict:
    """Update session from free text, return updated session + whether ready."""
    extracted = _extract_metrics_from_text(message)
    session = update_session(session_id, extracted)
    return {
        "session": session,
        "extracted": extracted,
        "ready": has_enough_context(session),
        "missing": missing_fields(session),
    }
