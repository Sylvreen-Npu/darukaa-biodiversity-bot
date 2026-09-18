import uuid
import streamlit as st

from app.conversation import handle_message, next_clarifying_question
from app.reasoning import ReasoningEngine

st.set_page_config(page_title="Darukaa Biodiversity Chatbot", page_icon="🌱")
st.title("🌱 Darukaa Biodiversity Intelligence Chatbot")
st.caption("Ask about your land's biodiversity — I'll ask for soil, rainfall, and land-use details, then reason across them.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "engine" not in st.session_state:
    st.session_state.engine = ReasoningEngine()
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Describe your land, or answer a clarifying question...")
if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    result = handle_message(st.session_state.session_id, user_input)
    session = result["session"]

    if not result["ready"]:
        question = next_clarifying_question(session)
        reply = question or "Could you share more about soil, rainfall, and crop type?"
        st.session_state.history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        reasoning = st.session_state.engine.reason(session, region=session.get("region"))
        rec = st.session_state.engine.synthesize(reasoning)
        with st.chat_message("assistant"):
            st.markdown(f"**Recommendation:** {rec['recommendation']}")
            st.markdown(f"**Mechanism:** {rec['mechanism']}")
            st.markdown(f"**Impacted metrics:** {', '.join(rec['impacted_metrics'])}")
            st.markdown(f"**Expected improvement:** {rec['expected_improvement']}")
            st.markdown(f"**Time horizon:** {rec['time_horizon']} | **Confidence:** {rec['confidence']}")
            st.markdown(f"**Sources:** {', '.join(rec['sources'])}")
            with st.expander("Retrieved knowledge chunks (transparency)"):
                st.write(rec["retrieved_chunks"])
        st.session_state.history.append(("assistant", f"Recommendation: {rec['recommendation']}"))
