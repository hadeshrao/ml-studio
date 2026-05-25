from __future__ import annotations

import streamlit as st

from src.state import get_project, init_state, render_pipeline_inspector

st.set_page_config(
    page_title="AI Interpretation — ML Studio", page_icon="🤖", layout="wide"
)
init_state()
render_pipeline_inspector()

project = get_project()

if project.df is None:
    st.warning("Upload data first — head to the **Data Upload** page.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

st.title("🤖 AI Interpretation")
st.markdown(
    "Chat with Claude about your dataset, pipeline, and model results. "
    "Claude has full visibility into your current project state."
)

# ── Configuration ─────────────────────────────────────────────────────────────
with st.expander("⚙️ Configuration", expanded=not project.anthropic_api_key):
    api_key_input = st.text_input(
        "Anthropic API key",
        value=project.anthropic_api_key,
        type="password",
        placeholder="sk-ant-…",
        help="Stored in session only — never written to disk.",
    )
    if api_key_input != project.anthropic_api_key:
        project.anthropic_api_key = api_key_input

    domain_input = st.text_area(
        "Domain context (optional)",
        value=project.domain_context,
        placeholder="e.g. 'This is customer churn data for a SaaS company. "
                    "Churn = 1 means the customer cancelled within 90 days.'",
        height=100,
        help="Helps Claude give domain-specific advice. Included in every message.",
    )
    if domain_input != project.domain_context:
        project.domain_context = domain_input

# ── Live context preview ──────────────────────────────────────────────────────
with st.expander("🔍 What Claude can see right now", expanded=False):
    from src.llm_client import _serialize_project
    st.markdown(_serialize_project(project, project.domain_context))

st.divider()

# ── Quick-prompt buttons ──────────────────────────────────────────────────────
col_q1, col_q2, col_q3, col_q4 = st.columns(4)

quick_prompts = {
    col_q1: ("📊 Describe my dataset", "Describe my dataset. What stands out about its structure, distributions, and any data quality issues?"),
    col_q2: ("🔍 Critique my pipeline", "Review my preprocessing and feature engineering pipeline. Are there any steps that look unnecessary, risky, or missing?"),
    col_q3: ("🤖 Explain my model results", "Explain my model's results in plain language. How well is it performing, and what do the metrics and feature importances tell us?"),
    col_q4: ("💡 What should I try next?", "Based on the current pipeline and model results, what are the most impactful things I should try to improve performance?"),
}

for col, (label, prompt) in quick_prompts.items():
    if col.button(label, use_container_width=True, disabled=not project.anthropic_api_key):
        st.session_state.pending_prompt = prompt
        st.rerun()

if not project.anthropic_api_key:
    st.caption("Enter your Anthropic API key above to enable chat.")

st.divider()

# ── Conversation history ──────────────────────────────────────────────────────
for msg in st.session_state.ai_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle pending quick-prompt or new chat input ─────────────────────────────
user_text = st.chat_input(
    "Ask Claude anything about your ML project…",
    disabled=not project.anthropic_api_key,
)

prompt_to_send: str | None = st.session_state.pop("pending_prompt", None) or user_text

if prompt_to_send and project.anthropic_api_key:
    # Show user message immediately
    st.session_state.ai_messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    # Stream assistant response
    with st.chat_message("assistant"):
        try:
            from src.llm_client import interpret

            response_chunks: list[str] = []

            def _token_stream():
                for token in interpret(
                    project=project,
                    api_key=project.anthropic_api_key,
                    domain_context=project.domain_context,
                    messages=st.session_state.ai_messages,
                ):
                    response_chunks.append(token)
                    yield token

            st.write_stream(_token_stream())
            full_response = "".join(response_chunks)

        except Exception as e:
            err_type = type(e).__name__
            if "AuthenticationError" in err_type or "auth" in str(e).lower():
                full_response = "Authentication failed — check that your API key is correct."
                st.error(full_response)
            elif "RateLimitError" in err_type or "rate" in str(e).lower():
                full_response = "Rate limit reached — wait a moment and try again."
                st.warning(full_response)
            else:
                full_response = f"Error communicating with Claude: {e}"
                st.error(full_response)

    st.session_state.ai_messages.append({"role": "assistant", "content": full_response})

# ── Footer controls ───────────────────────────────────────────────────────────
if st.session_state.ai_messages:
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=False):
        st.session_state.ai_messages = []
        st.rerun()
