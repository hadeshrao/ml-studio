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

st.title("🤖 AI Interpretation")
st.markdown(
    "Stream an LLM-powered narrative explaining your dataset, pipeline choices, "
    "and model results in plain language, grounded in your domain context."
)
st.info("Coming soon — this page is stubbed.")
