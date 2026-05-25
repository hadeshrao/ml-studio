import streamlit as st

from src.state import get_project, init_state, render_pipeline_inspector

st.set_page_config(page_title="Export — ML Studio", page_icon="📥", layout="wide")
init_state()
render_pipeline_inspector()

project = get_project()

if project.model is None:
    st.warning("No model trained yet — complete the Modelling step before exporting.")
    st.stop()

st.title("📥 Export")
st.markdown(
    "Download the full ML workflow as a runnable Kedro project, "
    "including rendered pipeline nodes, catalog, and parameters."
)
st.info("Coming soon — this page is stubbed.")
