import streamlit as st

from src.state import get_project, init_state, render_pipeline_inspector

st.set_page_config(page_title="Modelling — ML Studio", page_icon="🎯", layout="wide")
init_state()
render_pipeline_inspector()

project = get_project()

if project.df is None:
    st.warning("Upload data first — head to the **Data Upload** page.")
    st.stop()

if project.problem_type == "supervised" and project.target is None:
    st.warning("Select a target column first — head to the **Data Upload** page.")
    st.stop()

st.title("🎯 Modelling")
st.markdown(
    "Select and tune a model, run cross-validation, and inspect performance metrics."
)
st.info("Coming soon — this page is stubbed.")
