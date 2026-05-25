from collections import Counter

import streamlit as st

from src.state import get_project, init_state, render_pipeline_inspector

st.set_page_config(page_title="ML Studio", page_icon="🧪", layout="wide")
init_state()
render_pipeline_inspector()

project = get_project()

st.title("🧪 ML Studio")
st.markdown(
    "An end-to-end machine learning workflow tool. Upload your data, explore it, "
    "engineer features, train a model, and export the entire pipeline as a runnable "
    "Kedro project — all from a single UI."
)

st.divider()
st.subheader("Project Status")

col1, col2, col3 = st.columns(3)

with col1:
    if project.df is not None:
        st.metric("Dataset", f"{project.df.shape[0]:,} rows × {project.df.shape[1]} cols")
    else:
        st.metric("Dataset", "Not loaded")

with col2:
    st.metric("Target column", project.target or "—")
    if project.problem_type:
        task_label = project.task_type or "—"
        st.caption(f"{project.problem_type} / {task_label}")

with col3:
    st.metric("Model trained", "Yes" if project.model is not None else "No")

st.divider()
st.subheader("Pipeline Steps")

stage_counts = Counter(s["stage"] for s in project.steps)

cols = st.columns(3)
with cols[0]:
    st.metric("🟦 Preprocessing", stage_counts.get("preprocessing", 0))
with cols[1]:
    st.metric("🟨 Feature Engineering", stage_counts.get("feature_engineering", 0))
with cols[2]:
    st.metric("🟩 Modeling", stage_counts.get("modeling", 0))
