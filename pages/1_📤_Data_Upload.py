import pandas as pd
import streamlit as st

from src.state import get_project, init_state, render_pipeline_inspector

st.set_page_config(page_title="Data Upload — ML Studio", page_icon="📤", layout="wide")
init_state()
render_pipeline_inspector()

project = get_project()

st.title("📤 Data Upload")
st.markdown("Upload a CSV file to begin your ML workflow.")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()

    errors: list[str] = []

    if df.empty or len(df.columns) == 0:
        errors.append("File has no columns or is completely empty.")

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        errors.append(f"Duplicate column names detected: {dupes}. Rename them before uploading.")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        errors.append(
            f"Fully-empty columns (drop or rename before uploading): {empty_cols}"
        )

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    project.df = df
    project.features = df.columns.tolist()

    st.success(f"Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns** successfully.")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Shape")
        st.write(f"**{df.shape[0]:,}** rows · **{df.shape[1]}** columns")
    with col2:
        st.subheader("Column Types")
        dtype_df = pd.DataFrame(
            {"dtype": df.dtypes.astype(str), "non-null count": df.notna().sum()}
        )
        st.dataframe(dtype_df, use_container_width=True)

    st.divider()
    st.subheader("Problem Setup")

    problem_idx = 1 if project.problem_type == "unsupervised" else 0
    problem_type = st.radio(
        "Learning paradigm",
        ["Supervised", "Unsupervised"],
        index=problem_idx,
        horizontal=True,
    )
    project.problem_type = problem_type.lower()  # type: ignore[assignment]

    if problem_type == "Supervised":
        default_target_idx = 0
        if project.target and project.target in df.columns.tolist():
            default_target_idx = df.columns.tolist().index(project.target)

        target = st.selectbox("Target column", df.columns.tolist(), index=default_target_idx)
        project.target = target

        target_series = df[target]
        n_unique = target_series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        suggested = "regression" if (is_numeric and n_unique > 20) else "classification"

        default_task_idx = 1 if suggested == "regression" else 0
        if project.task_type == "regression":
            default_task_idx = 1
        elif project.task_type == "classification":
            default_task_idx = 0

        task_type = st.radio(
            "Task type",
            ["Classification", "Regression"],
            index=default_task_idx,
            horizontal=True,
            help=f"Auto-suggested **{suggested}** — target '{target}' has {n_unique} unique values (dtype: {target_series.dtype}).",
        )
        project.task_type = task_type.lower()  # type: ignore[assignment]

    else:
        project.target = None
        project.task_type = "clustering"  # type: ignore[assignment]
        st.info(
            "Unsupervised path selected. No target column needed — "
            "clustering algorithms will be available on the Modelling page."
        )

elif project.df is not None:
    st.info("A dataset is already loaded. Re-upload a file above to replace it.")
    st.dataframe(project.df.head(20), use_container_width=True)
else:
    st.info("Upload a CSV file above to get started.")
