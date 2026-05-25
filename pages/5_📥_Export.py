from __future__ import annotations

import pickle
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

from src.kedro_generator import generate_kedro_project
from src.state import current_df, get_project, get_steps, init_state, render_pipeline_inspector

st.set_page_config(page_title="Export — ML Studio", page_icon="📥", layout="wide")
init_state()
render_pipeline_inspector()

project = get_project()

if project.df is None:
    st.warning("Upload data first — head to the **Data Upload** page.")
    st.stop()

try:
    df = current_df()
except Exception as e:
    st.error(f"Step replay error: {e}")
    st.stop()

st.title("📥 Export")
st.markdown(
    "Download your processed dataset, trained model artifact, "
    "or the full workflow as a runnable Kedro project."
)

if project.model is None:
    st.warning(
        "No model trained yet — the **Model Artifact** and **Kedro Project** tabs "
        "require a trained model. Complete the Modelling step first."
    )

st.divider()

tab_review, tab_data, tab_model, tab_kedro = st.tabs(
    ["📋 Pipeline Review", "📊 Processed Data", "🤖 Model Artifact", "📦 Kedro Project"]
)

# ── Tab 1: Pipeline Review ────────────────────────────────────────────────────

with tab_review:
    # Dataset summary
    st.subheader("Dataset")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Raw rows", f"{project.df.shape[0]:,}")
    r1.metric("Raw columns", project.df.shape[1])
    r2.metric("Processed rows", f"{df.shape[0]:,}")
    r2.metric("Processed columns", df.shape[1])
    r3.metric("Target", project.target or "—")
    r3.metric("Task", project.task_type or "—")
    r4.metric("Feature columns", len(project.features) if project.features else "—")

    # Steps by stage
    st.subheader("Recorded Steps")
    stage_configs = [
        ("preprocessing", "🟦 Preprocessing"),
        ("feature_engineering", "🟨 Feature Engineering"),
        ("modeling", "🟩 Modeling"),
    ]
    any_steps = False
    for stage_key, stage_label in stage_configs:
        stage_steps = get_steps(stage_key)
        if not stage_steps:
            continue
        any_steps = True
        with st.expander(f"{stage_label}  ·  {len(stage_steps)} step(s)", expanded=True):
            for i, step in enumerate(stage_steps, 1):
                cols = st.columns([1, 4, 5])
                cols[0].caption(f"#{i}")
                cols[1].markdown(f"**`{step['name']}`**")
                cols[2].caption(step["description"])
                if step["params"]:
                    param_pairs = ", ".join(
                        f"{k}={v!r}"
                        for k, v in step["params"].items()
                        if not str(k).startswith("_")
                    )
                    st.caption(f"    params: `{param_pairs}`")

    if not any_steps:
        st.info("No steps recorded. Add preprocessing or feature engineering steps on their respective pages.")

    # Model summary
    if project.model is not None:
        st.subheader("Model")
        metrics = project.metrics
        algo_label = metrics.get("_algo_label", metrics.get("algorithm", "Unknown"))
        st.markdown(f"**Algorithm:** {algo_label}  ·  **Task:** {metrics.get('task_type', '—')}")

        display_keys = [
            "accuracy", "f1", "roc_auc",
            "r2", "rmse", "mae", "mape",
            "silhouette", "n_clusters",
            "cv_mean", "cv_std", "n_train", "n_test",
        ]
        metric_rows = {k: v for k, v in metrics.items() if k in display_keys and v is not None}
        if metric_rows:
            metric_df = pd.DataFrame(
                [{"Metric": k, "Value": round(v, 4) if isinstance(v, float) else v}
                 for k, v in metric_rows.items()]
            )
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

# ── Tab 2: Processed Data ─────────────────────────────────────────────────────

with tab_data:
    st.subheader("Processed Dataset")
    st.caption(
        f"{df.shape[0]:,} rows × {df.shape[1]} columns "
        f"(after {len(get_steps('preprocessing'))} preprocessing + "
        f"{len(get_steps('feature_engineering'))} feature engineering steps)"
    )
    st.dataframe(df.head(30), use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download processed_data.csv",
        data=csv_bytes,
        file_name="processed_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if project.df is not None and project.df.shape != df.shape:
        r_delta = df.shape[0] - project.df.shape[0]
        c_delta = df.shape[1] - project.df.shape[1]
        st.caption(
            f"Δ vs raw: "
            f"{'%+d' % r_delta} rows, {'%+d' % c_delta} columns"
        )

# ── Tab 3: Model Artifact ─────────────────────────────────────────────────────

with tab_model:
    if project.model is None:
        st.info("Train a model on the **Modelling** page to enable this download.")
    else:
        algo_label = project.metrics.get("_algo_label", "Model")
        st.subheader(f"Model: {algo_label}")

        # Pickle download
        model_bytes = pickle.dumps(project.model)
        st.download_button(
            label="⬇️  Download model.pkl",
            data=model_bytes,
            file_name="model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )
        st.caption(
            f"Pickle size: **{len(model_bytes) / 1024:.1f} KB**  ·  "
            f"Python {__import__('sys').version_info.major}.{__import__('sys').version_info.minor} compatible"
        )

        st.divider()

        # Metrics
        st.subheader("Training Metrics")
        display_keys = [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "r2", "rmse", "mae", "mape",
            "silhouette", "n_clusters", "inertia",
            "cv_mean", "cv_std", "cv_scoring",
            "n_train", "n_test",
        ]
        metric_rows = [
            {"Metric": k, "Value": round(v, 6) if isinstance(v, float) else v}
            for k, v in project.metrics.items()
            if k in display_keys and v is not None
        ]
        if metric_rows:
            st.dataframe(
                pd.DataFrame(metric_rows),
                use_container_width=True,
                hide_index=True,
            )

        # Feature list
        if project.features:
            with st.expander(f"Feature columns used ({len(project.features)})"):
                st.write(project.features)

        # Usage snippet
        st.subheader("Load & Predict (Python snippet)")
        snippet = f"""\
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Replace with new data — must have these columns:
# {project.features[:5]}{"..." if len(project.features) > 5 else ""}
X = pd.read_csv("new_data.csv")[{project.features!r}]
predictions = model.predict(X)
"""
        st.code(snippet, language="python")

# ── Tab 4: Kedro Project ──────────────────────────────────────────────────────

with tab_kedro:
    if project.model is None:
        st.info("Train a model on the **Modelling** page to enable Kedro export.")
        st.stop()

    st.subheader("Generate Kedro Project")
    st.markdown(
        "Creates a self-contained Kedro 0.19.x project that re-produces your entire "
        "ML workflow — from raw data through all transformation steps to model training. "
        "Run it with `kedro run` after unzipping."
    )

    k1, k2 = st.columns([2, 1])
    project_name = k1.text_input(
        "Project name", value="ml_studio_pipeline",
        help="Used as the Python package name — lowercase, underscores only."
    )
    project_name = project_name.strip().replace(" ", "_").replace("-", "_") or "ml_studio_pipeline"

    # File tree preview (static)
    with st.expander("📁 Files that will be generated", expanded=False):
        tree = f"""\
{project_name}/
├── pyproject.toml
├── README.md
├── conf/
│   └── base/
│       ├── catalog.yml
│       └── parameters.yml
├── data/
│   └── 01_raw/
│       └── input.csv          ← your dataset ({project.df.shape[0]:,} rows)
└── src/
    └── ml_studio_pipeline/
        ├── __init__.py
        ├── pipeline_registry.py
        ├── preprocessing.py   ← copied from ML Studio
        ├── feature_engineering.py
        └── pipelines/
            └── __default__/
                ├── __init__.py
                ├── nodes.py   ← train_model() using recorded hyperparams
                └── pipeline.py ← {len(get_steps('preprocessing')) + len(get_steps('feature_engineering'))} transform node(s) + train_model"""
        st.code(tree, language="text")

    # Generation
    if st.button("⚙️  Generate Kedro project", type="primary", use_container_width=True):
        with st.spinner("Rendering templates and building project…"):
            try:
                with tempfile.TemporaryDirectory() as tmp_root:
                    out_dir = Path(tmp_root) / project_name
                    out_dir.mkdir()
                    zip_path = generate_kedro_project(project, out_dir, project_name)
                    zip_bytes = zip_path.read_bytes()
                st.session_state["kedro_zip"] = zip_bytes
                st.session_state["kedro_zip_name"] = f"{project_name}.zip"
                st.toast("Kedro project generated!", icon="📦")
            except Exception as e:
                st.error(f"Generation failed: {e}")
                raise

    # Download
    zip_bytes = st.session_state.get("kedro_zip")
    zip_name = st.session_state.get("kedro_zip_name", f"{project_name}.zip")
    if zip_bytes:
        st.download_button(
            label=f"⬇️  Download {zip_name}",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )
        st.caption(f"Archive size: **{len(zip_bytes) / 1024:.1f} KB**")

    st.divider()
    st.subheader("Quick start after download")
    st.code(
        textwrap.dedent(
            f"""\
            unzip {project_name}.zip
            cd {project_name}
            pip install -e ".[dev]"
            kedro run
            """
        ),
        language="bash",
    )
    st.caption(
        ":information_source: **Scalers** in the generated pipeline are fitted on the full "
        "input dataset. For production, add a train/test split node before scaling. "
        "**Hyperparameters** are in `conf/base/parameters.yml` — edit them and re-run "
        "without touching any code."
    )
