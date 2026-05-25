"""
Kedro project generator.

Walks project.steps, enriches them with sequential input/output dataset names,
renders five Jinja2 templates from templates/kedro/, builds a complete Kedro
0.19.x project layout in a temp directory, and zips it for download.

The generated project is self-contained:
  - preprocessing.py and feature_engineering.py are copied verbatim (they are
    pure functions with no Streamlit dependency).
  - Each pipeline step becomes a Kedro node whose params are baked in via
    functools.partial so the pipeline is immediately runnable with `kedro run`.
  - The model training node re-trains from scratch using the recorded
    hyperparameters; to retune, edit conf/base/parameters.yml.

Note on scalers: they are currently fitted on the full input dataset. For
production, split the data before the scaling nodes and fit only on train data.
"""
from __future__ import annotations

import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2

if TYPE_CHECKING:
    from src.state import Project

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "kedro"

# Maps (algorithm_key, task_type) -> (import_module, class_name)
_ALGO_CLASS: dict[tuple[str, str], tuple[str, str]] = {
    ("logistic_regression", "classification"): ("sklearn.linear_model", "LogisticRegression"),
    ("random_forest", "classification"): ("sklearn.ensemble", "RandomForestClassifier"),
    ("random_forest", "regression"): ("sklearn.ensemble", "RandomForestRegressor"),
    ("xgboost", "classification"): ("xgboost", "XGBClassifier"),
    ("xgboost", "regression"): ("xgboost", "XGBRegressor"),
    ("lightgbm", "classification"): ("lightgbm", "LGBMClassifier"),
    ("lightgbm", "regression"): ("lightgbm", "LGBMRegressor"),
    ("knn", "classification"): ("sklearn.neighbors", "KNeighborsClassifier"),
    ("knn", "regression"): ("sklearn.neighbors", "KNeighborsRegressor"),
    ("svm", "classification"): ("sklearn.svm", "SVC"),
    ("svm", "regression"): ("sklearn.svm", "SVR"),
    ("linear_regression", "regression"): ("sklearn.linear_model", "LinearRegression"),
    ("ridge", "regression"): ("sklearn.linear_model", "Ridge"),
    ("lasso", "regression"): ("sklearn.linear_model", "Lasso"),
    ("elastic_net", "regression"): ("sklearn.linear_model", "ElasticNet"),
    ("kmeans", "clustering"): ("sklearn.cluster", "KMeans"),
    ("dbscan", "clustering"): ("sklearn.cluster", "DBSCAN"),
    ("agglomerative", "clustering"): ("sklearn.cluster", "AgglomerativeClustering"),
}

_EXTRA_DEPS: dict[str, list[str]] = {
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
}


def _enrich_steps(steps: list[dict]) -> tuple[list[dict], str]:
    """Add input_dataset, output_dataset, node_name to each step. Returns (enriched, last_output)."""
    enriched = []
    prev = "raw_data"
    for i, step in enumerate(steps):
        out = f"step_{i:02d}_{step['name']}"
        enriched.append(
            {
                **step,
                "input_dataset": prev,
                "output_dataset": out,
                "node_name": f"{step['name']}_{i:02d}",
            }
        )
        prev = out
    return enriched, prev


def _build_context(project: "Project", project_name: str) -> dict[str, Any]:
    transform_steps = [s for s in project.steps if s["stage"] != "modeling"]
    model_step = next((s for s in project.steps if s["stage"] == "modeling"), None)

    algorithm = project.metrics.get("algorithm", "")
    task_type = project.task_type or "classification"
    algo_label = project.metrics.get("_algo_label", algorithm)

    module, cls = _ALGO_CLASS.get(
        (algorithm, task_type), ("sklearn.base", "BaseEstimator")
    )
    model_import = f"from {module} import {cls}"

    hyperparams = {
        k: v
        for k, v in (model_step["params"] if model_step else {}).items()
        if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
    }

    extra_deps: list[str] = []
    for algo_key, deps in _EXTRA_DEPS.items():
        if algo_key in algorithm:
            extra_deps.extend(deps)

    steps_with_io, last_dataset = _enrich_steps(transform_steps)
    intermediate_datasets = [s["output_dataset"] for s in steps_with_io]

    # Unique fn_name → stage_module (preserves insertion order, deduplicates)
    fn_imports: dict[str, str] = {}
    for step in transform_steps:
        fn_imports.setdefault(step["name"], step["stage"])

    safe_metrics = {
        k: v
        for k, v in project.metrics.items()
        if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
    }

    return {
        "project_name": project_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "steps_with_io": steps_with_io,
        "last_dataset": last_dataset,
        "intermediate_datasets": intermediate_datasets,
        "fn_imports": fn_imports,
        "model_import": model_import,
        "model_class": cls,
        "algorithm": algorithm,
        "algorithm_label": algo_label,
        "hyperparams": hyperparams,
        "features": project.features or [],
        "target": project.target or "",
        "task_type": task_type,
        "metrics": safe_metrics,
        "extra_dependencies": extra_deps,
        "transform_steps": transform_steps,
    }


def _write_pipeline_registry(pkg_dir: Path) -> None:
    (pkg_dir / "pipeline_registry.py").write_text(
        textwrap.dedent(
            """\
            from kedro.framework.project import find_pipelines
            from kedro.pipeline import Pipeline


            def register_pipelines() -> dict[str, Pipeline]:
                pipelines = find_pipelines()
                pipelines["__default__"] = sum(pipelines.values())
                return pipelines
            """
        ),
        encoding="utf-8",
    )


def _write_readme(root: Path, ctx: dict) -> None:
    step_lines = "\n".join(
        f"- {s['description']}" for s in ctx["transform_steps"]
    )
    if ctx["algorithm_label"]:
        step_lines += f"\n- Train {ctx['algorithm_label']}"

    content = textwrap.dedent(
        f"""\
        # {ctx['project_name']}

        Auto-generated Kedro project from ML Studio.
        Generated: {ctx['generated_at']}

        ## Quick start

        ```bash
        pip install -e ".[dev]"
        kedro run
        ```

        ## Pipeline steps

        {step_lines or '(no preprocessing steps — trains directly on raw data)'}

        ## Notes

        - Scalers in this pipeline are fitted on the **full input dataset**.
          For production, split the data before scaling nodes and fit only on train.
        - The model is re-trained from scratch on each `kedro run`.
          Hyperparameters live in `conf/base/parameters.yml`.
        - Raw data is bundled in `data/01_raw/input.csv`.
          Replace it with new data and re-run to retrain on fresh data.
        """
    )
    (root / "README.md").write_text(content, encoding="utf-8")


def generate_kedro_project(project: "Project", output_dir: Path, project_name: str = "ml_studio_pipeline") -> Path:
    """
    Render templates and build a complete Kedro project in output_dir.
    Returns the path to the created .zip archive.
    """
    ctx = _build_context(project, project_name)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=False,
    )

    # Directory layout
    pkg = output_dir / "src" / "ml_studio_pipeline"
    pipeline_dir = pkg / "pipelines" / "__default__"
    conf_dir = output_dir / "conf" / "base"
    data_raw_dir = output_dir / "data" / "01_raw"
    data_int_dir = output_dir / "data" / "02_intermediate"
    models_dir = output_dir / "data" / "06_models"

    for d in [pkg, pipeline_dir, conf_dir, data_raw_dir, data_int_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pipeline_dir / "__init__.py").write_text("", encoding="utf-8")

    # Copy pure transformation modules verbatim
    src_root = Path(__file__).parent
    for fname in ("preprocessing.py", "feature_engineering.py"):
        shutil.copy(src_root / fname, pkg / fname)

    # Render templates
    def render(template_name: str, dest: Path) -> None:
        rendered = env.get_template(template_name).render(**ctx)
        dest.write_text(rendered, encoding="utf-8")

    render("nodes.py.j2", pipeline_dir / "nodes.py")
    render("pipeline.py.j2", pipeline_dir / "pipeline.py")
    render("catalog.yml.j2", conf_dir / "catalog.yml")
    render("parameters.yml.j2", conf_dir / "parameters.yml")
    render("pyproject.toml.j2", output_dir / "pyproject.toml")

    _write_pipeline_registry(pkg)
    _write_readme(output_dir, ctx)

    # Bundle raw data (cap at 50k rows to keep zip manageable)
    if project.df is not None:
        sample = project.df.head(50_000)
        sample.to_csv(data_raw_dir / "input.csv", index=False)

    # Create zip — archive is placed at output_dir.parent / output_dir.name + ".zip"
    zip_base = str(output_dir)
    shutil.make_archive(
        zip_base,
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    return Path(zip_base + ".zip")
