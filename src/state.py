from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd
import streamlit as st


@dataclass
class Project:
    df: Optional[pd.DataFrame] = None
    target: Optional[str] = None
    problem_type: Optional[Literal["supervised", "unsupervised"]] = None
    task_type: Optional[Literal["classification", "regression", "clustering"]] = None
    steps: list[dict] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    model: Optional[Any] = None
    metrics: dict = field(default_factory=dict)
    domain_context: str = ""
    anthropic_api_key: str = ""


def init_state() -> None:
    """Idempotent — safe to call at the top of every page."""
    if "project" not in st.session_state:
        st.session_state.project = Project()


def get_project() -> Project:
    return st.session_state.project


def add_step(
    stage: str,
    name: str,
    params: dict,
    input_columns: list[str],
    output_columns: list[str],
    description: str,
) -> str:
    step_id = str(uuid.uuid4())
    st.session_state.project.steps.append(
        {
            "id": step_id,
            "stage": stage,
            "name": name,
            "params": params,
            "input_columns": input_columns,
            "output_columns": output_columns,
            "description": description,
        }
    )
    return step_id


def remove_step(step_id: str) -> None:
    project = st.session_state.project
    project.steps = [s for s in project.steps if s["id"] != step_id]


def get_steps(stage: Optional[str] = None) -> list[dict]:
    steps = st.session_state.project.steps
    if stage is not None:
        return [s for s in steps if s["stage"] == stage]
    return steps


def _steps_hash(steps: list[dict]) -> str:
    return hashlib.md5(json.dumps(steps, sort_keys=True).encode()).hexdigest()


@st.cache_data
def _apply_steps_cached(steps_json: str, df: pd.DataFrame) -> pd.DataFrame:
    from src.feature_engineering import STEP_REGISTRY as fe_reg
    from src.preprocessing import STEP_REGISTRY as prep_reg

    registry = {**prep_reg, **fe_reg}
    steps = json.loads(steps_json)
    result = df.copy()
    for step in steps:
        fn = registry.get(step["name"])
        if fn is not None:
            result = fn(result, **step["params"])
    return result


def current_df() -> Optional[pd.DataFrame]:
    project = st.session_state.project
    if project.df is None:
        return None
    return _apply_steps_cached(
        json.dumps(project.steps, sort_keys=True),
        project.df,
    )


_STAGE_ICONS = {
    "preprocessing": "🟦",
    "feature_engineering": "🟨",
    "modeling": "🟩",
}


def render_pipeline_inspector() -> None:
    with st.sidebar:
        st.subheader("Pipeline Inspector")
        project = get_project()
        steps = project.steps
        if not steps:
            st.caption("No steps recorded yet.")
            return
        for stage in ("preprocessing", "feature_engineering", "modeling"):
            stage_steps = [s for s in steps if s["stage"] == stage]
            if not stage_steps:
                continue
            icon = _STAGE_ICONS.get(stage, "⬜")
            label = stage.replace("_", " ").title()
            st.markdown(f"**{icon} {label}** ({len(stage_steps)})")
            for step in stage_steps:
                col1, col2 = st.columns([5, 1])
                col1.caption(step["description"])
                if col2.button("✕", key=f"remove_{step['id']}"):
                    remove_step(step["id"])
                    st.rerun()
