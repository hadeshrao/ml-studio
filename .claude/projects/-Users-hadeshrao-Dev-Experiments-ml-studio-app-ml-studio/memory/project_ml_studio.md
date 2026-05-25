---
name: project-ml-studio
description: Architecture and scaffold state of the ML Studio Streamlit app
metadata:
  type: project
---

ML Studio is a Streamlit-based end-to-end ML workflow tool. Stack: Python 3.11+, Streamlit, scikit-learn, plotly, kedro (export only), anthropic, optuna, xgboost, lightgbm.

**Scaffold completed on feat/scaffold branch (2026-05-25).** Single source of truth is `project.steps` list — every transformation is a Step dict (id, stage, name, params, input/output columns, description). Pages replay steps via `state.current_df()`.

Core abstraction lives in `src/state.py` (Project dataclass, init_state, add_step, remove_step, current_df, render_pipeline_inspector). All other src/ modules are stubs with STEP_REGISTRY dicts.

**Currently live:** Data Upload page (CSV validation, df storage, supervised/unsupervised setup, target/task-type selection), Pipeline Inspector sidebar.

**Stubbed:** EDA, Feature Engineering, Modelling, Export, AI Interpretation.

**Why:** Build one vertical slice per conversation; spec-driven. User will direct which page to build next.

**How to apply:** When starting a new page build, read src/state.py and the corresponding stub page for context. No kedro runtime dependency — kedro is export-only (Jinja2 templates in templates/kedro/).
