# ML Studio

An end-to-end machine learning workflow tool built with Streamlit. Data scientists upload a dataset, perform EDA, preprocess data, engineer features, train a model, and export the entire pipeline as a runnable Kedro project — all through a point-and-click UI.

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Navigate pages using the sidebar.

## Architecture: the step pattern

Every transformation a user applies is recorded as a **Step** dict appended to `st.session_state.project.steps`. A Step captures the stage (`preprocessing`, `feature_engineering`, or `modeling`), the registry key (`impute_mean`, `one_hot_encode`, …), the params dict, and the affected columns.

`state.current_df()` replays the full steps list against the raw DataFrame on demand, using `@st.cache_data` keyed on a hash of the steps. The Modelling page replays the same steps on held-out test data. The Kedro exporter walks the list to generate typed nodes and a `pipeline.py` — meaning the exported project is a faithful reproduction of every decision made in the UI.

## Currently stubbed

- [ ] EDA — distributions, correlation heatmaps, missing-value charts
- [ ] Feature Engineering — log transforms, one-hot encoding, scaling, interaction terms
- [ ] Modelling — algorithm selection, cross-validation, hyperparameter tuning via Optuna
- [ ] Export — Jinja2 rendering of Kedro project templates, zip download
- [ ] AI Interpretation — streaming LLM narrative via Anthropic API
- [ ] Kedro template bodies (skeletal `.j2` files exist, content TBD)
