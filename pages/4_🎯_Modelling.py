from __future__ import annotations

import json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.modeling import (
    TrainResult,
    compute_feature_importance,
    train_supervised,
    train_unsupervised,
)
from src.state import (
    add_step,
    current_df,
    get_project,
    get_steps,
    init_state,
    render_pipeline_inspector,
)

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

try:
    df = current_df()
except Exception as e:
    st.error(f"Step replay error: {e}")
    st.stop()

target = project.target
task_type = project.task_type  # "classification" | "regression" | "clustering"

# Columns eligible as features: numeric only, excluding target
numeric_cols = df.select_dtypes(include="number").columns.tolist()
non_target_numeric = [c for c in numeric_cols if c != target]
non_numeric = [c for c in df.columns if c not in numeric_cols and c != target]

st.title("🎯 Modelling")
st.markdown(f"Task: **{task_type}** · {project.problem_type}")

if non_numeric:
    st.warning(
        f"Non-numeric columns will be excluded from training: `{', '.join(non_numeric)}`. "
        "Encode them on the **Feature Engineering** page first."
    )

if not non_target_numeric:
    st.error(
        "No usable numeric feature columns found. "
        "Go to **Feature Engineering** to encode categorical columns first."
    )
    st.stop()

# ── Algorithm catalogue ────────────────────────────────────────────────────────

ALGORITHMS: dict[str, dict[str, str]] = {
    "classification": {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "K-Nearest Neighbors": "knn",
        "Support Vector Machine": "svm",
    },
    "regression": {
        "Linear Regression": "linear_regression",
        "Ridge Regression": "ridge",
        "Lasso Regression": "lasso",
        "Elastic Net": "elastic_net",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
    },
    "clustering": {
        "K-Means": "kmeans",
        "DBSCAN": "dbscan",
        "Agglomerative Clustering": "agglomerative",
    },
}

algo_options = list(ALGORITHMS.get(task_type, {}).keys())


def render_hyperparams(algo_key: str, random_state: int) -> dict:
    """Render hyperparameter widgets and return the collected params dict."""
    p: dict = {"random_state": random_state}

    if algo_key == "logistic_regression":
        c1, c2, c3 = st.columns(3)
        p["C"] = c1.number_input(
            "C (inverse regularization)", 0.001, 100.0, 1.0, 0.1,
            format="%.3f", key="hp_C",
        )
        p["penalty"] = c2.selectbox("Penalty", ["l2", "l1", "elasticnet", "None"], key="hp_penalty")
        if p["penalty"] == "None":
            p["penalty"] = None
        p["max_iter"] = int(c3.number_input("Max iterations", 100, 10000, 1000, 100, key="hp_max_iter"))
        p["solver"] = "saga"

    elif algo_key == "random_forest":
        c1, c2 = st.columns(2)
        p["n_estimators"] = c1.slider("n_estimators", 10, 500, 100, 10, key="hp_n_est")
        max_depth = c2.slider("max_depth  (0 = unlimited)", 0, 30, 0, key="hp_max_depth")
        p["max_depth"] = max_depth if max_depth > 0 else None
        c3, c4 = st.columns(2)
        p["min_samples_split"] = c3.slider("min_samples_split", 2, 20, 2, key="hp_min_split")
        mf = c4.selectbox("max_features", ["sqrt", "log2", "None"], key="hp_max_feat")
        p["max_features"] = mf if mf != "None" else None

    elif algo_key == "xgboost":
        c1, c2, c3 = st.columns(3)
        p["n_estimators"] = c1.slider("n_estimators", 50, 500, 100, 10, key="hp_n_est")
        p["max_depth"] = c2.slider("max_depth", 1, 15, 6, key="hp_max_depth")
        p["learning_rate"] = c3.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="hp_lr")
        c4, c5 = st.columns(2)
        p["subsample"] = c4.slider("subsample", 0.5, 1.0, 1.0, 0.05, key="hp_subsample")
        p["colsample_bytree"] = c5.slider("colsample_bytree", 0.5, 1.0, 1.0, 0.05, key="hp_col_bt")

    elif algo_key == "lightgbm":
        c1, c2, c3, c4 = st.columns(4)
        p["n_estimators"] = c1.slider("n_estimators", 50, 500, 100, 10, key="hp_n_est")
        p["num_leaves"] = c2.slider("num_leaves", 10, 200, 31, key="hp_num_leaves")
        p["learning_rate"] = c3.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="hp_lr")
        p["min_child_samples"] = c4.slider("min_child_samples", 5, 100, 20, key="hp_min_child")

    elif algo_key == "knn":
        c1, c2, c3 = st.columns(3)
        p["n_neighbors"] = c1.slider("n_neighbors", 1, 50, 5, key="hp_k")
        p["weights"] = c2.selectbox("weights", ["uniform", "distance"], key="hp_weights")
        p["metric"] = c3.selectbox("metric", ["minkowski", "euclidean", "manhattan"], key="hp_metric")
        del p["random_state"]

    elif algo_key == "svm":
        c1, c2 = st.columns(2)
        p["C"] = c1.number_input("C", 0.001, 100.0, 1.0, 0.1, format="%.3f", key="hp_C")
        p["kernel"] = c2.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], key="hp_kernel")
        if p["kernel"] == "poly":
            p["degree"] = st.slider("degree", 2, 5, 3, key="hp_degree")
        st.caption(":information_source: SVM can be slow on large datasets. Consider sampling or using a linear kernel.")

    elif algo_key == "linear_regression":
        st.caption("No tunable hyperparameters. Ordinary least squares.")
        p["fit_intercept"] = st.checkbox("Fit intercept", True, key="hp_intercept")
        del p["random_state"]

    elif algo_key in ("ridge", "lasso"):
        c1, c2 = st.columns(2)
        p["alpha"] = c1.number_input(
            "alpha (regularization strength)", 0.0001, 1000.0, 1.0, 0.1,
            format="%.4f", key="hp_alpha",
        )
        p["max_iter"] = int(c2.number_input("max_iter", 100, 10000, 1000, 100, key="hp_max_iter"))
        del p["random_state"]

    elif algo_key == "elastic_net":
        c1, c2, c3 = st.columns(3)
        p["alpha"] = c1.number_input("alpha", 0.0001, 1000.0, 1.0, 0.1, format="%.4f", key="hp_alpha")
        p["l1_ratio"] = c2.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05, key="hp_l1")
        p["max_iter"] = int(c3.number_input("max_iter", 100, 10000, 1000, 100, key="hp_max_iter"))
        del p["random_state"]

    elif algo_key == "kmeans":
        c1, c2, c3 = st.columns(3)
        p["n_clusters"] = c1.slider("n_clusters", 2, 20, 5, key="hp_k")
        p["n_init"] = int(c2.number_input("n_init", 1, 50, 10, key="hp_n_init"))
        p["max_iter"] = int(c3.number_input("max_iter", 100, 2000, 300, key="hp_max_iter"))

    elif algo_key == "dbscan":
        c1, c2 = st.columns(2)
        p["eps"] = c1.number_input("eps", 0.01, 10.0, 0.5, 0.05, format="%.2f", key="hp_eps")
        p["min_samples"] = c2.slider("min_samples", 1, 50, 5, key="hp_min_samples")
        del p["random_state"]
        st.caption("DBSCAN discovers the number of clusters automatically; noise points are labelled −1.")

    elif algo_key == "agglomerative":
        c1, c2 = st.columns(2)
        p["n_clusters"] = c1.slider("n_clusters", 2, 20, 5, key="hp_k")
        p["linkage"] = c2.selectbox("linkage", ["ward", "complete", "average", "single"], key="hp_linkage")
        del p["random_state"]

    return p


# ── Config section ─────────────────────────────────────────────────────────────

st.subheader("1 · Feature Selection")
sel_features = st.multiselect(
    "Feature columns (X)",
    non_target_numeric,
    default=non_target_numeric,
    key="sel_features",
)
if not sel_features:
    st.warning("Select at least one feature column.")
    st.stop()

n_nan = df[sel_features + ([target] if target else [])].isna().sum().sum()
if n_nan > 0:
    st.caption(
        f":warning: {n_nan:,} NaN cells in the selected columns — "
        "rows with any NaN will be dropped before training."
    )

st.divider()

if task_type != "clustering":
    st.subheader("2 · Train / Test Split")
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test set size", 0.05, 0.4, 0.2, 0.05, key="test_size")
    random_state = int(c2.number_input("Random seed", 0, 99999, 42, step=1, key="rand_seed"))
    cv_folds = c3.slider("CV folds", 2, 10, 5, key="cv_folds")
    st.divider()
else:
    random_state = int(st.number_input("Random seed", 0, 99999, 42, step=1, key="rand_seed"))
    test_size = 0.2
    cv_folds = 5

st.subheader("3 · Algorithm & Hyperparameters")
algo_label = st.selectbox("Algorithm", algo_options, key="algo_select")
algo_key = ALGORITHMS[task_type][algo_label]

with st.expander("Hyperparameters", expanded=True):
    hyperparams = render_hyperparams(algo_key, random_state)

st.divider()

# ── Train button ───────────────────────────────────────────────────────────────

# Capture FE+prep steps hash so we can detect stale model later
_fe_steps = [s for s in project.steps if s["stage"] != "modeling"]
_current_fe_hash = json.dumps(_fe_steps, sort_keys=True)

train_clicked = st.button(
    "🚀 Train Model",
    type="primary",
    use_container_width=True,
    key="btn_train",
)

if train_clicked:
    with st.spinner(f"Training {algo_label}…"):
        try:
            if task_type == "clustering":
                result: TrainResult = train_unsupervised(
                    df, sel_features, algo_key, hyperparams
                )
            else:
                result = train_supervised(
                    df, target, sel_features, algo_key, hyperparams,
                    test_size=test_size,
                    random_state=random_state,
                    cv_folds=cv_folds,
                    task_type=task_type,
                )
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # Persist model, metrics, features
    project.model = result.model
    project.features = sel_features
    result.metrics["_steps_hash_at_train"] = _current_fe_hash
    result.metrics["_algo_label"] = algo_label
    project.metrics = result.metrics
    st.session_state["train_result"] = result

    # Replace any prior modeling step with this one
    project.steps = [s for s in project.steps if s["stage"] != "modeling"]
    add_step(
        "modeling",
        algo_key,
        {k: v for k, v in hyperparams.items() if isinstance(v, (str, int, float, bool, type(None)))},
        sel_features,
        [],
        f"Train {algo_label}",
    )
    st.toast(f"{algo_label} trained successfully!", icon="🎉")
    st.rerun()

# ── Results ────────────────────────────────────────────────────────────────────

train_result: TrainResult | None = st.session_state.get("train_result")

if project.model is None or train_result is None:
    st.info("Configure options above and click **Train Model** to see results here.")
    st.stop()

st.divider()
st.subheader(f"Results · {project.metrics.get('_algo_label', algo_label)}")

# Stale model warning
if project.metrics.get("_steps_hash_at_train") != _current_fe_hash:
    st.warning(
        "Pipeline steps have changed since the last training run. "
        "Results may not reflect the current feature set — consider retraining."
    )

metrics = project.metrics
result = train_result

tab_metrics, tab_importance, tab_diag = st.tabs(
    ["📊 Metrics", "📈 Feature Importance", "🔬 Diagnostic Plots"]
)

# ── Metrics tab ───────────────────────────────────────────────────────────────

with tab_metrics:
    if task_type == "classification":
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("Precision (w)", f"{metrics['precision']:.4f}")
        m3.metric("Recall (w)", f"{metrics['recall']:.4f}")
        m4.metric("F1 (weighted)", f"{metrics['f1']:.4f}")
        if "roc_auc" in metrics:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    elif task_type == "regression":
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{metrics['r2']:.4f}")
        m2.metric("RMSE", f"{metrics['rmse']:,.4f}")
        m3.metric("MAE", f"{metrics['mae']:,.4f}")
        m4.metric("MAPE", f"{metrics['mape']:.2f}%" if "mape" in metrics else "—")

    elif task_type == "clustering":
        m1, m2, m3 = st.columns(3)
        m1.metric("Clusters found", metrics["n_clusters"])
        m2.metric("Silhouette score",
                  f"{metrics['silhouette']:.4f}" if "silhouette" in metrics else "—")
        if "inertia" in metrics:
            m3.metric("Inertia", f"{metrics['inertia']:,.1f}")
        if metrics.get("n_noise", 0):
            st.caption(f"Noise points (DBSCAN label −1): {metrics['n_noise']:,}")

    # CV summary (supervised only)
    if result.cv_scores.size > 0:
        st.divider()
        scoring_label = metrics.get("cv_scoring", "score").replace("_", " ").title()
        st.subheader(f"Cross-validation ({len(result.cv_scores)}-fold {scoring_label})")
        cv_mean = metrics["cv_mean"]
        cv_std = metrics["cv_std"]
        st.metric(
            f"Mean {scoring_label}",
            f"{cv_mean:.4f}",
            delta=f"± {cv_std:.4f}",
            delta_color="off",
        )

        # Per-fold bar chart
        fold_df = {
            "Fold": [f"Fold {i+1}" for i in range(len(result.cv_scores))],
            scoring_label: result.cv_scores,
        }
        fig = px.bar(
            fold_df,
            x="Fold",
            y=scoring_label,
            title=f"Per-fold {scoring_label}",
            text_auto=".4f",
        )
        fig.add_hline(y=cv_mean, line_dash="dash", annotation_text=f"mean={cv_mean:.4f}")
        fig.update_layout(height=300, margin=dict(t=35, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Train / test sizes
    if "n_train" in metrics:
        st.caption(
            f"Train set: **{metrics['n_train']:,}** rows · "
            f"Test set: **{metrics['n_test']:,}** rows"
        )

# ── Feature Importance tab ────────────────────────────────────────────────────

with tab_importance:
    fi_df = compute_feature_importance(result.model, result.feature_names)
    if fi_df is None:
        st.info(
            f"{metrics.get('_algo_label', algo_key)} does not expose feature importances "
            "directly. Consider using a tree-based model (Random Forest, XGBoost, LightGBM) "
            "for interpretability."
        )
    else:
        top_n = min(30, len(fi_df))
        fig = px.bar(
            fi_df.head(top_n),
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {top_n} feature importances",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=max(350, top_n * 22),
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Full importance table"):
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

# ── Diagnostic Plots tab ──────────────────────────────────────────────────────

with tab_diag:
    y_test = result.y_test
    y_pred = result.y_pred
    y_prob = result.y_prob

    if task_type == "classification":
        from sklearn.metrics import confusion_matrix

        labels_uniq = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels_uniq)
        fig_cm = px.imshow(
            cm,
            x=[str(l) for l in labels_uniq],
            y=[str(l) for l in labels_uniq],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            labels={"x": "Predicted", "y": "Actual"},
        )
        fig_cm.update_layout(height=max(350, len(labels_uniq) * 40))
        st.plotly_chart(fig_cm, use_container_width=True)

        # ROC curve — binary only
        if y_prob is not None and len(labels_uniq) == 2:
            from sklearn.metrics import auc, roc_curve

            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc_val = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    name=f"AUC = {roc_auc_val:.4f}",
                    fill="tozeroy",
                    line=dict(color="steelblue"),
                )
            )
            fig_roc.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(color="grey", dash="dash"),
            )
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    elif task_type == "regression":
        col1, col2 = st.columns(2)

        # Predicted vs Actual
        with col1:
            pred_df = {"Actual": y_test.values, "Predicted": y_pred}
            fig_pva = px.scatter(
                pred_df, x="Actual", y="Predicted",
                opacity=0.4, title="Predicted vs Actual",
            )
            mn = min(y_test.min(), y_pred.min())
            mx = max(y_test.max(), y_pred.max())
            fig_pva.add_shape(
                type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                line=dict(color="red", dash="dash"),
            )
            fig_pva.update_layout(height=420)
            st.plotly_chart(fig_pva, use_container_width=True)

        # Residuals
        with col2:
            residuals = y_test.values - y_pred
            fig_res = px.histogram(
                {"Residuals": residuals}, x="Residuals",
                nbins=50, marginal="box", title="Residuals Distribution",
            )
            fig_res.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_res, use_container_width=True)

        # Residuals vs Predicted
        fig_rvp = px.scatter(
            {"Predicted": y_pred, "Residuals": residuals},
            x="Predicted", y="Residuals",
            opacity=0.4, title="Residuals vs Predicted",
        )
        fig_rvp.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rvp, use_container_width=True)

    elif task_type == "clustering":
        from sklearn.decomposition import PCA

        X_plot = result.X_test[result.feature_names].values
        cluster_labels = result.y_pred.astype(str)

        # Cluster size bar
        import pandas as pd
        label_counts = pd.Series(result.y_pred).value_counts().sort_index()
        fig_sizes = px.bar(
            x=label_counts.index.astype(str),
            y=label_counts.values,
            labels={"x": "Cluster", "y": "Count"},
            title="Cluster Sizes",
            color=label_counts.index.astype(str),
        )
        fig_sizes.update_layout(showlegend=False)
        st.plotly_chart(fig_sizes, use_container_width=True)

        # PCA 2-D scatter
        if X_plot.shape[1] >= 2:
            n_comps = min(2, X_plot.shape[1])
            pca_coords = PCA(n_components=n_comps).fit_transform(X_plot)
            pca_df = {"PC1": pca_coords[:, 0], "PC2": pca_coords[:, 1], "Cluster": cluster_labels}
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="Cluster",
                opacity=0.6,
                title="PCA (2-D) — coloured by cluster",
            )
            fig_pca.update_layout(height=480)
            st.plotly_chart(fig_pca, use_container_width=True)
