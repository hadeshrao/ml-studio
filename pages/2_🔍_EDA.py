import plotly.express as px
import streamlit as st

from src.eda import column_summary, correlation_matrix, missing_value_summary
from src.state import current_df, get_project, init_state, render_pipeline_inspector

st.set_page_config(page_title="EDA — ML Studio", page_icon="🔍", layout="wide")
init_state()
render_pipeline_inspector()

project = get_project()

if project.df is None:
    st.warning("Upload data first — head to the **Data Upload** page.")
    st.stop()

df = current_df()
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
missing_pct_overall = df.isna().sum().sum() / df.size * 100

st.title("🔍 Exploratory Data Analysis")

# --- Quick metrics row ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", df.shape[1])
m3.metric("Numeric", len(num_cols))
m4.metric("Categorical", len(cat_cols))
m5.metric("Missing cells", f"{missing_pct_overall:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Distributions", "🔗 Correlations", "🎯 Target"])

# ── Tab 1: Overview ──────────────────────────────────────────────────────────

with tab1:
    st.subheader("Column Summary")
    st.dataframe(column_summary(df), use_container_width=True, hide_index=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    miss = missing_value_summary(df)
    if miss.empty:
        st.success("No missing values — dataset is complete.")
    else:
        st.subheader(f"Missing Values ({len(miss)} column{'s' if len(miss) > 1 else ''} affected)")
        fig = px.bar(
            miss,
            x="column",
            y="missing_pct",
            text_auto=".1f",
            color="missing_pct",
            color_continuous_scale="Reds",
            labels={"column": "Column", "missing_pct": "Missing (%)"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_tickangle=-30,
            margin=dict(t=20, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Distributions ──────────────────────────────────────────────────────

with tab2:
    if num_cols:
        st.subheader("Numeric Distributions")
        sel_num = st.multiselect(
            "Columns to plot",
            num_cols,
            default=num_cols[:8],
            key="dist_num",
        )
        for i in range(0, len(sel_num), 2):
            batch = sel_num[i : i + 2]
            cols_ui = st.columns(2)
            for j, col in enumerate(batch):
                with cols_ui[j]:
                    fig = px.histogram(df, x=col, nbins=40, marginal="box", title=col)
                    fig.update_layout(
                        showlegend=False,
                        height=320,
                        margin=dict(t=35, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        st.subheader("Categorical Distributions")
        sel_cat = st.multiselect(
            "Columns to plot",
            cat_cols,
            default=cat_cols[:6],
            key="dist_cat",
        )
        for i in range(0, len(sel_cat), 2):
            batch = sel_cat[i : i + 2]
            cols_ui = st.columns(2)
            for j, col in enumerate(batch):
                with cols_ui[j]:
                    vc = df[col].value_counts().head(15).reset_index()
                    vc.columns = [col, "count"]
                    fig = px.bar(vc, x=col, y="count", title=col, text="count")
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        height=320,
                        margin=dict(t=35, b=10, l=10, r=10),
                        xaxis_tickangle=-30,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    if not num_cols and not cat_cols:
        st.info("No columns to display.")

# ── Tab 3: Correlations ───────────────────────────────────────────────────────

with tab3:
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        c1, _ = st.columns([1, 3])
        method = c1.radio(
            "Method", ["pearson", "spearman", "kendall"], horizontal=True, key="corr_method"
        )

        st.subheader("Correlation Heatmap")

        # Cap columns to keep the chart readable; warn if truncated
        HEATMAP_MAX = 30
        heatmap_cols = num_cols[:HEATMAP_MAX]
        if len(num_cols) > HEATMAP_MAX:
            st.caption(
                f"Showing first {HEATMAP_MAX} of {len(num_cols)} numeric columns. "
                "Use the scatter explorer below for specific pairs."
            )

        corr = correlation_matrix(df[heatmap_cols], method=method)
        show_text = len(heatmap_cols) <= 20
        fig = px.imshow(
            corr,
            text_auto=".2f" if show_text else False,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(height=max(400, len(heatmap_cols) * 38))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scatter Explorer")
        s1, s2, s3 = st.columns(3)
        x_col = s1.selectbox("X axis", num_cols, index=0, key="scatter_x")
        y_col = s2.selectbox(
            "Y axis", num_cols, index=min(1, len(num_cols) - 1), key="scatter_y"
        )
        color_opts = ["None"] + cat_cols + [c for c in num_cols if c not in (x_col, y_col)]
        color_sel = s3.selectbox("Color by", color_opts, key="scatter_color")
        color = None if color_sel == "None" else color_sel

        fig = px.scatter(df, x=x_col, y=y_col, color=color, opacity=0.6, height=450)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Target ────────────────────────────────────────────────────────────

with tab4:
    if not project.target:
        st.info("No target column set — configure it on the **Data Upload** page.")
        st.stop()

    if project.target not in df.columns:
        st.warning(f"Target column '{project.target}' not found in the current dataframe.")
        st.stop()

    target = project.target
    task = project.task_type

    st.subheader(f"Target: `{target}`  ·  task: {task}")

    if task == "classification":
        vc = df[target].value_counts().reset_index()
        vc.columns = [target, "count"]
        vc["pct"] = (vc["count"] / len(df) * 100).round(1)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                vc,
                x=target,
                y="count",
                color=target,
                text=vc["pct"].astype(str) + "%",
                title="Class Distribution",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Class breakdown**")
            st.dataframe(vc, use_container_width=True, hide_index=True)
            n_classes = df[target].nunique()
            majority_pct = vc["pct"].max()
            if n_classes == 2 and majority_pct >= 80:
                st.warning(f"Class imbalance: majority class is {majority_pct}%.")

    elif task == "regression":
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x=target, nbins=50, marginal="box", title="Target Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            desc = df[target].describe().round(4).to_frame(name="value")
            st.markdown("**Summary statistics**")
            st.dataframe(desc, use_container_width=True)

    elif task == "clustering":
        st.info("Unsupervised task — no target distribution to show.")

    # Feature vs target plots
    feat_num = [c for c in num_cols if c != target]
    if feat_num and task in ("classification", "regression"):
        st.subheader("Numeric Features vs Target")
        sel_feats = st.multiselect(
            "Select features",
            feat_num,
            default=feat_num[:4],
            key="feat_vs_target",
        )
        if sel_feats:
            for i in range(0, len(sel_feats), 2):
                batch = sel_feats[i : i + 2]
                cols_ui = st.columns(2)
                for j, feat in enumerate(batch):
                    with cols_ui[j]:
                        if task == "classification":
                            fig = px.box(
                                df,
                                x=target,
                                y=feat,
                                color=target,
                                title=f"{feat} by {target}",
                            )
                        else:
                            fig = px.scatter(
                                df,
                                x=feat,
                                y=target,
                                opacity=0.45,
                                title=f"{feat} vs {target}",
                            )
                        fig.update_layout(
                            height=320,
                            showlegend=False,
                            margin=dict(t=35, b=10, l=10, r=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)
