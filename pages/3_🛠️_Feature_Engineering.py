import plotly.express as px
import streamlit as st

from src.state import (
    add_step,
    current_df,
    get_project,
    get_steps,
    init_state,
    render_pipeline_inspector,
)

st.set_page_config(
    page_title="Feature Engineering — ML Studio", page_icon="🛠️", layout="wide"
)
init_state()
render_pipeline_inspector()

project = get_project()

if project.df is None:
    st.warning("Upload data first — head to the **Data Upload** page.")
    st.stop()

try:
    df = current_df()
except Exception as e:
    st.error(f"Step replay failed — a previous step may be invalid: {e}")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
all_cols = df.columns.tolist()
cols_with_missing = [c for c in all_cols if df[c].isna().any()]
n_prep = len(get_steps("preprocessing"))
n_fe = len(get_steps("feature_engineering"))

st.title("🛠️ Feature Engineering")

# Quick stats
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", df.shape[1])
m3.metric("Numeric", len(num_cols))
m4.metric("Categorical", len(cat_cols))
m5.metric("Steps applied", n_prep + n_fe)

st.divider()

tab_prep, tab_fe, tab_preview = st.tabs(
    ["🔧 Preprocessing", "✨ Feature Creation", "👁️ Preview"]
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _add(stage, name, params, in_cols, out_cols, desc):
    add_step(stage, name, params, in_cols, out_cols, desc)
    st.toast(f"Step added: {desc}", icon="✅")


def _no_cols_warning(label="columns"):
    st.warning(f"Select at least one {label} to add this step.")


# ── Tab 1: Preprocessing ─────────────────────────────────────────────────────

with tab_prep:

    # --- Drop columns --------------------------------------------------------
    with st.expander("🗑️ Drop columns"):
        st.caption("Permanently remove columns from the working dataframe.")
        drop_sel = st.multiselect("Columns to drop", all_cols, key="drop_cols")
        if st.button("Add step", key="btn_drop"):
            if not drop_sel:
                _no_cols_warning()
            else:
                _add("preprocessing", "drop_columns",
                     {"columns": drop_sel},
                     drop_sel, [],
                     f"Drop columns: {', '.join(drop_sel)}")

    # --- Impute missing values ------------------------------------------------
    with st.expander("🩹 Impute missing values",
                     expanded=bool(cols_with_missing)):
        if not cols_with_missing:
            st.success("No missing values in the current dataframe.")
        else:
            st.caption(
                f"{len(cols_with_missing)} column(s) contain missing values. "
                "Choose a fill strategy."
            )
            imp_strategy = st.selectbox(
                "Strategy",
                ["mean", "median", "mode", "constant"],
                key="imp_strategy",
            )
            imp_cols = st.multiselect(
                "Columns",
                cols_with_missing,
                default=cols_with_missing,
                key="imp_cols",
            )
            fill_val = ""
            if imp_strategy == "constant":
                fill_val = st.text_input("Fill value", value="0", key="imp_const")

            if st.button("Add step", key="btn_impute"):
                if not imp_cols:
                    _no_cols_warning()
                else:
                    params = {"columns": imp_cols}
                    name = f"impute_{imp_strategy}"
                    if imp_strategy == "constant":
                        params["value"] = fill_val
                        desc = f"Impute {imp_cols} with constant '{fill_val}'"
                    else:
                        desc = f"Impute {imp_cols} with {imp_strategy}"
                    _add("preprocessing", name, params, imp_cols, imp_cols, desc)

    # --- Remove duplicates ----------------------------------------------------
    with st.expander("🔁 Remove duplicate rows"):
        n_dupes = int(df.duplicated().sum())
        if n_dupes == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"{n_dupes:,} duplicate row(s) detected.")
        if st.button("Add step", key="btn_dedup", disabled=(n_dupes == 0)):
            _add("preprocessing", "remove_duplicates", {}, [], [],
                 f"Remove {n_dupes:,} duplicate rows")

    # --- Cast to numeric ------------------------------------------------------
    with st.expander("🔢 Cast columns to numeric"):
        st.caption(
            "Converts object/string columns to float (non-parseable values become NaN). "
            "Useful for columns stored as strings like '1,234' or '3.14%'."
        )
        cast_sel = st.multiselect("Columns", cat_cols, key="cast_num_cols")
        if st.button("Add step", key="btn_cast"):
            if not cast_sel:
                _no_cols_warning()
            else:
                _add("preprocessing", "cast_numeric",
                     {"columns": cast_sel},
                     cast_sel, cast_sel,
                     f"Cast to numeric: {', '.join(cast_sel)}")

# ── Tab 2: Feature Creation ──────────────────────────────────────────────────

with tab_fe:

    # --- Encode categoricals -------------------------------------------------
    with st.expander("🏷️ Encode categorical columns"):
        enc_type = st.selectbox(
            "Encoding type",
            ["One-hot encoding (OHE)", "Label encoding (integer codes)"],
            key="enc_type",
        )
        enc_cols = st.multiselect("Columns", cat_cols, key="enc_cols")

        if enc_type.startswith("One-hot"):
            drop_first = st.checkbox(
                "Drop first dummy (avoid multicollinearity)",
                value=False,
                key="ohe_drop_first",
            )
            st.caption(
                "Creates one binary column per category value. "
                "Replaces the original column(s)."
            )
            if st.button("Add step", key="btn_ohe"):
                if not enc_cols:
                    _no_cols_warning()
                else:
                    _add("feature_engineering", "encode_onehot",
                         {"columns": enc_cols, "drop_first": drop_first},
                         enc_cols, [],
                         f"One-hot encode: {', '.join(enc_cols)}"
                         + (" (drop first)" if drop_first else ""))
        else:
            st.caption(
                "Replaces each category with an integer code (alphabetical order). "
                "Use ordinal encoding when categories have a natural order."
            )
            if st.button("Add step", key="btn_label"):
                if not enc_cols:
                    _no_cols_warning()
                else:
                    _add("feature_engineering", "encode_label",
                         {"columns": enc_cols},
                         enc_cols, enc_cols,
                         f"Label encode: {', '.join(enc_cols)}")

    # --- Scale numerics -------------------------------------------------------
    with st.expander("📏 Scale numeric columns"):
        scaler_type = st.selectbox(
            "Scaler",
            [
                "Standard (z-score, mean=0 std=1)",
                "Min-Max (range 0–1)",
                "Robust (median + IQR, outlier-resistant)",
            ],
            key="scaler_type",
        )
        scaler_cols = st.multiselect("Columns", num_cols, key="scaler_cols")
        st.caption(
            ":warning: Scalers are fitted on the **full dataset** here. "
            "The kedro export wraps them in proper train/test-safe nodes."
        )

        scaler_map = {
            "Standard (z-score, mean=0 std=1)": ("scale_standard", "Standard scale"),
            "Min-Max (range 0–1)": ("scale_minmax", "Min-max scale"),
            "Robust (median + IQR, outlier-resistant)": ("scale_robust", "Robust scale"),
        }
        fn_name, fn_label = scaler_map[scaler_type]

        if st.button("Add step", key="btn_scale"):
            if not scaler_cols:
                _no_cols_warning()
            else:
                _add("feature_engineering", fn_name,
                     {"columns": scaler_cols},
                     scaler_cols, scaler_cols,
                     f"{fn_label}: {', '.join(scaler_cols)}")

    # --- Numeric transforms ---------------------------------------------------
    with st.expander("📐 Apply numeric transform"):
        transform_type = st.selectbox(
            "Transform",
            [
                "log1p  (ln(x+1), good for right-skewed)",
                "sqrt   (√x, moderate skew)",
                "power  Yeo-Johnson (handles negatives)",
                "power  Box-Cox (positive values only)",
            ],
            key="transform_type",
        )
        transform_cols = st.multiselect("Columns", num_cols, key="transform_cols")

        transform_map = {
            "log1p  (ln(x+1), good for right-skewed)":
                ("transform_log1p", {}, "log1p"),
            "sqrt   (√x, moderate skew)":
                ("transform_sqrt", {}, "sqrt"),
            "power  Yeo-Johnson (handles negatives)":
                ("transform_power", {"method": "yeo-johnson"}, "power (yeo-johnson)"),
            "power  Box-Cox (positive values only)":
                ("transform_power", {"method": "box-cox"}, "power (box-cox)"),
        }
        t_fn, t_extra_params, t_label = transform_map[transform_type]

        st.caption("Creates new columns with a suffix, keeping the originals intact.")

        if st.button("Add step", key="btn_transform"):
            if not transform_cols:
                _no_cols_warning()
            else:
                params = {"columns": transform_cols, **t_extra_params}
                suffix = t_label.split("(")[0].strip()
                out_cols = [f"{c}_{suffix.replace(' ', '_')}" for c in transform_cols]
                _add("feature_engineering", t_fn,
                     params,
                     transform_cols, out_cols,
                     f"{t_label} transform: {', '.join(transform_cols)}")

    # --- Bin numeric column ---------------------------------------------------
    with st.expander("🪣 Bin numeric column"):
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            bin_col = st.selectbox("Column to bin", num_cols, key="bin_col")
            n_bins = st.slider("Number of bins", min_value=2, max_value=20,
                               value=5, key="bin_n")
            bin_strategy = st.radio(
                "Bin strategy",
                ["quantile (equal-frequency)", "uniform (equal-width)"],
                horizontal=True,
                key="bin_strategy",
            )
            strategy_key = "quantile" if bin_strategy.startswith("quantile") else "uniform"
            out_name = f"{bin_col}_bin"
            st.caption(f"Creates new column: **{out_name}** (integer 0…{n_bins - 1})")

            if st.button("Add step", key="btn_bin"):
                _add("feature_engineering", "bin_numeric",
                     {"column": bin_col, "n_bins": n_bins, "strategy": strategy_key},
                     [bin_col], [out_name],
                     f"Bin '{bin_col}' → {n_bins} {strategy_key} bins")

    # --- Interaction term -----------------------------------------------------
    with st.expander("✖️ Create interaction term"):
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns to create an interaction.")
        else:
            ia1, ia2 = st.columns(2)
            col_a = ia1.selectbox("Column A", num_cols, index=0, key="ia_col_a")
            col_b = ia2.selectbox("Column B", num_cols, index=1, key="ia_col_b")
            out_ia = f"{col_a}_x_{col_b}"
            st.caption(f"Creates new column: **{out_ia}** = {col_a} × {col_b}")

            if st.button("Add step", key="btn_interaction"):
                if col_a == col_b:
                    st.warning("Choose two different columns.")
                else:
                    _add("feature_engineering", "create_interaction",
                         {"col_a": col_a, "col_b": col_b},
                         [col_a, col_b], [out_ia],
                         f"Interaction: {col_a} × {col_b}")

# ── Tab 3: Preview ────────────────────────────────────────────────────────────

with tab_preview:
    raw = project.df
    raw_cols = set(raw.columns)
    cur_cols = set(df.columns)

    added = sorted(cur_cols - raw_cols)
    removed = sorted(raw_cols - cur_cols)
    row_delta = df.shape[0] - raw.shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}",
              delta=f"{row_delta:+,}" if row_delta != 0 else None)
    c2.metric("Columns", df.shape[1],
              delta=f"{df.shape[1] - raw.shape[1]:+}" if df.shape[1] != raw.shape[1] else None)
    c3.metric("Columns added", len(added))
    c4.metric("Columns removed", len(removed))

    if added:
        st.success(f"New columns: {', '.join(added)}")
    if removed:
        st.info(f"Removed columns: {', '.join(removed)}")

    if n_prep + n_fe == 0:
        st.caption("No steps applied yet — showing raw data.")

    st.subheader("Working Dataframe (first 30 rows)")
    st.dataframe(df.head(30), use_container_width=True)

    # Distribution of newly created columns
    if added:
        added_num = [c for c in added if c in df.select_dtypes(include="number").columns]
        if added_num:
            st.subheader("Distributions of new numeric columns")
            for i in range(0, len(added_num), 2):
                batch = added_num[i : i + 2]
                grid = st.columns(2)
                for j, col in enumerate(batch):
                    with grid[j]:
                        fig = px.histogram(df, x=col, nbins=40, marginal="box",
                                           title=col)
                        fig.update_layout(showlegend=False, height=280,
                                          margin=dict(t=35, b=10, l=10, r=10))
                        st.plotly_chart(fig, use_container_width=True)
