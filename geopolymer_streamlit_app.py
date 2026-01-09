#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geopolymer_streamlit_app.py

Interactive Streamlit GUI for geopolymer strength prediction using an
XGBoost model trained on DataSet.xlsx.

Requirements (install as needed):

    pip install streamlit numpy pandas joblib xgboost plotly openpyxl

Run with:

    streamlit run geopolymer_streamlit_app.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor  # noqa: F401 (for type hints)
import plotly.express as px
import streamlit as st


# ---------------------- Utilities ---------------------- #

def clean_column_names(cols):
    """Rough equivalent of MATLAB's makeValidName; matches training script."""
    cleaned = []
    used = set()
    for c in cols:
        if not isinstance(c, str):
            c = str(c)
        name = "".join(ch if ch.isalnum() else "_" for ch in c)
        if not name:
            name = "Var"
        if not name[0].isalpha():
            name = "Var_" + name
        base = name
        k = 1
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        cleaned.append(name)
    return cleaned


@st.cache_resource
def load_model_and_artifacts(res_dir: str = "results"):
    """Load scaler, polynomial transformer, mask, base names and DeepGA–PSO model."""
    artifacts_path = os.path.join(res_dir, "artifacts_xgb.pkl")
    model_path = os.path.join(res_dir, "xgb_model.pkl")

    if not os.path.exists(artifacts_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Artifacts or DeepGA–PSO model not found in '{res_dir}'. "
            f"Please run train_xgb_for_gui.py first."
        )

    artifacts = joblib.load(artifacts_path)
    scaler = artifacts["scaler"]
    poly = artifacts["poly"]
    sel_mask = artifacts["sel_mask"]
    base_names = list(artifacts["base_names"])

    model = joblib.load(model_path)  # XGBRegressor

    return scaler, poly, sel_mask, base_names, model


@st.cache_data
def load_dataset(data_path: str, sheet_name: str, base_names_from_artifacts):
    """
    Load DataSet.xlsx for plotting only.
    Column cleaning matches training script as closely as possible.
    """
    if not os.path.exists(data_path):
        return None, None, None

    _, ext = os.path.splitext(data_path)
    if ext.lower() == ".csv":
        df_raw = pd.read_csv(data_path)
    else:
        df_raw = pd.read_excel(data_path, sheet_name=sheet_name)

    var_names = clean_column_names(df_raw.columns)
    df_raw.columns = var_names

    lower = [c.lower() for c in var_names]
    idx_cs = next((i for i, c in enumerate(lower) if "compressive" in c),
                  len(var_names) - 1)
    target_name = var_names[idx_cs]
    var_names[idx_cs] = "CompressiveStrength"
    df_raw.columns = var_names
    target_name = "CompressiveStrength"

    # Try to align base names with artifacts
    base_names_local = [c for c in df_raw.columns if c != target_name]
    # Ensure ordering matches artifacts if possible
    base_names_final = [
        b for b in base_names_from_artifacts if b in base_names_local
    ]

    df_num = df_raw.apply(pd.to_numeric, errors="coerce")
    return df_num, base_names_final, target_name


def predict_strength(values, scaler, poly, sel_mask, model):
    """Transform raw input vector and return predicted strength."""
    x0 = np.array(values, dtype=float).reshape(1, -1)
    Xz = scaler.transform(x0)
    Xpoly = poly.transform(Xz)
    Xsel = Xpoly[:, sel_mask]
    ypred = float(model.predict(Xsel)[0])
    return ypred


def compute_poly_importance_per_base(poly, sel_mask, base_names, model):
    """
    Aggregate XGBoost feature_importances_ from polynomial space
    down to original base variables.
    """
    poly_names = poly.get_feature_names_out(base_names)
    sel_names = poly_names[sel_mask]
    importances = model.feature_importances_

    # Per-poly-feature importance
    feat_imp_df = pd.DataFrame({
        "PolyFeature": sel_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    # Aggregate to base variables
    agg = {b: 0.0 for b in base_names}
    for feat, imp in zip(sel_names, importances):
        # "Precursor", "Precursor^2", "Precursor FA", ...
        cleaned = feat.replace("^2", "")
        parts = cleaned.split(" ")
        vars_in_feat = [p for p in parts if p in base_names]
        if not vars_in_feat:
            continue
        share = imp / len(vars_in_feat)
        for v in vars_in_feat:
            agg[v] += share

    agg_df = pd.DataFrame({
        "Variable": list(agg.keys()),
        "Importance": list(agg.values()),
    }).sort_values("Importance", ascending=False)

    return feat_imp_df, agg_df


# ---------------------- Streamlit App ---------------------- #

def main():
    warnings.filterwarnings("ignore")
    st.set_page_config(
        page_title="Geopolymer Strength Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧱 Geopolymer Concrete Strength Studio")
        st.markdown(
        """
        <div style="
            display: inline-block;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(49, 51, 63, 0.18);
            background: rgba(240, 242, 246, 0.65);
            font-size: 0.85rem;
        ">
          <b>Developed by:</b> Neha Sharma &nbsp;|&nbsp;
          Department of Civil Engineering, Chandigarh University, Mohali (Punjab), India
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        This app uses a trained **DeepGA–PSO** model to predict the
        **compressive strength** of geopolymer concrete mixes.

        - Inputs are z-scored, expanded to **2nd-order polynomial features**,  
          and then passed to the trained DeepGA–PSO model.
        - The app also provides **interactive plots** to explore the dataset
          and model behaviour.
        """
    )
        # ---- Developer / Affiliation card (sidebar) ----
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="
            padding: 0.9rem 0.9rem;
            border-radius: 14px;
            border: 1px solid rgba(49, 51, 63, 0.18);
            background: rgba(240, 242, 246, 0.85);
        ">
          <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.2px;">
            👩‍💻 Developed by
          </div>

          <div style="font-size: 1.05rem; font-weight: 800; margin-top: 0.35rem;">
            Neha Sharma
          </div>

          <div style="font-size: 0.82rem; line-height: 1.35; margin-top: 0.35rem; opacity: 0.9;">
            Department of Civil Engineering, Chandigarh University,<br/>
            Mohali, Punjab 140413, India
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------- Load model & data ------------- #
    try:
        scaler, poly, sel_mask, base_names, model = load_model_and_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return

    data_path = "DataSet.xlsx"
    sheet_name = "Sheet1"
    df, base_names_in_data, target_name = load_dataset(
        data_path, sheet_name, base_names
    )

    if df is not None and base_names_in_data:
        X_df = df[base_names_in_data]
        y_series = df[target_name]
        stats = X_df.describe()
    else:
        X_df = None
        y_series = None
        stats = None
        st.warning(
            "Could not load `DataSet.xlsx` for dataset plots. "
            "Prediction still works with the trained model."
        )

    # ------------- Sidebar info ------------- #
    st.sidebar.header("About this app")
    st.sidebar.markdown(
        """
        **Steps:**

        1. Go to **Prediction** tab  
           → enter mix-design variables → click *Predict*  
        2. Use **What-if analysis** to see how one variable changes strength.  
        3. Explore **Model insights** and **Dataset overview** for more details.

        **Files used:**

        - `results/artifacts_DeepGA–PSO.pkl`  
        - `results/DeepGA–PSO_model.pkl`  
        - `DataSet.xlsx`
        """
    )

    # ------------- Tabs ------------- #
    tab_pred, tab_whatif, tab_insights, tab_data = st.tabs(
        ["🔮 Prediction", "🔁 What-if analysis", "🧠 Model insights", "📊 Dataset overview"]
    )

    # =====================================================
    # Tab 1: Prediction
    # =====================================================
    with tab_pred:
        st.subheader("Single-mix strength prediction")

        st.markdown(
            "Enter the **original input variables** used in the training data. "
            "Defaults are set to the dataset mean (where available)."
        )

        input_values = []
        n_feat = len(base_names)

        # Arrange inputs in 3 columns
        n_cols = 3 if n_feat >= 3 else max(1, n_feat)
        cols = st.columns(n_cols)

        for i, name in enumerate(base_names):
            col = cols[i % n_cols]
            if stats is not None and name in stats.columns:
                mean_val = float(stats.loc["mean", name])
                min_val = float(stats.loc["min", name])
                max_val = float(stats.loc["max", name])
            else:
                mean_val = 0.0
                min_val = -1000.0
                max_val = 1000.0

            val = col.number_input(
                label=name,
                value=mean_val,
                min_value=min_val,
                max_value=max_val,
                step=(max_val - min_val) / 200 if max_val > min_val else 1.0,
                key=f"inp_{name}",
            )
            input_values.append(val)

        st.markdown("---")
        col_left, col_right = st.columns([1, 2])

        with col_left:
            if st.button("🚀 Predict strength (MPa)"):
                ypred = predict_strength(input_values, scaler, poly, sel_mask, model)
                st.session_state["last_pred"] = ypred
                st.success(f"Predicted compressive strength: **{ypred:.2f} MPa**")

        with col_right:
            if "last_pred" in st.session_state and y_series is not None:
                ypred = st.session_state["last_pred"]
                percentile = float((y_series < ypred).mean() * 100.0)
                st.metric(
                    "Percentile within dataset",
                    f"{percentile:.1f}th percentile",
                )

                # Histogram with prediction marker
                fig_hist = px.histogram(
                    y_series,
                    nbins=30,
                    labels={target_name: "Compressive Strength (MPa)"},
                    title="Distribution of compressive strength",
                )
                fig_hist.add_vline(
                    x=ypred,
                    line_color="red",
                    line_width=3,
                    annotation_text=f"Prediction: {ypred:.1f}",
                    annotation_position="top right",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            elif y_series is None:
                st.info("Prediction percentile/histogram requires DataSet.xlsx.")

    # =====================================================
    # Tab 2: What-if analysis
    # =====================================================
    with tab_whatif:
        st.subheader("What-if analysis for a single variable")

        if X_df is None:
            st.info("What-if analysis requires DataSet.xlsx to be available.")
        else:
            st.markdown(
                """
                Choose one variable to vary across its range while
                holding all others at their **dataset mean**.  
                The plot shows how the predicted strength changes.
                """
            )

            var = st.selectbox("Variable to vary:", base_names, index=0)

            base_means = X_df.mean()
            v_min = float(X_df[var].min())
            v_max = float(X_df[var].max())

            st.write(f"Range for **{var}**: {v_min:.3f} → {v_max:.3f}")

            n_points = st.slider("Number of points in sweep:", 10, 200, 80, step=10)

            grid = np.linspace(v_min, v_max, n_points)
            X_template = base_means.values.reshape(1, -1)
            preds = []

            for val in grid:
                x_vals = X_template.copy()
                idx = base_names.index(var)
                x_vals[0, idx] = val
                y_hat = predict_strength(x_vals[0], scaler, poly, sel_mask, model)
                preds.append(y_hat)

            df_sweep = pd.DataFrame({var: grid, "PredStrength": preds})
            fig_line = px.line(
                df_sweep,
                x=var,
                y="PredStrength",
                labels={"PredStrength": "Predicted Strength (MPa)"},
                title=f"Effect of {var} on predicted compressive strength",
            )
            st.plotly_chart(fig_line, use_container_width=True)

            st.caption(
                "All non-selected variables are fixed at their dataset mean values."
            )

    # =====================================================
    # Tab 3: Model insights
    # =====================================================
    with tab_insights:
        st.subheader("DeepGA–PSO model insights")

        st.markdown(
            """
            The model operates in **polynomial feature space** (original inputs,
            squares, and pairwise products). Below you can inspect:

            1. Importances of individual polynomial features.  
            2. Importances aggregated back to original input variables.
            """
        )

        feat_imp_df, agg_df = compute_poly_importance_per_base(
            poly, sel_mask, base_names, model
        )

        col_a, col_b = st.columns(2)

        with col_a:
            top_k = st.slider("Top polynomial features to display:", 5, 50, 20)
            st.markdown("**Top polynomial features by importance**")
            fig_poly = px.bar(
                feat_imp_df.head(top_k)[::-1],
                x="Importance",
                y="PolyFeature",
                orientation="h",
                title=f"Top {top_k} polynomial features",
            )
            st.plotly_chart(fig_poly, use_container_width=True)

        with col_b:
            st.markdown("**Aggregated importance per original variable**")
            fig_agg = px.bar(
                agg_df[::-1],
                x="Importance",
                y="Variable",
                orientation="h",
                title="Aggregated importances (sum of related poly features)",
            )
            st.plotly_chart(fig_agg, use_container_width=True)

        st.markdown("### Raw importance tables")
        st.expander("Show aggregated importances table").write(agg_df)
        st.expander("Show polynomial features importance table").write(feat_imp_df)

    # =====================================================
    # Tab 4: Dataset overview
    # =====================================================
    with tab_data:
        st.subheader("Dataset overview")

        if df is None:
            st.info("Dataset overview requires DataSet.xlsx to be available.")
        else:
            st.markdown(
                """
                Below you can inspect the raw dataset used to train the model,
                its basic statistics, and interactive scatter plots.
                """
            )

            st.markdown("#### First rows of the dataset")
            st.dataframe(df.head())

            st.markdown("#### Summary statistics (numeric columns)")
            st.dataframe(df.describe().T)

            col1, col2 = st.columns(2)

            with col1:
                fig_hist = px.histogram(
                    df,
                    x=target_name,
                    nbins=30,
                    title="Compressive strength distribution",
                    labels={target_name: "Compressive Strength (MPa)"},
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                x_var = st.selectbox(
                    "X-axis variable for scatter plot:",
                    base_names,
                    index=0,
                )
                fig_scatter = px.scatter(
                    df,
                    x=x_var,
                    y=target_name,
                    title=f"{target_name} vs {x_var}",
                    labels={target_name: "Compressive Strength (MPa)"},
                    trendline="ols",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("#### Correlation heatmap (Pearson)")
            corr = df[base_names + [target_name]].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation matrix",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()


