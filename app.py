"""
Churn Predictor — consumable UI.

Run:
    export GEMINI_API_KEY="your-key-here"
    streamlit run app.py

Shows the at-risk customer list, lets you drill into any customer to see
their SHAP-driven churn factors, and generates a live, LLM-written retention
recommendation on demand (so it's not pre-baked -- it calls Gemini when you
click the button).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
import mlflow
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from a .env file in this folder, if present

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_prep import load_and_prepare
from src.tracing_setup import get_tracer

tracer = get_tracer("churn-ui")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Churn_Modelling.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model_bundle.pkl")

st.set_page_config(page_title="Churn Predictor", layout="wide")


@st.cache_resource
def load_everything():
    with tracer.start_as_current_span("load_model_and_data"):
        bundle = joblib.load(MODEL_PATH)
        model, encoders, feature_cols = bundle["model"], bundle["encoders"], bundle["feature_cols"]
        X_train, X_test, y_train, y_test, feature_cols, encoders, raw_test = load_and_prepare(DATA_PATH)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, y_proba, feature_cols, raw_test, shap_values


@st.cache_data
def get_mlflow_history():
    try:
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")
        runs = mlflow.search_runs(experiment_names=["churn-predictor"])
        cols = ["start_time", "metrics.roc_auc", "metrics.precision", "metrics.recall", "metrics.f1"]
        return runs[[c for c in cols if c in runs.columns]].sort_values("start_time")
    except Exception:
        return pd.DataFrame()


model, X_test, y_test, y_proba, feature_cols, raw_test, shap_values = load_everything()

results = raw_test.copy().reset_index(drop=True)
results["churn_probability"] = y_proba
results["actual_churn"] = y_test.values

st.title("Churn Predictor")
st.caption("XGBoost model + SHAP explainability + Gemini-generated retention recommendations, tracked with MLflow.")

tab1, tab2, tab3 = st.tabs(["At-Risk Customers", "Model Observability", "About this pipeline"])

# --------------------------------------------------------------------------
with tab1:
    threshold = st.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.01)
    filtered = results[results["churn_probability"] >= threshold].sort_values(
        "churn_probability", ascending=False
    )
    st.write(f"**{len(filtered)} customers** above this threshold (out of {len(results)} in the test set)")

    st.dataframe(
        filtered[["CustomerId" if "CustomerId" in filtered.columns else filtered.columns[0],
                  "Age", "Geography", "Balance", "NumOfProducts", "IsActiveMember", "churn_probability"]]
        if "CustomerId" in filtered.columns else filtered.head(50),
        use_container_width=True,
        height=300,
    )

    st.divider()
    st.subheader("Drill into a customer")
    row_options = filtered.index.tolist()
    if row_options:
        selected_idx = st.selectbox(
            "Select a customer (by row index)",
            row_options,
            format_func=lambda i: f"Row {i} — {results.loc[i, 'churn_probability']:.1%} churn probability",
        )

        customer_row = X_test.loc[selected_idx] if selected_idx in X_test.index else X_test.iloc[list(results.index).index(selected_idx)]
        raw_row = results.loc[selected_idx]
        row_position = list(results.index).index(selected_idx)
        row_shap = shap_values[row_position]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Customer profile**")
            profile_dict = {c: raw_row[c] for c in feature_cols if c in raw_row}
            st.json(profile_dict)

        with col2:
            st.markdown("**Top churn drivers (SHAP)**")
            driver_df = pd.DataFrame({
                "feature": feature_cols,
                "shap_value": row_shap,
            }).reindex(np.abs(row_shap).argsort()[::-1])
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in driver_df["shap_value"].head(6)]
            ax.barh(driver_df["feature"].head(6)[::-1], driver_df["shap_value"].head(6)[::-1], color=colors[::-1])
            ax.set_title("Red = pushes toward churn, Green = pushes toward staying")
            fig.tight_layout()
            st.pyplot(fig)

        st.divider()
        if st.button("Generate AI retention recommendation", type="primary"):
            if not os.environ.get("GEMINI_API_KEY"):
                st.error("Set GEMINI_API_KEY in your environment before generating recommendations.")
            else:
                from src.llm_recommender import generate_recommendation
                top_drivers = list(driver_df.head(3)[["feature", "shap_value"]].itertuples(index=False, name=None))
                with st.spinner("Asking Gemini..."):
                    rec = generate_recommendation(profile_dict, top_drivers, raw_row["churn_probability"])
                st.markdown(f"**Priority:** {rec.get('priority', 'N/A')}")
                st.markdown(f"**Why this customer is at risk:** {rec.get('risk_summary', '')}")
                st.markdown(f"**Recommended action:** {rec.get('recommended_action', '')}")
    else:
        st.info("No customers above this threshold.")

# --------------------------------------------------------------------------
with tab2:
    st.subheader("Model performance (from MLflow)")
    history = get_mlflow_history()
    if not history.empty:
        st.line_chart(history.set_index("start_time")[
            [c for c in ["metrics.roc_auc", "metrics.precision", "metrics.recall", "metrics.f1"] if c in history.columns]
        ])
        st.dataframe(history, use_container_width=True)
        st.caption("Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` for the full MLflow dashboard.")
    else:
        st.warning("No MLflow runs found yet. Run `python -m src.train` first.")

    st.subheader("Global churn drivers")
    st.image(os.path.join(BASE_DIR, "artifacts", "shap_importance.png"))

# --------------------------------------------------------------------------
with tab3:
    st.markdown("""
    **Pipeline:**
    1. XGBoost classifier trained on the Kaggle bank churn dataset
    2. SHAP explains *why* each customer is flagged
    3. Gemini turns the SHAP drivers into a specific, written retention plan (generated live, not templated)
    4. MLflow tracks every training run's params/metrics and registers the model
    5. OpenTelemetry traces each pipeline stage (data load, train, predict, LLM call) — see console output when running `python -m src.train`
    """)
