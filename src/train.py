"""
Training pipeline: XGBoost churn classifier, instrumented with
  - OpenTelemetry spans around each pipeline stage (observability)
  - MLflow tracking + model registry (experiment tracking / MLOps)

Run:
    python -m src.train

Then view results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    -> open http://localhost:5000
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import load_and_prepare
from src.tracing_setup import get_tracer

tracer = get_tracer("churn-training")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Churn_Modelling.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
EXPERIMENT_NAME = "churn-predictor"
REGISTERED_MODEL_NAME = "churn_xgb"

# local sqlite backend -> gives us the MLflow Model Registry without needing
# a running tracking server; app.py points at the same URI to load the model.
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlflow.db')}")
mlflow.set_experiment(EXPERIMENT_NAME)


def train():
    with tracer.start_as_current_span("full_training_pipeline"):

        with tracer.start_as_current_span("load_and_prepare_data"):
            X_train, X_test, y_train, y_test, feature_cols, encoders, raw_test = load_and_prepare(DATA_PATH)

        with mlflow.start_run(run_name="xgb_churn_v1") as run:
            params = dict(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=42,
            )
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            params["scale_pos_weight"] = scale_pos_weight
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))

            with tracer.start_as_current_span("train_model") as span:
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)
                span.set_attribute("n_estimators", params["n_estimators"])

            with tracer.start_as_current_span("evaluate_model") as span:
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)
                metrics = {
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                }
                for k, v in metrics.items():
                    span.set_attribute(k, v)
                mlflow.log_metrics(metrics)
                print("Metrics:", metrics)

            with tracer.start_as_current_span("explain_with_shap"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                importance = pd.DataFrame({
                    "feature": feature_cols,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                }).sort_values("mean_abs_shap", ascending=False)

                os.makedirs(ARTIFACT_DIR, exist_ok=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1])
                ax.set_title("Global churn drivers (mean |SHAP|)")
                fig.tight_layout()
                shap_plot_path = os.path.join(ARTIFACT_DIR, "shap_importance.png")
                fig.savefig(shap_plot_path)
                plt.close(fig)
                mlflow.log_artifact(shap_plot_path)

            with tracer.start_as_current_span("log_confusion_matrix"):
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, cm[i, j], ha="center", va="center")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["Stayed", "Churned"]); ax.set_yticklabels(["Stayed", "Churned"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                fig.tight_layout()
                cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
                fig.savefig(cm_path)
                plt.close(fig)
                mlflow.log_artifact(cm_path)

            with tracer.start_as_current_span("register_model"):
                mlflow.xgboost.log_model(
                    model, "model", registered_model_name=REGISTERED_MODEL_NAME,
                )

            # also save a local copy for the Streamlit app to load quickly,
            # alongside encoders/feature list it needs for preprocessing
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(
                {"model": model, "encoders": encoders, "feature_cols": feature_cols},
                os.path.join(MODEL_DIR, "churn_model_bundle.pkl"),
            )
            print(f"\nRun ID: {run.info.run_id}")
            print(f"Registered model: {REGISTERED_MODEL_NAME}")
            return model, feature_cols, encoders, metrics, importance


if __name__ == "__main__":
    train()
