#!/bin/sh
set -e

if [ ! -f models/churn_model_bundle.pkl ]; then
  echo "No trained model found -- training now (this also sets up MLflow tracking)..."
  python -m src.train
else
  echo "Found existing trained model, skipping training. Delete models/churn_model_bundle.pkl to retrain."
fi

exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501
