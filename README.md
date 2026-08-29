# Churn Predictor — AI-driven retention recommendations

**Pipeline:** XGBoost churn model → SHAP explainability → Gemini-generated,
customer-specific retention recommendations → tracked with MLflow →
instrumented with OpenTelemetry → served through a Streamlit UI.

Dataset: [Kaggle — Bank Customer Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
(10,000 customers, `Churn_Modelling.csv` already included in `data/`).

---

## What makes the recommendation step "AI" and not a lookup table

The model (XGBoost) predicts *who* is likely to churn. SHAP explains *why*
(which features drove that specific customer's score). Those SHAP drivers —
not a hardcoded dictionary — get sent to Gemini, which reasons over that
customer's specific combination of factors and writes a tailored
recommendation live, on demand, from the UI. Nothing about the wording is
pre-baked.

---

## 1. Setup — two ways to run this

### Option A: Docker (recommended if you have Docker Desktop)

```bash
cp .env.example .env
# edit .env, add your real GEMINI_API_KEY

docker compose up --build
```

That's it — this builds the app image, trains the model on first run
(skips training on later runs if a model already exists), and starts
Tempo + Grafana alongside it. No venv, no manual `pip install`.

- App: http://localhost:8501
- Grafana (traces): http://localhost:3000 → Explore → Tempo datasource
- `mlflow.db`, `models/`, `artifacts/` are bind-mounted, so they persist
  on your machine — you can run `mlflow ui --backend-store-uri sqlite:///mlflow.db`
  directly on your host (outside Docker) and it'll see the same data.

To retrain from scratch: delete `models/churn_model_bundle.pkl` and
re-run `docker compose up --build`.

### Option B: Local Python (venv)

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a file named `.env` in this folder (copy `.env.example`) containing:
```
GEMINI_API_KEY=your-key-here
```
Get a key at https://aistudio.google.com/apikey. `.env` is already listed in
`.gitignore` so it won't get committed if you push this to GitHub.

## 2. Train the model (creates MLflow tracking DB + registers the model)

Skip this step if you used Docker (Option A above) — the container's
entrypoint already trains automatically on first run.

```bash
python -m src.train
```

You'll see OpenTelemetry spans printed to the console for each pipeline
stage (data load, train, evaluate, explain, register), and a summary of
metrics (expect ROC-AUC ≈ 0.87 on this dataset).

## 3. View experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 — you'll see the run, its params/metrics, the
registered model version, and logged artifacts (SHAP importance plot,
confusion matrix).

## 4. Launch the app

Skip this if you used Docker (Option A) — it's already running at
http://localhost:8501.

```bash
streamlit run app.py
```

Opens at http://localhost:8501 with three tabs:
- **At-Risk Customers** — filterable list, drill into any customer to see
  their SHAP drivers and generate a live Gemini recommendation
- **Model Observability** — MLflow run history + global feature importance
- **About this pipeline** — the architecture, in plain language

## 5. Visualize traces on a UI

Two options, in order of setup effort:

### Option A — Jaeger (simplest, one container, built-in UI)
```bash
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
export OTEL_EXPORTER=otlp
pip install opentelemetry-exporter-otlp
python -m src.train
```
Open http://localhost:16686, select service `churn-training` (or `churn-ui`),
click "Find Traces" — you'll see every run's trace as a waterfall of spans.
Good enough to demo; nothing else to configure.

### Option B — Grafana + Tempo (open source, more production-representative)
Grafana itself only renders dashboards — it doesn't store traces. Pair it
with **Tempo** (also open source, Apache 2.0), which is the actual trace
backend. This mirrors how many real production stacks are built (Grafana
as the shared UI across metrics/logs/traces).

```bash
cd observability
docker compose up -d
```
This starts Tempo (receiving OTLP on port 4317) and Grafana (with the Tempo
datasource auto-provisioned). Then, back in the project root:
```bash
export OTEL_EXPORTER=otlp
pip install opentelemetry-exporter-otlp
python -m src.train
```
Open http://localhost:3000 → **Explore** → select the **Tempo** datasource
→ search by `service.name = churn-training`. You'll see every request's
trace, and clicking one shows the full span timeline (data load → train →
evaluate → explain → register), same idea as Jaeger but on Grafana's UI,
which is what you'd use if metrics/logs are also going to Grafana in a
real deployment.

**Config files** for this are in `observability/`: `docker-compose.yml`,
`tempo.yaml` (trace storage config), `grafana-datasources.yaml`
(auto-provisions the Tempo datasource so you don't have to click through
Grafana's UI to add it manually).

### Option C — skip self-hosting entirely
[Grafana Cloud](https://grafana.com/products/cloud/) has a free tier with
managed Tempo — you'd point `OTEL_EXPORTER` at their OTLP endpoint instead
of `localhost:4317`, no Docker needed. Worth it if you don't want to run
infrastructure just for a screening project.

## 6. Get a public link for submission

Push this folder to a GitHub repo, then deploy free on
[Streamlit Community Cloud](https://streamlit.io/cloud) — point it at
`app.py`, add `GEMINI_API_KEY` under app secrets. You'll get a public URL
you can submit directly.

---

## Project structure

```
churn_project/
  data/Churn_Modelling.csv       # Kaggle dataset
  src/
    data_prep.py                 # load + encode + split
    tracing_setup.py             # OpenTelemetry tracer (console or OTLP)
    train.py                     # train + MLflow tracking + SHAP + model registry
    llm_recommender.py           # Gemini-based recommendation generator
  observability/
    docker-compose.yml           # Tempo + Grafana only (if not using the app's Docker option)
    tempo.yaml                   # Tempo trace-storage config
    grafana-datasources.yaml     # auto-provisions Grafana's Tempo datasource
  models/churn_model_bundle.pkl  # trained model + encoders (created by train.py)
  artifacts/                     # SHAP + confusion matrix plots (created by train.py)
  app.py                         # Streamlit UI
  Dockerfile                     # containerizes the app itself
  entrypoint.sh                  # trains (if needed) then launches Streamlit
  docker-compose.yml             # full stack: app + Tempo + Grafana together
  mlflow.db                      # MLflow tracking store (created by train.py)
```
