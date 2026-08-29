"""
LLM-driven retention recommendations.

Instead of a hardcoded lookup table, this sends each at-risk customer's
profile + their SHAP-derived churn drivers to Gemini and asks it to reason
about *that specific customer* and produce a tailored recommendation.

Setup:
    Create a file named .env in the project root (same folder as app.py)
    containing:
        GEMINI_API_KEY=your-key-here
    (see .env.example for the template). It's loaded automatically below.

Get a key at: https://aistudio.google.com/apikey
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from src.tracing_setup import get_tracer

# looks for a .env file in the project root and loads GEMINI_API_KEY from it
load_dotenv()

tracer = get_tracer("churn-llm")

MODEL_NAME = "gemini-3.6-flash"  # fast + cheap, fine for this use case

_configured = False


def _configure():
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your-key'"
            )
        genai.configure(api_key=api_key)
        _configured = True


PROMPT_TEMPLATE = """You are a retention strategist at a bank. A machine learning
model has flagged the customer below as high risk of churning, with the reasons
(SHAP feature attributions) listed. Write a short, specific retention plan.

Customer profile:
{profile}

Top factors driving this customer's churn risk (feature: SHAP contribution,
positive = pushes toward churn):
{drivers}

Churn probability from the model: {probability:.1%}

Respond ONLY with a JSON object, no markdown fences, no preamble, with fields:
  "risk_summary": one sentence on WHY this specific customer is at risk
  "recommended_action": one concrete, specific retention action for this customer
  "priority": "High", "Medium", or "Low"
"""


def generate_recommendation(customer_profile: dict, top_drivers: list, probability: float) -> dict:
    """
    customer_profile: dict of raw feature values, e.g. {"Age": 45, "Balance": 120000, ...}
    top_drivers: list of (feature_name, shap_value) tuples, e.g. [("Age", 0.42), ("NumOfProducts", 0.31)]
    probability: model's predicted churn probability (0-1)
    """
    _configure()

    with tracer.start_as_current_span("llm_generate_recommendation") as span:
        profile_str = "\n".join(f"  {k}: {v}" for k, v in customer_profile.items())
        drivers_str = "\n".join(f"  {name}: {val:+.3f}" for name, val in top_drivers)
        prompt = PROMPT_TEMPLATE.format(
            profile=profile_str, drivers=drivers_str, probability=probability
        )

        span.set_attribute("model", MODEL_NAME)
        span.set_attribute("probability", probability)

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Gemini sometimes wraps JSON in ```json fences despite instructions -- strip them
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            span.set_attribute("parse_error", True)
            result = {
                "risk_summary": "Could not parse model response.",
                "recommended_action": raw_text[:200],
                "priority": "Medium",
            }
        return result


if __name__ == "__main__":
    # quick manual test -- requires GEMINI_API_KEY to be set
    sample_profile = {
        "Age": 51, "Geography": "Germany", "Gender": "Female",
        "Balance": 132000.0, "NumOfProducts": 1, "IsActiveMember": 0,
        "CreditScore": 610, "Tenure": 2,
    }
    sample_drivers = [("Age", 0.42), ("NumOfProducts", 0.31), ("IsActiveMember", 0.18)]
    print(generate_recommendation(sample_profile, sample_drivers, 0.87))
