# Obzerra Fraud Detection System

Obzerra is a local-first analytics workbench built for Philippine insurance fraud teams. The application combines rule-based heuristics with machine-learning signals to help analysts prioritise suspicious claims in real time.

## Key capabilities

- **Interactive dashboard** – monitor overall claim volumes, risk mix, and fraud patterns at a glance.
- **Single-claim triage** – capture claim information through a guided form and receive an immediate risk assessment.
- **Batch analysis** – upload CSV files, map columns to the internal schema, and process dozens of claims simultaneously.
- **Explainable outputs** – see which rules and features contributed to each score and download enriched results for offline review.
- **On-device workflow** – all processing happens inside the Streamlit session; no external APIs or services are required.

## Getting started

```bash
# install dependencies
uv sync

# launch the Streamlit interface
uv run streamlit run app.py
```

The default configuration runs locally. When the app is open, use the **Batch upload** tab to analyse CSV files or the **Single claim** tab for ad-hoc assessments.

## Architecture overview

| Layer | Description |
| --- | --- |
| **Interface** | Streamlit UI with custom styling and Plotly charts. |
| **Data processing** | `DataProcessor` standardises column names, performs validation, and engineers statistical features. |
| **Rule engine** | `FraudEngine` applies weighted rules such as z-score outliers, Benford analysis, and unusual-hour flags. |
| **Machine learning** | `MLModelManager` trains Logistic Regression and Random Forest models (with SMOTE balancing) once enough labelled-like data is available. |
| **Explainability** | `ExplanationEngine` translates rule/ML signals into plain-language insights and recommended actions. |
| **Session storage** | `SessionManager` persists recent runs and metrics in Streamlit session state. |

## Dataset expectations

| Column | Purpose |
| --- | --- |
| `claim_id` | Unique identifier for each claim. |
| `total_claim_amount` | Amount being claimed (₱). |
| `incident_hour_of_the_day` | Hour when the incident occurred (0-23). |
| Additional optional fields | Age, incident type/severity, number of witnesses, property damage indicator, police report flag. |

A downloadable CSV template is available directly inside the app to help teams align their exports.

## Contributing

1. Fork the repository and create a feature branch.
2. Make changes and ensure the Streamlit app still launches without errors.
3. Run `uv run streamlit run app.py` locally to verify UI updates.
4. Submit a pull request describing the improvement or fix.

---

Made with 🛡️ for investigative analysts who need clear, defensible fraud decisions.
