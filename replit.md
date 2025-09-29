# Architecture notes

This document summarises the main decisions behind the Obzerra prototype so future contributors have quick context.

## Front-end

- **Framework:** Streamlit running in wide-layout mode with a collapsed sidebar.
- **Styling:** Custom CSS for typography, card layouts, and gradients that align with the Obzerra visual identity.
- **Visualisations:** Plotly Express/Graph Objects are used for donut, line, and bar charts on the dashboard and history views.
- **UX touches:** Column mapping for batch uploads, accordions for template guidance, card-based results with contextual icons, and responsive grid layouts.

## Fraud analytics engine

- **Data processing:** The `DataProcessor` class renames incoming columns, enforces required fields, handles duplicates, and engineers statistical features (log transforms, Benford scores, hour buckets, etc.).
- **Rules engine:** `FraudEngine` applies weighted heuristics covering z-score outliers, unusual hours, round amounts, high-value thresholds, demographic risk, and witness checks.
- **Machine learning:** `MLModelManager` orchestrates a Logistic Regression + Random Forest ensemble. SMOTE balances training data once enough claims have been analysed. Metrics and feature importances feed the sidebar.
- **Explainability:** `ExplanationEngine` converts triggered rules and ML probabilities into analyst-friendly summaries and recommended actions.
- **Session management:** `SessionManager` persists run history, totals, and derived metrics directly in Streamlit session state for a local-first experience.

## Operational considerations

- Streamlit session memory is used instead of an external database; the prototype is intended for offline demos.
- Training data is inferred from rule scores until labelled outcomes are available.
- CSV exports combine original columns with model outputs to support investigations in downstream tools.

For deeper changes, scan the `utils/` package to understand how each component collaborates with the Streamlit interface in `app.py`.
