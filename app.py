"""Streamlit interface for the Obzerra fraud detection experience."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from secrets import randbelow
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_processor import DataProcessor
from utils.explanations import ExplanationEngine
from utils.fraud_engine import FraudEngine
from utils.ml_models import MLModelManager
from utils.session_manager import SessionManager


PAGE_CONFIG = {
    "page_title": "Obzerra - Fraud Detection System",
    "page_icon": "🛡️",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
}

CUSTOM_CSS = """
<style>
    .main-header {
        background: #0f172a;
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.4);
    }
    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
    }
    .brand-logo {
        font-size: 2.2rem;
        font-weight: 700;
        color: #6366f1;
        margin: 0;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }
    .kpi-card {
        background: linear-gradient(160deg, #111827 0%, #1e293b 100%);
        padding: 1.6rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.25);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.35);
        height: 100%;
    }
    .kpi-card h3 {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .kpi-card h2 {
        font-size: 2.4rem;
        margin: 0;
    }
    .kpi-card p {
        color: #64748b;
        margin: 0;
        font-size: 0.85rem;
    }
    .claim-card {
        background: #111827;
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.45);
    }
    .claim-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .claim-id {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .fraud-indicator {
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .fraud-yes {
        background: rgba(252, 165, 165, 0.85);
        color: #7f1d1d;
    }
    .fraud-no {
        background: rgba(134, 239, 172, 0.85);
        color: #14532d;
    }
    .risk-score {
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        padding: 0.75rem 1rem;
        border-radius: 14px;
        min-width: 110px;
        line-height: 1.2;
    }
    .risk-score span {
        display: block;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    .risk-score-low {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .risk-score-medium {
        background: rgba(234, 179, 8, 0.12);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.35);
    }
    .risk-score-high {
        background: rgba(248, 113, 113, 0.12);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.35);
        animation: pulse 2.4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.35); }
        50% { box-shadow: 0 0 0 10px rgba(248, 113, 113, 0); }
    }
    .claim-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1rem;
    }
    .detail-section {
        padding: 1rem;
        background: #0f172a;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        min-height: 150px;
    }
    .detail-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.6rem;
    }
    .detail-content {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .rule-badge {
        background: rgba(79, 70, 229, 0.15);
        color: #c7d2fe;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin: 0.15rem;
        display: inline-block;
    }
</style>
"""

RULE_LIBRARY: Dict[str, Tuple[str, str, str]] = {
    "z-score outlier": ("📊", "Amount analysis", "Claim amount is statistically unusual versus peers."),
    "benford anomaly": ("🔢", "Number pattern", "Digit distribution deviates from Benford's Law."),
    "unusual hour": ("🕕", "Time pattern", "Incident occurred at an atypical time of day."),
    "round amount": ("🎯", "Round number", "Amount is a perfectly round figure and may be fabricated."),
    "high amount": ("💵", "High value", "Claim exceeds the usual amount for similar incidents."),
    "young high claim": ("🧑", "Demographic risk", "High-value claim submitted by a young policy holder."),
    "no witnesses high": ("👥", "No witnesses", "Large claim submitted without any supporting witnesses."),
    "duplicate amount": ("🔄", "Duplicate pattern", "Similar claim amounts detected across recent submissions."),
}

RISK_BADGE = {
    "Low": ("🟢", "risk-score-low"),
    "Medium": ("🟡", "risk-score-medium"),
    "High": ("🔴", "risk-score-high"),
}


@dataclass
class KpiMetric:
    """Structure representing the dashboard KPI tiles."""

    title: str
    icon: str
    value: str
    description: str
    accent: str
    status: str


def configure_page() -> None:
    """Apply Streamlit configuration and shared styling."""
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialise_managers() -> Tuple[SessionManager, DataProcessor, FraudEngine, MLModelManager, ExplanationEngine]:
    """Ensure heavy objects persist inside session state."""
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()
        st.session_state.data_processor = DataProcessor()
        st.session_state.fraud_engine = FraudEngine()
        st.session_state.ml_manager = MLModelManager()
        st.session_state.explanation_engine = ExplanationEngine()

    return (
        st.session_state.session_manager,
        st.session_state.data_processor,
        st.session_state.fraud_engine,
        st.session_state.ml_manager,
        st.session_state.explanation_engine,
    )


def render_header() -> None:
    """Render the Obzerra brand banner."""
    st.markdown(
        """
        <div class="main-header">
            <div class="header-content">
                <div>
                    <h1 class="brand-logo">🛡️ Obzerra</h1>
                    <p class="brand-subtitle">Philippine Insurance Fraud Detection System</p>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    Advanced ML-powered fraud detection for local insurers
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_kpi_metrics(total_claims: int, high_risk_pct: float, avg_runtime: float, accuracy: float) -> List[KpiMetric]:
    """Prepare KPI cards with contextual colours."""
    risk_color = "#dc2626" if high_risk_pct > 15 else "#059669" if high_risk_pct < 5 else "#d97706"
    risk_status = "High alert" if high_risk_pct > 15 else "Normal" if high_risk_pct < 5 else "Monitor closely"

    runtime_color = "#059669" if avg_runtime < 2 else "#d97706" if avg_runtime < 5 else "#dc2626"
    runtime_status = "Fast" if avg_runtime < 2 else "On track" if avg_runtime < 5 else "Investigate"

    accuracy_color = "#059669" if accuracy > 0.85 else "#d97706" if accuracy > 0.7 else "#dc2626"
    accuracy_status = "Excellent" if accuracy > 0.85 else "Good" if accuracy > 0.7 else "Training"

    return [
        KpiMetric("Total Claims", "📊", f"{total_claims:,}", "Claims analysed", "#6366f1", ""),
        KpiMetric("High Risk", "🚨", f"{high_risk_pct:.1f}%", risk_status, risk_color, ""),
        KpiMetric("Processing", "⏱️", f"{avg_runtime:.1f}s", runtime_status, runtime_color, ""),
        KpiMetric("ML Accuracy", "🤖", f"{accuracy * 100:.0f}%", accuracy_status, accuracy_color, ""),
    ]


def render_kpi_card(metric: KpiMetric) -> str:
    """Return the HTML markup for a KPI tile."""
    return f"""
        <div class=\"kpi-card\" style=\"border-left: 4px solid {metric.accent};\">
            <div style=\"display:flex; justify-content: space-between; align-items: center;\">
                <div>
                    <h3>{metric.icon} {metric.title}</h3>
                    <h2 style=\"color:{metric.accent};\">{metric.value}</h2>
                    <p>{metric.description}</p>
                    {f'<p style="color:{metric.accent}; font-weight:600;">{metric.status}</p>' if metric.status else ''}
                </div>
                <div style=\"font-size: 3rem; opacity: 0.25; color:{metric.accent};\">{metric.icon}</div>
            </div>
        </div>
    """


def summarise_recent_results(recent_results: Iterable[Dict]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Derive risk counts, top rules, and daily volumes from recent runs."""
    risk_counts: Dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
    rule_counts: Dict[str, int] = {}
    daily_counts: Dict[str, int] = {}

    for run in recent_results:
        results_df = run.get("results")
        timestamp = run.get("timestamp", datetime.now())
        date_key = timestamp.strftime("%Y-%m-%d")
        daily_counts[date_key] = daily_counts.get(date_key, 0) + len(results_df)

        for _, row in results_df.iterrows():
            risk_counts[row.get("risk_band", "Low")] += 1
            rules = [r.strip() for r in str(row.get("triggered_rules", "")).split(",") if r.strip()]
            for rule in rules:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1

    return risk_counts, rule_counts, daily_counts


def render_risk_distribution(risk_counts: Dict[str, int]) -> None:
    """Render donut chart for risk distribution."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=0.6,
                marker_colors=["#059669", "#d97706", "#dc2626"],
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=40, b=40, l=40, r=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=320,
    )
    total_claims = sum(risk_counts.values())
    fig.add_annotation(
        text=f"<b>{total_claims}</b><br>Total Claims",
        x=0.5,
        y=0.5,
        font_size=15,
        showarrow=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_claim_trend(daily_counts: Dict[str, int]) -> None:
    """Render line chart for claim volumes over time."""
    if len(daily_counts) <= 1:
        st.info("📅 More data is needed to display a trend line.")
        return

    dates = sorted(daily_counts.keys())
    counts = [daily_counts[date] for date in dates]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=dates,
                y=counts,
                mode="lines+markers",
                line=dict(color="#6366f1", width=3),
                marker=dict(size=8, color="#6366f1"),
                fill="tonexty",
                fillcolor="rgba(99,102,241,0.12)",
                hovertemplate="<b>%{x}</b><br>Claims: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Claims processed",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=320,
        margin=dict(t=40, b=40, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_top_rules(rule_counts: Dict[str, int]) -> None:
    """Render horizontal bar chart for top triggered rules."""
    if not rule_counts:
        st.success("✅ No fraud indicators triggered in recent analyses.")
        return

    top_rules = sorted(rule_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    rule_names = [name for name, _ in top_rules]
    rule_values = [count for _, count in top_rules]

    fig = go.Figure(
        data=[
            go.Bar(
                y=rule_names,
                x=rule_values,
                orientation="h",
                marker_color=["#dc2626", "#d97706", "#059669", "#6366f1", "#8b5cf6"][: len(rule_names)],
                hovertemplate="<b>%{y}</b><br>Triggered: %{x} times<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Frequency",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=320,
        margin=dict(t=40, b=40, l=60, r=40),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def parse_triggered_rules(rules_text: str) -> List[str]:
    """Split triggered rules string into clean entries."""
    if not rules_text or str(rules_text).lower() == "nan":
        return []
    return [rule.strip() for rule in str(rules_text).split(",") if rule.strip()]


def render_rule_cards(rules_text: str) -> str:
    """Create HTML blocks describing each triggered rule."""
    rules = parse_triggered_rules(rules_text)
    if not rules:
        return '<div style="color:#10b981;font-style:italic;">✅ No suspicious patterns detected.</div>'

    cards = []
    for rule in rules:
        icon, heading, description = RULE_LIBRARY.get(rule.lower(), ("ℹ️", rule.title(), "Further review recommended."))
        cards.append(
            f"""
            <div style=\"background:#0b1220;padding:0.9rem;border-radius:12px;margin:0.35rem 0;border-left:3px solid #6366f1;\">
                <div style=\"display:flex;align-items:center;gap:0.6rem;margin-bottom:0.35rem;\">
                    <span style=\"font-size:1.3rem;\">{icon}</span>
                    <strong style=\"color:#e2e8f0;font-size:0.95rem;\">{heading}</strong>
                </div>
                <div style=\"color:#94a3b8;font-size:0.85rem;line-height:1.4;\">{description}</div>
            </div>
            """
        )
    return "".join(cards)


def render_claim_card(row: pd.Series) -> None:
    """Render a single claim summary card."""
    risk_band = row.get("risk_band", "Low")
    icon, css_class = RISK_BADGE.get(risk_band, ("⚪", "risk-score-low"))
    fraud_prediction = row.get("fraud_prediction", "N") == "Y"
    fraud_class = "fraud-yes" if fraud_prediction else "fraud-no"
    fraud_text = "🚨 High fraud risk" if fraud_prediction else "✅ Legitimate claim"

    st.markdown(
        f"""
        <div class=\"claim-card\">
            <div class=\"claim-header\">
                <div class=\"claim-id\">📋 {row.get('claim_id', 'N/A')}</div>
                <div style=\"display:flex;align-items:center;gap:1rem;\">
                    <div class=\"fraud-indicator {fraud_class}\">{fraud_text}</div>
                    <div class=\"risk-score {css_class}\">
                        {icon}
                        <span style=\"font-size:1.8rem;font-weight:700;\">{int(row.get('risk_score', 0))}</span>
                        <span>{risk_band}</span>
                    </div>
                </div>
            </div>
            <div class=\"claim-details\">
                <div class=\"detail-section\">
                    <div class=\"detail-title\">🤖 AI Assessment</div>
                    <div class=\"detail-content\">
                        <strong>Risk level:</strong> {risk_band}<br>
                        <strong>Primary reason:</strong> {row.get('reason_1', 'N/A')}<br>
                        <strong>Secondary reason:</strong> {row.get('reason_2', 'N/A')}
                    </div>
                </div>
                <div class=\"detail-section\">
                    <div class=\"detail-title\">🔍 Why this result?</div>
                    <div class=\"detail-content\">{render_rule_cards(row.get('triggered_rules', ''))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def random_identifier(prefix: str, digits: int = 6) -> str:
    """Generate a pseudo-random identifier for default form values."""
    return f"{prefix}_{randbelow(10 ** digits):0{digits}d}"


def render_dashboard(session_manager: SessionManager, ml_manager: MLModelManager) -> None:
    """Dashboard tab with KPIs and visualisations."""
    st.header("📊 Dashboard overview")

    metrics = build_kpi_metrics(
        session_manager.get_total_claims(),
        session_manager.get_high_risk_percentage(),
        session_manager.get_avg_runtime(),
        ml_manager.get_model_accuracy(),
    )

    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        col.markdown(render_kpi_card(metric), unsafe_allow_html=True)

    st.divider()

    recent_results = session_manager.get_recent_results()
    if not recent_results:
        st.info("📊 Run an analysis to unlock performance analytics.")
        return

    risk_counts, rule_counts, daily_counts = summarise_recent_results(recent_results)
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.subheader("🍩 Risk distribution")
        render_risk_distribution(risk_counts)

    with chart_col2:
        st.subheader("📈 Claims trend")
        render_claim_trend(daily_counts)

    with chart_col3:
        st.subheader("🔍 Top fraud indicators")
        render_top_rules(rule_counts)


def render_batch_upload(
    data_processor: DataProcessor,
    fraud_engine: FraudEngine,
    ml_manager: MLModelManager,
    explanation_engine: ExplanationEngine,
    session_manager: SessionManager,
) -> None:
    """Batch processing tab."""
    st.header("📂 Batch claims analysis")
    st.subheader("Upload claims data")

    with st.expander("📋 Download sample CSV template"):
        sample_data = {
            "claim_id": ["CLM_001", "CLM_002", "CLM_003"],
            "policy_number": ["POL_123456", "POL_789012", "POL_345678"],
            "total_claim_amount": [25_000, 75_000, 150_000],
            "incident_date": ["2024-01-15", "2024-01-20", "2024-01-25"],
            "incident_hour_of_the_day": [14, 22, 8],
            "incident_type": ["Vehicle Collision", "Property Damage", "Vehicle Theft"],
            "incident_severity": ["Minor Damage", "Major Damage", "Total Loss"],
            "witnesses": [2, 0, 1],
            "property_damage": [1, 1, 0],
            "police_report_available": [1, 1, 0],
            "fraud_reported": [0, 0, 1],
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        st.download_button(
            label="📥 Download CSV template",
            data=sample_df.to_csv(index=False),
            file_name="obzerra_claims_template.csv",
            mime="text/csv",
        )

    with st.expander("ℹ️ Need help with file format?"):
        st.markdown(
            """
            **Required columns**

            - **Claim ID** – unique identifier for each claim.
            - **Claim amount** – monetary value being claimed (₱).
            - **Incident hour** – time of incident (0-23).

            **Optional columns for richer insights**

            - Policy number, incident date, incident type and severity.
            - Age of insured party and number of witnesses.
            - Property damage and police report availability (1 = Yes, 0 = No).

            💡 Your CSV headings do not need to match ours exactly—use the mapping controls below to align them with Obzerra's schema.
            """
        )

    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here or click to browse",
        type=["csv"],
        help="Upload a CSV file containing insurance claims data for fraud analysis.",
    )

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - surface read errors gracefully
        st.error(f"Unable to read file: {exc}")
        return

    st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
    st.subheader("Data preview (first 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Map your columns")
    col1, col2 = st.columns(2)

    with col1:
        claim_id_col = st.selectbox(
            "Claim identifier",
            options=df.columns.tolist(),
            index=0,
            help="Column containing unique claim identifiers.",
        )
        amount_col = st.selectbox(
            "Claim amount",
            options=df.columns.tolist(),
            index=0,
            help="Column containing the monetary amount being claimed.",
        )
        hour_col = st.selectbox(
            "Incident hour (0-23)",
            options=df.columns.tolist(),
            index=0,
            help="Column specifying when the incident occurred (24-hour format).",
        )

    with col2:
        age_col = st.selectbox("Insured age", options=["None"] + df.columns.tolist(), index=0)
        type_col = st.selectbox("Incident type", options=["None"] + df.columns.tolist(), index=0)
        severity_col = st.selectbox("Incident severity", options=["None"] + df.columns.tolist(), index=0)
        witnesses_col = st.selectbox("Number of witnesses", options=["None"] + df.columns.tolist(), index=0)
        property_col = st.selectbox("Property damage (1/0)", options=["None"] + df.columns.tolist(), index=0)
        police_col = st.selectbox("Police report available (1/0)", options=["None"] + df.columns.tolist(), index=0)

    if not st.button("🔍 Analyse claims for fraud", type="primary"):
        return

    with st.spinner("Processing claims through the fraud detection engine…"):
        start_time = datetime.now()
        column_mapping = {
            "claim_id": claim_id_col,
            "total_claim_amount": amount_col,
            "incident_hour_of_the_day": hour_col,
        }
        optional_fields = [
            ("age", age_col),
            ("incident_type", type_col),
            ("incident_severity", severity_col),
            ("witnesses", witnesses_col),
            ("property_damage", property_col),
            ("police_report_available", police_col),
        ]
        for field, column in optional_fields:
            if column and column != "None":
                column_mapping[field] = column

        analysis_df = data_processor.prepare_data(df, column_mapping)
        results = fraud_engine.analyse_batch(analysis_df) if hasattr(fraud_engine, "analyse_batch") else fraud_engine.analyze_batch(analysis_df)

        if len(results) > 50:
            if ml_manager.train_models(results):
                ml_predictions = ml_manager.predict_batch(results)
                results = fraud_engine.combine_predictions(results, ml_predictions)

        results = explanation_engine.add_explanations(results)
        runtime = (datetime.now() - start_time).total_seconds()
        session_manager.store_analysis(results, runtime)

    st.success(f"✅ Analysis complete! Processed {len(results)} claims in {runtime:.2f} seconds.")

    st.subheader("Fraud analysis results")
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Filter by risk level",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
    with col2:
        fraud_filter = st.selectbox("Filter by fraud prediction", options=["All", "Fraud", "Legitimate"])
    with col3:
        sort_by = st.selectbox(
            "Sort results by",
            options=["Risk score (High to Low)", "Risk score (Low to High)", "Claim ID"],
        )

    filtered_results = results[results["risk_band"].isin(risk_filter)]
    if fraud_filter != "All":
        flag = "Y" if fraud_filter == "Fraud" else "N"
        filtered_results = filtered_results[filtered_results["fraud_prediction"] == flag]

    if sort_by == "Risk score (High to Low)":
        filtered_results = filtered_results.sort_values("risk_score", ascending=False)
    elif sort_by == "Risk score (Low to High)":
        filtered_results = filtered_results.sort_values("risk_score", ascending=True)
    else:
        filtered_results = filtered_results.sort_values("claim_id")

    st.subheader(f"Analysis results ({len(filtered_results)} claims)")
    if filtered_results.empty:
        st.info("No claims match the selected filters.")
    else:
        for _, row in filtered_results.iterrows():
            render_claim_card(row)

    st.divider()
    st.subheader("Export results")

    try:
        original_df = df.copy()
        if original_df[claim_id_col].duplicated().any():
            st.warning(
                f"⚠️ Found {original_df[claim_id_col].duplicated().sum()} duplicate claim IDs. Unique identifiers were generated for export."
            )
            original_df[claim_id_col] = original_df[claim_id_col].astype(str) + "_" + original_df.index.astype(str)
        original_df = original_df.set_index(claim_id_col)

        export_results = filtered_results.copy()
        if export_results["claim_id"].duplicated().any():
            export_results["claim_id"] = export_results["claim_id"].astype(str) + "_" + export_results.index.astype(str)
        export_results = export_results.set_index("claim_id")

        export_df = pd.concat([original_df, export_results], axis=1, join="outer")
        st.download_button(
            label="📥 Download results as CSV",
            data=export_df.reset_index().to_csv(index=False),
            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Unable to prepare export: {exc}")


def render_single_claim(
    data_processor: DataProcessor,
    fraud_engine: FraudEngine,
    ml_manager: MLModelManager,
    explanation_engine: ExplanationEngine,
) -> None:
    """Render the single-claim assessment form."""
    st.header("🔍 Single claim assessment")
    st.write("Enter claim details for instant fraud risk assessment.")

    with st.form("single_claim_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Claim information")
            claim_id = st.text_input("Claim ID", value=random_identifier("CLM"))
            policy_number = st.text_input("Policy number", value=random_identifier("POL"))
            claim_amount = st.number_input(
                "Total claim amount",
                min_value=0.0,
                value=50_000.0,
                step=1_000.0,
                format="%.0f",
            )
            st.caption(f"💰 Claim amount: ₱{claim_amount:,.0f}")
            st.subheader("📅 Incident details")
            incident_date = st.date_input("Incident date")
            incident_hour = st.slider("Incident hour (24-hour format)", 0, 23, 12)

        with col2:
            st.subheader("🔍 Incident classification")
            incident_type = st.selectbox(
                "Incident type",
                [
                    "Vehicle Collision",
                    "Vehicle Theft",
                    "Property Damage",
                    "Personal Injury",
                    "Natural Disaster",
                    "Fire Damage",
                    "Other",
                ],
            )
            severity = st.selectbox("Incident severity", ["Minor Damage", "Major Damage", "Total Loss"])
            st.subheader("👥 Additional information")
            age = st.number_input("Insured age", min_value=18, max_value=100, value=35)
            witnesses = st.number_input("Number of witnesses", min_value=0, max_value=10, value=1)
            property_damage = st.selectbox("Property damage", ["Yes", "No"])
            police_report = st.selectbox("Police report filed", ["Yes", "No"])

        submitted = st.form_submit_button("🔍 Assess fraud risk", type="primary")

    if not submitted:
        return

    with st.spinner("Analyzing claim for fraud indicators…"):
        claim_data = pd.DataFrame(
            {
                "claim_id": [claim_id],
                "policy_number": [policy_number],
                "total_claim_amount": [claim_amount],
                "incident_date": [incident_date.strftime("%Y-%m-%d")],
                "incident_hour_of_the_day": [incident_hour],
                "age": [age],
                "incident_severity": [severity],
                "incident_type": [incident_type],
                "witnesses": [witnesses],
                "property_damage": [1 if property_damage == "Yes" else 0],
                "police_report_available": [1 if police_report == "Yes" else 0],
            }
        )

        processed = data_processor.prepare_single_claim(claim_data)
        result = fraud_engine.analyze_single_claim(processed)

        if ml_manager.is_trained():
            ml_prediction = ml_manager.predict_single(processed)
            result = fraud_engine.combine_single_prediction(result, ml_prediction)

        result = explanation_engine.add_single_explanation(result)

    st.subheader("Fraud risk assessment results")
    col1, col2 = st.columns([1, 2])

    with col1:
        risk_score = result["risk_score"]
        risk_band = result["risk_band"]
        color = "#059669" if risk_band == "Low" else "#d97706" if risk_band == "Medium" else "#dc2626"
        st.markdown(
            f"""
            <div style=\"text-align:center;padding:2rem;background:{color};border-radius:12px;color:white;\">
                <h2>Risk score: {risk_score}/100</h2>
                <h3>Risk level: {risk_band}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prediction_flag = result["fraud_prediction"] == "Y"
        pred_text = "FRAUDULENT" if prediction_flag else "LEGITIMATE"
        pred_color = "#dc2626" if prediction_flag else "#059669"
        st.markdown(
            f"""
            <div style=\"text-align:center;padding:1rem;background:{pred_color};border-radius:10px;color:white;margin-top:1rem;\">
                <h4>Prediction: {pred_text}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Analysis details")
        st.write("**Top risk factors:**")
        st.write(f"1. {result['reason_1']}")
        st.write(f"2. {result['reason_2']}")

        st.write("**Triggered rules:**")
        for rule in parse_triggered_rules(result.get("triggered_rules", "")):
            st.markdown(f'<span class="rule-badge">{rule}</span>', unsafe_allow_html=True)

        st.write("**Detailed explanation:**")
        st.write(result["explanation"])

    st.subheader("Recommended actions")
    if risk_score >= 70:
        st.error("🚨 **ESCALATE**: high fraud risk detected. Immediate investigation required.")
        st.write("- Assign to a senior fraud investigator.")
        st.write("- Request additional documentation.")
        st.write("- Consider claim suspension pending investigation.")
    elif risk_score >= 40:
        st.warning("⚠️ **VERIFY**: medium risk detected. Additional verification recommended.")
        st.write("- Contact claimant for supplementary information.")
        st.write("- Verify incident details with authorities.")
        st.write("- Review supporting documentation carefully.")
    else:
        st.success("✅ **APPROVE**: low fraud risk. Standard processing recommended.")
        st.write("- Process claim through the normal workflow.")
        st.write("- Perform routine documentation review.")
        st.write("- Continue monitoring for any subsequent red flags.")


def render_history(session_manager: SessionManager) -> None:
    """Display historical batch runs."""
    st.header("📈 Analysis history")
    history = session_manager.get_analysis_history()
    if not history:
        st.info("No analysis history yet. Run a batch analysis to populate this section.")
        return

    st.subheader("Recent analysis runs")
    for idx, run in enumerate(reversed(history[-10:]), start=1):
        run_number = len(history) - idx + 1
        timestamp = run["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        with st.expander(f"Run {run_number} – {timestamp}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Claims processed", len(run["results"]))
            with col2:
                st.metric("High risk claims", int(run["results"]["risk_band"].eq("High").sum()))
            with col3:
                st.metric("Processing time", f"{run['runtime']:.2f}s")

            risk_counts = run["results"]["risk_band"].value_counts()
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title=f"Risk distribution – Run {run_number}",
                color=risk_counts.index,
                color_discrete_map={"Low": "#059669", "Medium": "#d97706", "High": "#dc2626"},
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            csv_data = run["results"].to_csv(index=False)
            st.download_button(
                label=f"📥 Download run {run_number} results",
                data=csv_data,
                file_name=f"fraud_analysis_run_{run_number}_{run['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


def render_sidebar(ml_manager: MLModelManager) -> None:
    """Provide contextual system information in the sidebar."""
    st.sidebar.title("System info")
    if st.sidebar.button("Show system details"):
        st.sidebar.markdown(
            """
            **Architecture**

            - Front-end: Streamlit
            - Machine learning: scikit-learn, SMOTE
            - Visualisation: Plotly
            - Storage: session memory
            """
        )

    st.sidebar.header("Model performance")
    if ml_manager.is_trained():
        metrics = ml_manager.get_model_metrics()
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col2:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            st.metric("F1-score", f"{metrics.get('f1', 0):.3f}")
        with col3:
            st.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}")
            st.metric("Samples", metrics.get("n_samples", 0))

        importance_df = ml_manager.get_feature_importance()
        if not importance_df.empty:
            st.sidebar.subheader("Top features")
            fig = px.bar(
                importance_df.head(10),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 most important features",
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.sidebar.plotly_chart(fig, use_container_width=True)
    else:
        st.sidebar.info("Train the ML models by analysing at least 50 claims.")

    st.sidebar.header("Rule-based detection components")
    st.sidebar.markdown(
        """
        - 📊 **Z-Score analysis** – detects statistical outliers.
        - 🔢 **Benford's Law** – spots unnatural digit patterns.
        - ⏰ **Unusual hours** – flags suspicious time-of-day activity.
        - 🔄 **Frequency analysis** – catches repeat claiming patterns.
        - 💰 **Amount thresholds** – highlights unusually high claims.
        """
    )


def main() -> None:
    configure_page()
    session_manager, data_processor, fraud_engine, ml_manager, explanation_engine = initialise_managers()
    render_header()
    render_sidebar(ml_manager)

    tab_dashboard, tab_single, tab_batch, tab_history = st.tabs(
        ["📊 Dashboard", "🔍 Single claim", "📤 Batch upload", "📋 History"]
    )

    with tab_dashboard:
        render_dashboard(session_manager, ml_manager)
    with tab_single:
        render_single_claim(data_processor, fraud_engine, ml_manager, explanation_engine)
    with tab_batch:
        render_batch_upload(data_processor, fraud_engine, ml_manager, explanation_engine, session_manager)
    with tab_history:
        render_history(session_manager)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#6b7280;font-size:0.8rem;">
            Obzerra Fraud Detection System v1.0 · Local-first processing for Philippine insurers
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
