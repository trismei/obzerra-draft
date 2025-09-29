import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
import io
import json
from utils.data_processor import DataProcessor
from utils.fraud_engine import FraudEngine
from utils.ml_models import MLModelManager
from utils.explanations import ExplanationEngine
from utils.session_manager import SessionManager

# Page configuration
st.set_page_config(
    page_title="Obzerra - Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state and managers
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
    st.session_state.data_processor = DataProcessor()
    st.session_state.fraud_engine = FraudEngine()
    st.session_state.ml_manager = MLModelManager()
    st.session_state.explanation_engine = ExplanationEngine()

session_manager = st.session_state.session_manager
data_processor = st.session_state.data_processor
fraud_engine = st.session_state.fraud_engine
ml_manager = st.session_state.ml_manager
explanation_engine = st.session_state.explanation_engine

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: #1e293b;
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        text-align: left;
        border-bottom: 2px solid #6366f1;
    }
    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-logo {
        font-size: 2rem;
        font-weight: bold;
        color: #6366f1;
        margin: 0;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0;
    }
    .kpi-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        margin-bottom: 1rem;
    }
    .risk-badge-low {
        background: linear-gradient(135deg, #059669, #047857);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(5, 150, 105, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .risk-badge-medium {
        background: linear-gradient(135deg, #d97706, #b45309);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(217, 119, 6, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .risk-badge-high {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(220, 38, 38, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 2px 4px rgba(220, 38, 38, 0.3); }
        50% { box-shadow: 0 4px 8px rgba(220, 38, 38, 0.5); }
    }
    .rule-badge {
        background-color: #374151;
        color: #f3f4f6;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.7rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .claim-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .claim-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .claim-id {
        font-size: 1.2rem;
        font-weight: bold;
        color: #f1f5f9;
    }
    .risk-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        min-width: 80px;
    }
    .risk-score-low {
        background: #059669;
        color: white;
    }
    .risk-score-medium {
        background: #d97706;
        color: white;
    }
    .risk-score-high {
        background: #dc2626;
        color: white;
    }
    .fraud-indicator {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .fraud-yes {
        background: #fca5a5;
        color: #991b1b;
    }
    .fraud-no {
        background: #86efac;
        color: #166534;
    }
    .claim-details {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    .detail-section {
        padding: 1rem;
        background: #0f172a;
        border-radius: 8px;
    }
    .detail-title {
        font-weight: bold;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .detail-content {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .rule-icon {
        margin-right: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with Obzerra branding
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <div>
            <h1 class="brand-logo">🛡️ Obzerra</h1>
            <p class="brand-subtitle">Philippine Insurance Fraud Detection System</p>
        </div>
        <div style="color: #94a3b8; font-size: 0.9rem;">
            Advanced ML-powered fraud detection
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Single Claim", "📤 Batch Upload", "📋 History"])

# Dashboard Tab
with tab1:
    st.header("📊 Dashboard Overview")
    
    # Enhanced KPI Cards with Icons and Visual Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    total_claims = session_manager.get_total_claims()
    high_risk_pct = session_manager.get_high_risk_percentage()
    avg_runtime = session_manager.get_avg_runtime()
    model_accuracy = ml_manager.get_model_accuracy()
    
    with col1:
        icon_color = "#6366f1" if total_claims > 0 else "#64748b"
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 4px solid {icon_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #94a3b8; margin: 0; font-size: 0.9rem;">📋 Total Claims</h3>
                    <h2 style="color: {icon_color}; margin: 0.5rem 0; font-size: 2.2rem;">{total_claims:,}</h2>
                    <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Claims analyzed</p>
                </div>
                <div style="font-size: 3rem; opacity: 0.3; color: {icon_color};">📊</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_color = "#dc2626" if high_risk_pct > 15 else "#059669" if high_risk_pct < 5 else "#d97706"
        risk_status = "High Alert" if high_risk_pct > 15 else "Normal" if high_risk_pct < 5 else "Monitor"
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 4px solid {risk_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #94a3b8; margin: 0; font-size: 0.9rem;">⚠️ High Risk</h3>
                    <h2 style="color: {risk_color}; margin: 0.5rem 0; font-size: 2.2rem;">{high_risk_pct:.1f}%</h2>
                    <p style="color: {risk_color}; font-size: 0.8rem; margin: 0; font-weight: bold;">{risk_status}</p>
                </div>
                <div style="font-size: 3rem; opacity: 0.3; color: {risk_color};">🚨</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        speed_color = "#059669" if avg_runtime < 2 else "#d97706" if avg_runtime < 5 else "#dc2626"
        speed_status = "Fast" if avg_runtime < 2 else "Normal" if avg_runtime < 5 else "Slow"
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 4px solid {speed_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #94a3b8; margin: 0; font-size: 0.9rem;">⚡ Processing</h3>
                    <h2 style="color: {speed_color}; margin: 0.5rem 0; font-size: 2.2rem;">{avg_runtime:.1f}s</h2>
                    <p style="color: {speed_color}; font-size: 0.8rem; margin: 0; font-weight: bold;">{speed_status}</p>
                </div>
                <div style="font-size: 3rem; opacity: 0.3; color: {speed_color};">⏱️</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        acc_color = "#059669" if model_accuracy > 0.85 else "#d97706" if model_accuracy > 0.7 else "#dc2626"
        acc_status = "Excellent" if model_accuracy > 0.85 else "Good" if model_accuracy > 0.7 else "Training"
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-left: 4px solid {acc_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #94a3b8; margin: 0; font-size: 0.9rem;">🎯 ML Accuracy</h3>
                    <h2 style="color: {acc_color}; margin: 0.5rem 0; font-size: 2.2rem;">{model_accuracy*100:.0f}%</h2>
                    <p style="color: {acc_color}; font-size: 0.8rem; margin: 0; font-weight: bold;">{acc_status}</p>
                </div>
                <div style="font-size: 3rem; opacity: 0.3; color: {acc_color};">🤖</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Enhanced Analytics Dashboard
    recent_results = session_manager.get_recent_results()
    if recent_results:
        # Create comprehensive analytics
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        all_rules = []
        timestamps = []
        daily_counts = {}
        
        for result in recent_results:
            timestamp = result.get('timestamp', datetime.now())
            timestamps.append(timestamp)
            date_key = timestamp.strftime('%Y-%m-%d')
            daily_counts[date_key] = daily_counts.get(date_key, 0) + len(result['results'])
            
            for _, row in result['results'].iterrows():
                risk_counts[row['risk_band']] += 1
                if pd.notna(row.get('triggered_rules', '')):
                    rules = str(row['triggered_rules']).split(', ')
                    all_rules.extend([r.strip() for r in rules if r.strip()])
        
        # Create three-column layout for charts
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        with chart_col1:
            st.subheader("🍩 Risk Distribution")
            # Enhanced donut chart
            fig_donut = go.Figure(data=[go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=0.6,
                marker_colors=['#059669', '#d97706', '#dc2626'],
                textinfo='label+percent',
                textfont_size=12,
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=40, b=40, l=40, r=40),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            
            # Add center text
            fig_donut.add_annotation(
                text=f"<b>{sum(risk_counts.values())}</b><br>Total Claims",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )
            
            st.plotly_chart(fig_donut, width='stretch', config={'displayModeBar': False})
        
        with chart_col2:
            st.subheader("📈 Claims Trend")
            if len(daily_counts) > 1:
                # Time series chart
                dates = sorted(daily_counts.keys())
                counts = [daily_counts[date] for date in dates]
                
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=dates,
                    y=counts,
                    mode='lines+markers',
                    line=dict(color='#6366f1', width=3),
                    marker=dict(size=8, color='#6366f1'),
                    fill='tonexty',
                    fillcolor='rgba(99, 102, 241, 0.1)',
                    hovertemplate='<b>%{x}</b><br>Claims: %{y}<extra></extra>'
                ))
                
                fig_line.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Claims Processed",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=300,
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                
                st.plotly_chart(fig_line, width='stretch', config={'displayModeBar': False})
            else:
                st.info("📅 More data needed for trend analysis")
        
        with chart_col3:
            st.subheader("🔍 Top Fraud Indicators")
            if all_rules:
                # Count rule frequencies
                rule_counts = {}
                for rule in all_rules:
                    if rule and rule != '':
                        rule_counts[rule] = rule_counts.get(rule, 0) + 1
                
                # Get top 5 rules
                top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                if top_rules:
                    rule_names = [rule[0] for rule in top_rules]
                    rule_values = [rule[1] for rule in top_rules]
                    
                    # Map rules to colors
                    rule_colors = ['#dc2626', '#d97706', '#059669', '#6366f1', '#8b5cf6']
                    
                    fig_bar = go.Figure(data=[go.Bar(
                        y=rule_names,
                        x=rule_values,
                        orientation='h',
                        marker_color=rule_colors[:len(rule_names)],
                        hovertemplate='<b>%{y}</b><br>Triggered: %{x} times<extra></extra>'
                    )])
                    
                    fig_bar.update_layout(
                        xaxis_title="Frequency",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        height=300,
                        margin=dict(t=40, b=40, l=60, r=40)
                    )
                    
                    st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})
                else:
                    st.success("✅ No fraud indicators triggered recently")
            else:
                st.success("✅ No fraud indicators triggered recently")
    else:
        st.info("📊 No analysis data available yet. Upload and process some claims to see comprehensive analytics.")

# Batch Upload Tab
with tab3:
    st.header("📂 Batch Claims Analysis")
    
    # File upload section with help
    st.subheader("Upload Claims Data")
    
    # Add sample CSV template download - available before file upload
    with st.expander("📋 Download Sample CSV Template"):
        st.write("**Download our standard Philippine insurance claims template:")
        sample_data = {
            'claim_id': ['CLM_001', 'CLM_002', 'CLM_003'],
            'policy_number': ['POL_123456', 'POL_789012', 'POL_345678'],
            'total_claim_amount': [25000, 75000, 150000],
            'incident_date': ['2024-01-15', '2024-01-20', '2024-01-25'],
            'incident_hour_of_the_day': [14, 22, 8],
            'incident_type': ['Vehicle Collision', 'Property Damage', 'Vehicle Theft'],
            'incident_severity': ['Minor Damage', 'Major Damage', 'Total Loss'],
            'witnesses': [2, 0, 1],
            'property_damage': [1, 1, 0],
            'police_report_available': [1, 1, 0],
            'fraud_reported': [0, 0, 1]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, width='stretch')
        
        csv_template = sample_df.to_csv(index=False)
        st.download_button(
            label="📋 Download CSV Template",
            data=csv_template,
            file_name="obzerra_claims_template.csv",
            mime="text/csv",
            help="Use this template as a reference for your CSV structure"
        )
    
    with st.expander("ℹ️ Need help with file format?"):
        st.write("""
        **Required columns:**
        - **Claim ID**: Unique identifier for each claim
        - **Claim Amount**: The monetary value being claimed (in PHP)
        - **Incident Hour**: When the incident occurred (0-23 format)
        
        **Optional columns (for better accuracy):**
        - **Policy Number**: Insurance policy number for traceability
        - **Incident Date**: When the incident occurred
        - **Incident Type**: Type of incident (collision, theft, etc.)
        - **Incident Severity**: Minor Damage, Major Damage, or Total Loss
        - **Age**: Age of the insured person
        - **Witnesses**: Number of witnesses present
        - **Property Damage**: Whether property damage occurred (1=Yes, 0=No)
        - **Police Report Available**: Whether police report was filed (1=Yes, 0=No)
        
        💡 **Tip**: Your CSV can have different column names - we'll help you map them to our Philippine insurance system!
        """)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here or click to browse",
        type=['csv'],
        help="Upload a CSV file containing insurance claims data for fraud analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load and preview data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
            
            # Show preview
            st.subheader("Data Preview (First 20 rows)")
            st.dataframe(df.head(20), width='stretch')
            
            # Column mapping section - Universal Schema for Philippine Insurance
            st.subheader("Column Mapping")
            st.write("Map your CSV columns to our system fields. Our universal schema accepts any CSV structure:")
            
            # Column mapping help
            st.info("💡 Use the template above as reference, then map your CSV columns below to match our Philippine insurance schema.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Required Fields:**")
                claim_id_col = st.selectbox("Claim ID (or Policy Number)", options=df.columns, 
                                          index=next((i for i, col in enumerate(df.columns) if any(term in col.lower() for term in ['claim', 'policy', 'id'])), 0),
                                          help="Unique identifier for each claim")
                amount_col = st.selectbox("Total Claim Amount (₱)", 
                                        options=[col for col in df.columns if any(term in col.lower() for term in ['amount', 'claim', 'value', 'cost'])],
                                        index=0 if any('amount' in col.lower() for col in df.columns) else 0,
                                        help="Monetary amount being claimed in PHP")
                hour_col = st.selectbox("Incident Hour (0-23)", 
                                      options=[col for col in df.columns if any(term in col.lower() for term in ['hour', 'time'])],
                                      index=0 if any('hour' in col.lower() for col in df.columns) else 0,
                                      help="Hour when incident occurred (24-hour format)")
            
            with col2:
                st.write("**Optional Fields (for better accuracy):**")
                age_col = st.selectbox("Insured Age", options=['None'] + list(df.columns), 
                                     index=next((i+1 for i, col in enumerate(df.columns) if 'age' in col.lower()), 0),
                                     help="Age of the insured person")
                type_col = st.selectbox("Incident Type", options=['None'] + list(df.columns),
                                      index=next((i+1 for i, col in enumerate(df.columns) if 'type' in col.lower()), 0),
                                      help="Type of incident (collision, theft, etc.)")
                severity_col = st.selectbox("Incident Severity", options=['None'] + list(df.columns),
                                          index=next((i+1 for i, col in enumerate(df.columns) if 'severity' in col.lower()), 0),
                                          help="Extent of damage (Minor/Major/Total Loss)")
                witnesses_col = st.selectbox("Number of Witnesses", options=['None'] + list(df.columns),
                                           index=next((i+1 for i, col in enumerate(df.columns) if 'witness' in col.lower()), 0),
                                           help="Number of witnesses present")
                property_col = st.selectbox("Property Damage (Y/N)", options=['None'] + list(df.columns),
                                          index=next((i+1 for i, col in enumerate(df.columns) if 'property' in col.lower()), 0),
                                          help="Whether property damage occurred")
                police_col = st.selectbox("Police Report Available (Y/N)", options=['None'] + list(df.columns),
                                        index=next((i+1 for i, col in enumerate(df.columns) if 'police' in col.lower()), 0),
                                        help="Whether police report was filed")
            
            # Process button
            if st.button("🔍 Analyze Claims for Fraud", type="primary"):
                with st.spinner("Processing claims through fraud detection engine..."):
                    start_time = datetime.now()
                    
                    # Prepare data for analysis - Philippine universal schema
                    column_mapping = {
                        'claim_id': claim_id_col,
                        'total_claim_amount': amount_col,
                        'incident_hour_of_the_day': hour_col
                    }
                    
                    # Add optional Philippine-specific fields
                    optional_mappings = [
                        ('age', age_col),
                        ('incident_type', type_col),
                        ('incident_severity', severity_col),
                        ('witnesses', witnesses_col),
                        ('property_damage', property_col),
                        ('police_report_available', police_col)
                    ]
                    
                    for field, col_var in optional_mappings:
                        if col_var != 'None' and col_var is not None:
                            column_mapping[field] = col_var
                    
                    analysis_df = data_processor.prepare_data(df, column_mapping)
                    
                    # Run fraud detection
                    results = fraud_engine.analyze_batch(analysis_df)
                    
                    # Train/update ML models if enough data
                    if len(results) > 50:
                        ml_manager.train_models(results)
                        ml_predictions = ml_manager.predict_batch(results)
                        results = fraud_engine.combine_predictions(results, ml_predictions)
                    
                    # Add explanations
                    results = explanation_engine.add_explanations(results)
                    
                    end_time = datetime.now()
                    runtime = (end_time - start_time).total_seconds()
                    
                    # Store results
                    session_manager.store_analysis(results, runtime)
                    
                    st.success(f"✅ Analysis complete! Processed {len(results)} claims in {runtime:.2f} seconds.")
                    
                    # Display results
                    st.subheader("Fraud Analysis Results")
                    
                    # Filter options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        risk_filter = st.multiselect("Filter by Risk Level", 
                                                   options=['Low', 'Medium', 'High'],
                                                   default=['Low', 'Medium', 'High'])
                    
                    with col2:
                        fraud_filter = st.selectbox("Filter by Fraud Prediction", 
                                                   options=['All', 'Fraud', 'Legitimate'])
                    
                    with col3:
                        sort_by = st.selectbox("Sort by", 
                                             options=['Risk Score (High to Low)', 'Risk Score (Low to High)', 'Claim ID'])
                    
                    # Apply filters
                    filtered_results = results[results['risk_band'].isin(risk_filter)]
                    if fraud_filter != 'All':
                        fraud_value = 'Y' if fraud_filter == 'Fraud' else 'N'
                        filtered_results = filtered_results[filtered_results['fraud_prediction'] == fraud_value]
                    
                    # Apply sorting
                    if sort_by == 'Risk Score (High to Low)':
                        filtered_results = filtered_results.sort_values('risk_score', ascending=False)
                    elif sort_by == 'Risk Score (Low to High)':
                        filtered_results = filtered_results.sort_values('risk_score', ascending=True)
                    else:
                        filtered_results = filtered_results.sort_values('claim_id')
                    
                    # Display results in modern card layout
                    st.subheader(f"Analysis Results ({len(filtered_results)} claims)")
                    
                    if len(filtered_results) > 0:
                        # Function to get risk score class
                        def get_risk_class(risk_band):
                            return {
                                'Low': 'risk-score-low',
                                'Medium': 'risk-score-medium', 
                                'High': 'risk-score-high'
                            }.get(risk_band, 'risk-score-low')
                        
                        # Function to get rule icons with user-friendly explanations
                        def get_rule_icons(rules_text):
                            if pd.isna(rules_text) or rules_text == '':
                                return []
                            
                            rule_explanations = {
                                'z-score': ('📊', 'Amount Analysis', 'Claim amount is unusually high or low compared to similar claims'),
                                'benford': ('🔢', 'Number Pattern Check', 'The claim amount has an unusual digit pattern'),
                                'unusual hour': ('🕕', 'Time Pattern', 'Incident occurred at an unusual time (late night/early morning)'),
                                'high amount': ('💵', 'Large Claim', 'This is a high-value claim requiring extra verification'),
                                'round amount': ('🎯', 'Round Number', 'Claim amount is a suspiciously round number (e.g., exactly $10,000)'),
                                'frequency': ('🔄', 'Repeat Pattern', 'Multiple similar claims detected recently'),
                                'outlier': ('⚠️', 'Statistical Outlier', 'This claim stands out significantly from normal patterns'),
                                'threshold': ('🚨', 'Policy Limit', 'Claim approaches or exceeds policy thresholds')
                            }
                            
                            matched_rules = []
                            rules_lower = rules_text.lower()
                            
                            for key, (icon, title, explanation) in rule_explanations.items():
                                if key in rules_lower:
                                    matched_rules.append(f'''
                                    <div style="background: #0f172a; padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid #6366f1;">
                                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem;">
                                            <span style="font-size: 1.2rem;">{icon}</span>
                                            <strong style="color: #e2e8f0; font-size: 0.9rem;">{title}</strong>
                                        </div>
                                        <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.3;">{explanation}</div>
                                    </div>
                                    ''')
                            return matched_rules
                        
                        # Display each claim as a card
                        for idx, row in filtered_results.iterrows():
                            risk_class = get_risk_class(row['risk_band'])
                            fraud_class = 'fraud-yes' if row['fraud_prediction'] == 'Y' else 'fraud-no'
                            # Enhanced fraud indicators with icons
                            if row['fraud_prediction'] == 'Y':
                                fraud_text = '🚨 High Fraud Risk'
                                fraud_class = 'fraud-yes'
                            else:
                                fraud_text = '✅ Legitimate Claim'
                                fraud_class = 'fraud-no'
                            
                            # Add risk level icons
                            risk_icons = {
                                'Low': '🟢',
                                'Medium': '🟡', 
                                'High': '🔴'
                            }
                            risk_icon = risk_icons.get(row['risk_band'], '⚪')
                            
                            rule_icons = get_rule_icons(row.get('triggered_rules', ''))
                            
                            st.markdown(f"""
                            <div class="claim-card">
                                <div class="claim-header">
                                    <div class="claim-id">📋 {row['claim_id']}</div>
                                    <div style="display: flex; align-items: center; gap: 1rem;">
                                        <div class="fraud-indicator {fraud_class}">{fraud_text}</div>
                                        <div class="risk-score {risk_class}">
                                            {risk_icon}<br>
                                            <span style="font-size: 1.5rem;">{int(row['risk_score'])}</span><br>
                                            <span style="font-size: 0.8rem;">{row['risk_band']}</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="claim-details">
                                    <div class="detail-section">
                                        <div class="detail-title">🤖 AI Assessment</div>
                                        <div class="detail-content">
                                            <strong>Risk Level:</strong> {row['risk_band']}<br>
                                            <strong>Primary Reason:</strong> {row.get('reason_1', 'N/A')}<br>
                                            <strong>Secondary Reason:</strong> {row.get('reason_2', 'N/A')}
                                        </div>
                                    </div>
                                    
                                    <div class="detail-section">
                                        <div class="detail-title">🔍 Why This Result?</div>
                                        <div class="detail-content">
                                            {''.join(rule_icons) if rule_icons else '<div style="color: #10b981; font-style: italic;">✅ No suspicious patterns detected - claim appears normal</div>'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No claims match the current filters.")
                    
                    st.divider()
                    
                    # Export functionality
                    st.subheader("Export Results")
                    
                    # Prepare export data - handle duplicate indexes gracefully
                    try:
                        # Check for duplicate claim IDs in original data
                        if df[claim_id_col].duplicated().any():
                            st.warning(f"⚠️ Found {df[claim_id_col].duplicated().sum()} duplicate claim IDs. Auto-generating unique identifiers for export.")
                            # Create unique claim IDs by appending suffix
                            df_export = df.copy()
                            df_export[claim_id_col] = df_export[claim_id_col].astype(str) + '_' + df_export.index.astype(str)
                            original_df = df_export.set_index(claim_id_col)
                        else:
                            original_df = df.set_index(claim_id_col)
                        
                        # Ensure results also have unique indexes
                        if filtered_results['claim_id'].duplicated().any():
                            filtered_results_export = filtered_results.copy()
                            filtered_results_export['claim_id'] = filtered_results_export['claim_id'].astype(str) + '_' + filtered_results_export.index.astype(str)
                            results_df = filtered_results_export.set_index('claim_id')
                        else:
                            results_df = filtered_results.set_index('claim_id')
                        
                        export_df = pd.concat([original_df, results_df], axis=1, join='outer')
                        
                    except Exception as e:
                        st.error(f"Error preparing export data: {str(e)}")
                        # Fallback: export without joining
                        export_df = filtered_results
                    
                    csv = export_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Single Claim Tab
with tab2:
    st.header("🔍 Single Claim Assessment")
    
    st.write("Enter claim details for instant fraud risk assessment:")
    
    # Input form - Philippine Insurance Context
    with st.form("single_claim_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Claim Information")
            claim_id = st.text_input("Claim ID", value="CLM_" + str(np.random.randint(100000, 999999)), help="Unique identifier for this claim")
            policy_number = st.text_input("Policy Number", value="POL_" + str(np.random.randint(100000, 999999)), help="Insurance policy number for traceability")
            claim_amount = st.number_input("Total Claim Amount", min_value=0.0, value=50000.0, step=1000.0, format="%.0f", help="Amount being claimed in Philippine Peso (₱)")
            st.caption(f"💰 Claim Amount: ₱{claim_amount:,.0f}")
            
            st.subheader("📅 Incident Details")
            incident_date = st.date_input("Incident Date", help="When did the incident occur?")
            incident_hour = st.slider("Incident Hour (24-hour format)", 0, 23, 12, help="What time did the incident happen?")
        
        with col2:
            st.subheader("🔍 Incident Classification")
            incident_type = st.selectbox("Incident Type", 
                                       ['Vehicle Collision', 'Vehicle Theft', 'Property Damage', 'Personal Injury', 'Natural Disaster', 'Fire Damage', 'Other'],
                                       help="Type of incident that occurred")
            severity = st.selectbox("Incident Severity", ['Minor Damage', 'Major Damage', 'Total Loss'],
                                   help="Extent of damage or loss")
            
            st.subheader("👥 Additional Information")
            age = st.number_input("Insured Age", min_value=18, max_value=100, value=35, help="Age of the insured person")
            witnesses = st.number_input("Number of Witnesses", min_value=0, max_value=10, value=1, help="How many witnesses were present?")
            property_damage = st.selectbox("Property Damage", ['Yes', 'No'], help="Was there damage to property?")
            police_report = st.selectbox("Police Report Filed", ['Yes', 'No'], help="Was a police report filed for this incident?")
        
        submitted = st.form_submit_button("🔍 Assess Fraud Risk", type="primary")
        
        if submitted:
            with st.spinner("Analyzing claim for fraud indicators..."):
                # Create single claim dataframe - Philippine context
                claim_data = pd.DataFrame({
                    'claim_id': [claim_id],
                    'policy_number': [policy_number],
                    'total_claim_amount': [claim_amount],
                    'incident_date': [incident_date.strftime('%Y-%m-%d')],
                    'incident_hour_of_the_day': [incident_hour],
                    'age': [age],
                    'incident_severity': [severity],
                    'incident_type': [incident_type],
                    'witnesses': [witnesses],
                    'property_damage': [1 if property_damage == 'Yes' else 0],
                    'police_report_available': [1 if police_report == 'Yes' else 0]
                })
                
                # Process through fraud engine
                processed_data = data_processor.prepare_single_claim(claim_data)
                result = fraud_engine.analyze_single_claim(processed_data)
                
                # Get ML prediction if model is trained
                if ml_manager.is_trained():
                    ml_pred = ml_manager.predict_single(processed_data)
                    result = fraud_engine.combine_single_prediction(result, ml_pred)
                
                # Add explanation
                result = explanation_engine.add_single_explanation(result)
                
                # Display results
                st.subheader("Fraud Risk Assessment Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Risk score display
                    risk_score = result['risk_score']
                    risk_band = result['risk_band']
                    
                    if risk_band == 'Low':
                        color = '#059669'
                    elif risk_band == 'Medium':
                        color = '#d97706'
                    else:
                        color = '#dc2626'
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: {color}; border-radius: 10px; color: white;">
                        <h2>Risk Score: {risk_score}/100</h2>
                        <h3>Risk Level: {risk_band}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Fraud prediction
                    fraud_pred = result['fraud_prediction']
                    pred_text = "FRAUDULENT" if fraud_pred == 'Y' else "LEGITIMATE"
                    pred_color = "#dc2626" if fraud_pred == 'Y' else "#059669"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: {pred_color}; border-radius: 8px; color: white; margin-top: 1rem;">
                        <h4>Prediction: {pred_text}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Explanations and reasons
                    st.subheader("Analysis Details")
                    
                    st.write("**Top Risk Factors:**")
                    st.write(f"1. {result['reason_1']}")
                    st.write(f"2. {result['reason_2']}")
                    
                    st.write("**Triggered Rules:**")
                    rules = result['triggered_rules'].split(', ') if result['triggered_rules'] else []
                    for rule in rules:
                        st.markdown(f'<span class="rule-badge">{rule}</span>', unsafe_allow_html=True)
                    
                    st.write("**Detailed Explanation:**")
                    st.write(result['explanation'])
                
                # Recommended actions
                st.subheader("Recommended Actions")
                
                if risk_score >= 70:
                    st.error("🚨 **ESCALATE**: High fraud risk detected. Immediate investigation required.")
                    st.write("- Assign to senior fraud investigator")
                    st.write("- Request additional documentation")
                    st.write("- Consider claim suspension pending investigation")
                elif risk_score >= 40:
                    st.warning("⚠️ **VERIFY**: Medium risk detected. Additional verification recommended.")
                    st.write("- Contact claimant for additional information")
                    st.write("- Verify incident details with authorities")
                    st.write("- Review supporting documentation carefully")
                else:
                    st.success("✅ **APPROVE**: Low fraud risk. Standard processing recommended.")
                    st.write("- Process claim through normal workflow")
                    st.write("- Routine documentation review")
                    st.write("- Standard settlement procedures")

# History Tab
with tab4:
    st.header("📈 Analysis History")
    
    history = session_manager.get_analysis_history()
    
    if history:
        st.subheader("Recent Analysis Runs")
        
        for i, run in enumerate(reversed(history[-10:])):  # Show last 10 runs
            with st.expander(f"Run {len(history)-i} - {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Claims Processed", len(run['results']))
                
                with col2:
                    high_risk = len(run['results'][run['results']['risk_band'] == 'High'])
                    st.metric("High Risk Claims", high_risk)
                
                with col3:
                    st.metric("Processing Time", f"{run['runtime']:.2f}s")
                
                # Risk distribution for this run
                risk_counts = run['results']['risk_band'].value_counts()
                fig = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title=f"Risk Distribution - Run {len(history)-i}",
                    color=risk_counts.index,
                    color_discrete_map={'Low': '#059669', 'Medium': '#d97706', 'High': '#dc2626'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, width='stretch')
                
                # Download button for this run
                csv = run['results'].to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Run {len(history)-i} Results",
                    data=csv,
                    file_name=f"fraud_analysis_run_{len(history)-i}_{run['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("No analysis history available yet. Run some batch analyses to see history here.")

# Add system info to sidebar instead
with st.sidebar:
    st.title("System Info")
    if st.button("Show System Details"):
        st.write("**System Architecture:**")
        st.write("- **Frontend**: Streamlit")
        st.write("- **ML**: Scikit-learn, SMOTE")
        st.write("- **Visualization**: Plotly")
        st.write("- **Storage**: Local session")
    st.header("ℹ️ System Information")
    
    st.subheader("Model Performance")
    
    if ml_manager.is_trained():
        metrics = ml_manager.get_model_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Precision", f"{metrics['precision']:.3f}")
        
        with col2:
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1-Score", f"{metrics['f1']:.3f}")
        
        with col3:
            st.metric("AUC-ROC", f"{metrics['auc']:.3f}")
            st.metric("Training Samples", metrics['n_samples'])
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = ml_manager.get_feature_importance()
        
        if not importance_df.empty:
            fig = px.bar(
                importance_df.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, width='stretch')
    
    else:
        st.info("Machine learning models not yet trained. Process at least 50 claims to enable ML predictions.")
    
    st.subheader("Rule-Based Detection Components")
    
    st.write("**Active Detection Rules:**")
    st.write("- 📊 **Z-Score Analysis**: Detects statistical outliers in claim amounts")
    st.write("- 🔢 **Benford's Law**: Identifies unnatural digit distributions")
    st.write("- ⏰ **Unusual Hours**: Flags incidents at suspicious times")
    st.write("- 🔄 **Frequency Analysis**: Detects repeated claiming patterns")
    st.write("- 💰 **Amount Thresholds**: Identifies unusually high/low claims")
    st.write("- 📅 **Temporal Patterns**: Analyzes timing and seasonal trends")
    
    st.subheader("System Architecture")
    
    st.write("**Technology Stack:**")
    st.write("- **Frontend**: Streamlit with dark theme")
    st.write("- **Data Processing**: Pandas, NumPy")
    st.write("- **Machine Learning**: Scikit-learn, Imbalanced-learn (SMOTE)")
    st.write("- **Visualization**: Plotly")
    st.write("- **Explainability**: SHAP (converted to plain language)")
    st.write("- **Storage**: Session-based local persistence")
    
    st.write("**Model Pipeline:**")
    st.write("1. **Data Preprocessing**: Column mapping, feature engineering")
    st.write("2. **Rule-Based Analysis**: Statistical tests and business rules")
    st.write("3. **ML Prediction**: Ensemble of Logistic Regression + Random Forest")
    st.write("4. **SMOTE Balancing**: Handles imbalanced fraud datasets")
    st.write("5. **Explanation Generation**: Converts technical outputs to plain language")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.8rem;">
    Obzerra Fraud Detection System v1.0 | 
    Built with Streamlit and Machine Learning | 
    Local-first processing ensures data privacy
</div>
""", unsafe_allow_html=True)
