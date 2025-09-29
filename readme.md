# Obzerra - Fraud Detection System

## Overview

Obzerra is a local-first fraud detection MVP designed specifically for Philippine insurance claims officers. The system provides an intuitive interface for analyzing insurance claims using rule-based fraud detection algorithms and machine learning models. Built with Streamlit, it offers both single claim analysis and batch CSV processing capabilities, focusing on simplicity and actionable insights for non-technical users.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with custom CSS for a modern dark UI with blue-purple gradient
- **Layout**: Wide layout with collapsed sidebar, responsive design
- **Components**: 
  - Landing dashboard with KPI cards and Plotly visualizations
  - Batch CSV upload with drag-and-drop functionality
  - Column mapping interface for flexible data input
  - Results tables with risk scoring and explanations

### Backend Architecture
- **Core Engine**: Rule-based fraud detection system (`FraudEngine`) with weighted scoring
- **Data Processing**: Modular data processor for cleaning, validation, and feature engineering
- **ML Pipeline**: Ensemble approach using Random Forest and Logistic Regression with SMOTE balancing
- **Explanation System**: Plain-language explanation engine that converts technical outputs to user-friendly insights

### Fraud Detection Logic
- **Rule-Based Analysis**: 8 weighted fraud indicators including:
  - Z-score outliers for claim amounts
  - Benford's Law analysis for number authenticity
  - Temporal pattern analysis (unusual hours)
  - Round amount detection
  - High-value claim flagging
  - Demographic risk factors
  - Witness validation
  - Frequency analysis
- **Risk Scoring**: 0-100 scale with Low/Medium/High bands
- **Feature Engineering**: Automated creation of derived features from raw claim data

### Data Management
- **Session State**: In-memory storage for analysis history and session statistics
- **Local Processing**: No external data transmission, all processing done locally
- **Column Mapping**: Flexible mapping system allowing users to map their CSV columns to internal schema
- **Data Validation**: Comprehensive cleaning and validation with duplicate handling

### ML Model Architecture
- **Ensemble Approach**: Combines Logistic Regression and Random Forest models
- **Feature Scaling**: StandardScaler for numerical feature normalization
- **Imbalanced Data Handling**: SMOTE oversampling for minority class balancing
- **Model Metrics**: Comprehensive evaluation including accuracy, precision, recall, F1-score, and ROC-AUC
- **Feature Importance**: Analysis and visualization of most predictive features

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas/Numpy**: Data manipulation and numerical computing
- **Plotly**: Interactive data visualization and charting
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Imbalanced-learn**: SMOTE implementation for handling class imbalance
- **SciPy**: Statistical analysis functions for fraud detection rules
- **Joblib**: Model serialization and persistence

### Data Requirements
- **Input Format**: CSV files with flexible column mapping
- **Required Fields**: claim_id, total_claim_amount, incident_hour_of_the_day
- **Optional Fields**: age, incident_state, incident_severity, incident_type, witnesses, policy_number, incident_date, property_damage, police_report_available, fraud_reported

### Statistical Methods
- **Benford's Law**: First-digit distribution analysis for fraud detection
- **Z-score Analysis**: Statistical outlier detection for claim amounts
- **Cross-validation**: Model validation and performance assessment
- **Feature Correlation**: Analysis of predictive feature relationships

### No External APIs
- The system is designed to be completely local-first with no external API dependencies
- All processing, analysis, and storage happens within the Streamlit application
- No database connections or cloud services required