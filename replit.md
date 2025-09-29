# replit.md

## Overview

Obzerra is a local-first fraud detection MVP specifically designed for Philippine insurance claims officers. The system provides an intuitive Streamlit-based interface for analyzing insurance claims using a combination of rule-based detection algorithms and machine learning models. The application focuses on simplicity and user-friendliness, allowing claims officers to either check individual claims or upload batch CSV files for comprehensive fraud analysis. All processing is done locally without external data transmission, ensuring data privacy and security.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with custom CSS for a modern dark UI featuring blue-purple gradient design
- **Layout**: Wide layout with collapsed sidebar for maximum screen real estate
- **User Interface Components**:
  - Landing dashboard with KPI cards showing analysis statistics
  - Batch CSV upload with drag-and-drop functionality
  - Column mapping interface allowing flexible data input schema
  - Results tables with risk scoring and plain-language explanations
  - Interactive Plotly visualizations for risk distribution analysis

### Backend Architecture
- **Core Detection Engine**: Rule-based fraud detection system (`FraudEngine`) implementing weighted scoring across 8 fraud indicators
- **Data Processing Pipeline**: Modular data processor handling cleaning, validation, feature engineering, and duplicate resolution
- **ML Pipeline**: Ensemble approach combining Random Forest and Logistic Regression models with SMOTE balancing for imbalanced datasets
- **Explanation System**: Converts technical ML outputs into user-friendly insights using plain-language templates
- **Session Management**: In-memory persistence for analysis history and session statistics

### Fraud Detection Logic
- **Rule-Based Analysis**: 8 weighted fraud indicators including:
  - Z-score outliers for claim amounts (weight: 0.2)
  - Benford's Law analysis for number authenticity (weight: 0.15)
  - Temporal pattern analysis for unusual incident hours (weight: 0.1)
  - Round amount detection (weight: 0.1)
  - High-value claim flagging (weight: 0.15)
  - Young claimant high-value combinations (weight: 0.1)
  - Witness validation patterns (weight: 0.1)
  - Frequency analysis for repeat claims (weight: 0.1)
- **Risk Scoring**: 0-100 scale with Low (0-30), Medium (31-70), High (71-100) risk bands
- **Feature Engineering**: Automated creation of derived features from raw claim data

### Data Management
- **Local Processing**: All data processing occurs locally without external transmission
- **Column Mapping System**: Flexible mapping allowing users to map CSV columns to internal schema
- **Required Fields**: claim_id, total_claim_amount, incident_hour_of_the_day
- **Optional Fields**: age, incident_state, incident_severity, incident_type, witnesses
- **Data Validation**: Comprehensive cleaning with automatic duplicate handling and missing value imputation

### ML Model Architecture
- **Ensemble Approach**: Combines Logistic Regression and Random Forest models for robust predictions
- **Feature Preprocessing**: StandardScaler for numerical feature normalization
- **Imbalanced Data Handling**: SMOTE oversampling to address minority class imbalance in fraud datasets
- **Model Training**: Requires minimum 50 samples with at least 5 positive fraud cases
- **Performance Metrics**: Tracks accuracy, precision, recall, F1-score, and ROC-AUC

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing for statistical operations
- **Plotly**: Interactive data visualization (express and graph_objects)
- **Scikit-learn**: Machine learning models and preprocessing utilities
- **Imbalanced-learn**: SMOTE implementation for handling imbalanced datasets
- **SciPy**: Statistical functions for fraud detection algorithms
- **Joblib**: Model serialization and persistence

### Data Processing Dependencies
- **datetime**: Time-based feature engineering and session management
- **re**: Regular expression processing for data validation
- **base64/io**: File handling for CSV upload functionality
- **json**: Configuration and session data serialization

### No External Services
The system is designed to be completely local-first with no external API calls, database connections, or cloud service dependencies, ensuring data privacy and offline functionality.