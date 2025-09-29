import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class DataProcessor:
    """Handles data preprocessing and feature engineering for fraud detection."""
    
    def __init__(self):
        self.required_columns = ['claim_id', 'total_claim_amount', 'incident_hour_of_the_day']
        self.optional_columns = ['age', 'incident_state', 'incident_severity', 'incident_type', 'witnesses']
    
    def prepare_data(self, df, column_mapping):
        """Prepare batch data for fraud analysis."""
        try:
            # Create a copy and rename columns
            processed_df = df.copy()
            
            # Rename columns based on mapping
            rename_dict = {}
            for target_col, source_col in column_mapping.items():
                if source_col and source_col in df.columns:
                    rename_dict[source_col] = target_col
            
            processed_df = processed_df.rename(columns=rename_dict)
            
            # Ensure required columns exist
            for col in self.required_columns:
                if col not in processed_df.columns:
                    if col == 'claim_id':
                        processed_df[col] = ['CLM_' + str(i) for i in range(len(processed_df))]
                    elif col == 'total_claim_amount':
                        raise ValueError(f"Required column '{col}' not found in data")
                    elif col == 'incident_hour_of_the_day':
                        processed_df[col] = 12  # Default to noon
            
            # Handle duplicate claim_ids by making them unique
            if 'claim_id' in processed_df.columns and processed_df['claim_id'].duplicated().any():
                duplicates_count = processed_df['claim_id'].duplicated().sum()
                # Make unique by appending row index to duplicates
                mask = processed_df['claim_id'].duplicated(keep='first')
                processed_df.loc[mask, 'claim_id'] = processed_df.loc[mask, 'claim_id'].astype(str) + '_dup_' + processed_df.loc[mask].index.astype(str)
                print(f"Warning: Found {duplicates_count} duplicate claim IDs. Made unique automatically.")
            
            # Clean and validate data
            processed_df = self._clean_data(processed_df)
            
            # Feature engineering
            processed_df = self._engineer_features(processed_df)
            
            return processed_df
            
        except Exception as e:
            print(f"Data preparation error: {str(e)}")
            raise Exception(f"Data preparation failed: {str(e)}")
    
    def prepare_single_claim(self, claim_data):
        """Prepare single claim data for analysis."""
        processed_data = claim_data.copy()
        processed_data = self._clean_data(processed_data)
        processed_data = self._engineer_features(processed_data)
        return processed_data
    
    def _clean_data(self, df):
        """Clean and validate the data."""
        # Convert claim amount to numeric
        if 'total_claim_amount' in df.columns:
            df['total_claim_amount'] = pd.to_numeric(df['total_claim_amount'], errors='coerce')
            df = df.dropna(subset=['total_claim_amount'])
            df = df[df['total_claim_amount'] > 0]  # Remove zero or negative claims
        
        # Convert incident hour to numeric
        if 'incident_hour_of_the_day' in df.columns:
            df['incident_hour_of_the_day'] = pd.to_numeric(df['incident_hour_of_the_day'], errors='coerce')
            df['incident_hour_of_the_day'] = df['incident_hour_of_the_day'].fillna(12)  # Default to noon
            df['incident_hour_of_the_day'] = df['incident_hour_of_the_day'].clip(0, 23)
        
        # Convert age to numeric if present
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].fillna(35)  # Default age
            df['age'] = df['age'].clip(18, 100)
        
        # Convert witnesses to numeric if present
        if 'witnesses' in df.columns:
            df['witnesses'] = pd.to_numeric(df['witnesses'], errors='coerce')
            df['witnesses'] = df['witnesses'].fillna(1).clip(0, 10)
        
        return df
    
    def _engineer_features(self, df):
        """Engineer new features for fraud detection."""
        
        # Amount-based features
        if 'total_claim_amount' in df.columns:
            # Log transform of amount (helps with skewed distributions)
            df['log_claim_amount'] = np.log1p(df['total_claim_amount'])
            
            # Amount categories
            df['amount_category'] = pd.cut(
                df['total_claim_amount'],
                bins=[0, 5000, 20000, 50000, 100000, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            
            # Round number indicator (potential red flag)
            df['is_round_amount'] = (df['total_claim_amount'] % 1000 == 0).astype(int)
        
        # Time-based features
        if 'incident_hour_of_the_day' in df.columns:
            # Hour categories
            df['hour_category'] = pd.cut(
                df['incident_hour_of_the_day'],
                bins=[-1, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
            
            # Unusual hours flag (late night/early morning)
            df['unusual_hour'] = ((df['incident_hour_of_the_day'] >= 22) | 
                                 (df['incident_hour_of_the_day'] <= 5)).astype(int)
        
        # Age-based features if available
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 25, 35, 50, 65, 100],
                labels=['Young', 'Young Adult', 'Middle Age', 'Senior', 'Elderly']
            )
            
            # Young driver flag (higher risk)
            df['young_driver'] = (df['age'] < 25).astype(int)
        
        # Severity encoding if available
        if 'incident_severity' in df.columns:
            severity_map = {'Minor Damage': 1, 'Major Damage': 2, 'Total Loss': 3}
            df['severity_numeric'] = df['incident_severity'].map(severity_map).fillna(1)
        
        # State risk encoding if available
        if 'incident_state' in df.columns:
            # Some states might have higher fraud rates (simplified assumption)
            high_risk_states = ['NY', 'CA', 'FL', 'TX']
            df['high_risk_state'] = df['incident_state'].isin(high_risk_states).astype(int)
        
        # Benford's Law features
        if 'total_claim_amount' in df.columns:
            # First digit analysis
            df['first_digit'] = df['total_claim_amount'].astype(str).str[0].astype(int)
            
            # Expected vs actual first digit distribution (Benford's Law)
            benford_expected = {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
                              6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}
            
            # Calculate Benford score (how much it deviates from expected)
            first_digit_counts = df['first_digit'].value_counts(normalize=True)
            benford_score = 0
            for digit in range(1, 10):
                expected = benford_expected.get(digit, 0)
                actual = first_digit_counts.get(digit, 0)
                benford_score += abs(expected - actual)
            
            df['benford_anomaly_score'] = benford_score
        
        # Interaction features
        if 'age' in df.columns and 'total_claim_amount' in df.columns:
            df['amount_per_age'] = df['total_claim_amount'] / df['age']
        
        if 'witnesses' in df.columns and 'total_claim_amount' in df.columns:
            # Claims with no witnesses might be more suspicious for high amounts
            df['high_amount_no_witnesses'] = ((df['total_claim_amount'] > 50000) & 
                                             (df['witnesses'] == 0)).astype(int)
        
        return df
    
    def get_numeric_features(self, df):
        """Get numeric features for ML models."""
        numeric_features = []
        
        # Always include basic features
        if 'total_claim_amount' in df.columns:
            numeric_features.extend(['total_claim_amount', 'log_claim_amount', 'is_round_amount'])
        
        if 'incident_hour_of_the_day' in df.columns:
            numeric_features.extend(['incident_hour_of_the_day', 'unusual_hour'])
        
        # Optional features
        optional_numeric = ['age', 'young_driver', 'severity_numeric', 'high_risk_state',
                           'benford_anomaly_score', 'amount_per_age', 'high_amount_no_witnesses',
                           'witnesses', 'first_digit']
        
        for feature in optional_numeric:
            if feature in df.columns:
                numeric_features.append(feature)
        
        return [f for f in numeric_features if f in df.columns]
    
    def validate_data_quality(self, df):
        """Validate data quality and return quality metrics."""
        quality_report = {
            'total_rows': len(df),
            'missing_values': {},
            'outliers': {},
            'data_types': {},
            'quality_score': 0
        }
        
        # Check missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            quality_report['missing_values'][col] = missing_pct
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            quality_report['outliers'][col] = (outliers / len(df)) * 100
        
        # Data types
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
        
        # Calculate overall quality score
        avg_missing = np.mean(list(quality_report['missing_values'].values()))
        avg_outliers = np.mean(list(quality_report['outliers'].values())) if quality_report['outliers'] else 0
        
        quality_score = max(0.0, 100.0 - avg_missing - (avg_outliers * 0.5))
        quality_report['quality_score'] = quality_score
        
        return quality_report
