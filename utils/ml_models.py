"""Machine learning model management for the Obzerra fraud engine."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

class MLModelManager:
    """Manages machine learning models for fraud detection."""
    
    def __init__(self):
        self.logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained_flag = False
        self.feature_columns = []
        self.model_metrics = {}
        self.feature_importance = pd.DataFrame()
        
    def train_models(self, training_data):
        """Train the ensemble of ML models."""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if X is None or len(X) < 50 or (y is not None and y.sum() < 5):  # Need minimum samples and positive cases
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Apply SMOTE to handle imbalanced data
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Train models
            self.logistic_model.fit(X_train_balanced, y_train_balanced)
            self.rf_model.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate models
            self._evaluate_models(X_test_scaled, y_test)
            
            # Store feature importance
            self._calculate_feature_importance()
            
            self.is_trained_flag = True
            return True
            
        except Exception as err:
            logger.exception("Model training failed")
            return False
    
    def predict_batch(self, data):
        """Make predictions on a batch of data."""
        if not self.is_trained_flag:
            return np.full(len(data), 0.5)  # Return neutral probability
        
        try:
            X = self._prepare_prediction_data(data)
            if X is None or len(X) == 0:
                return np.full(len(data), 0.5)
            
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from both models
            lr_probs = self.logistic_model.predict_proba(X_scaled)
            rf_probs = self.rf_model.predict_proba(X_scaled)
            
            # Extract positive class probabilities
            lr_probs = lr_probs[:, 1] if lr_probs.shape[1] > 1 else lr_probs[:, 0]
            rf_probs = rf_probs[:, 1] if rf_probs.shape[1] > 1 else rf_probs[:, 0]
            
            # Ensemble prediction (weighted average)
            ensemble_probs = 0.4 * lr_probs + 0.6 * rf_probs
            
            return ensemble_probs
            
        except Exception as err:
            logger.exception("Batch prediction failed")
            return np.full(len(data), 0.5)
    
    def predict_single(self, data):
        """Make prediction on a single claim."""
        if not self.is_trained_flag:
            return 0.5
        
        try:
            # Convert single claim to DataFrame if needed
            if isinstance(data, pd.Series):
                data = data.to_frame().T
            elif isinstance(data, dict):
                data = pd.DataFrame([data])
            
            X = self._prepare_prediction_data(data)
            if X is None or len(X) == 0:
                return 0.5
            
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from both models
            lr_proba = self.logistic_model.predict_proba(X_scaled)
            rf_proba = self.rf_model.predict_proba(X_scaled)
            
            # Extract positive class probabilities
            lr_prob = lr_proba[0, 1] if lr_proba.shape[1] > 1 else lr_proba[0, 0]
            rf_prob = rf_proba[0, 1] if rf_proba.shape[1] > 1 else rf_proba[0, 0]
            
            # Ensemble prediction
            ensemble_prob = 0.4 * lr_prob + 0.6 * rf_prob
            
            return float(ensemble_prob)
            
        except Exception as err:
            logger.exception("Single prediction failed")
            return 0.5
    
    def _prepare_training_data(self, data):
        """Prepare data for training."""
        # Create synthetic labels based on risk scores for training
        # In a real system, you would use actual fraud labels
        X_features = self._get_feature_columns(data)
        
        if X_features.empty:
            return None, None
        
        # Create labels based on risk scores and triggered rules
        # High risk (>70) and multiple rules triggered = likely fraud
        y_labels = (
            (data['risk_score'] > 70) & 
            (data['triggered_rules'].str.count(',') >= 2)
        ).astype(int)
        
        # Ensure we have some positive cases
        if y_labels.sum() == 0:
            # If no high-risk cases, create some based on highest scores
            high_score_threshold = data['risk_score'].quantile(0.9)
            y_labels = (data['risk_score'] > high_score_threshold).astype(int)
        
        self.feature_columns = X_features.columns.tolist()
        return X_features, y_labels
    
    def _prepare_prediction_data(self, data):
        """Prepare data for prediction."""
        X_features = self._get_feature_columns(data)
        
        if X_features.empty:
            return None
        
        # Ensure we have the same features as training
        missing_features = set(self.feature_columns) - set(X_features.columns)
        for feature in missing_features:
            X_features[feature] = 0  # Fill missing features with 0
        
        # Reorder columns to match training
        X_features = X_features[self.feature_columns]
        
        return X_features
    
    def _get_feature_columns(self, data):
        """Extract relevant features for ML models."""
        numeric_features = []
        
        # Core features
        core_features = ['total_claim_amount', 'incident_hour_of_the_day', 'log_claim_amount']
        for feature in core_features:
            if feature in data.columns:
                numeric_features.append(feature)
        
        # Additional features
        additional_features = [
            'age', 'unusual_hour', 'is_round_amount', 'young_driver',
            'severity_numeric', 'high_risk_state', 'benford_anomaly_score',
            'amount_per_age', 'high_amount_no_witnesses', 'witnesses', 'first_digit'
        ]
        
        for feature in additional_features:
            if feature in data.columns:
                numeric_features.append(feature)
        
        # Select only numeric features that exist
        available_features = [f for f in numeric_features if f in data.columns]
        
        if not available_features:
            return pd.DataFrame()
        
        return data[available_features].select_dtypes(include=[np.number]).fillna(0)
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate model performance."""
        try:
            # Logistic Regression predictions
            lr_pred = self.logistic_model.predict(X_test)
            lr_proba_matrix = self.logistic_model.predict_proba(X_test)
            lr_probs = lr_proba_matrix[:, 1] if lr_proba_matrix.shape[1] > 1 else lr_proba_matrix[:, 0]
            
            # Random Forest predictions
            rf_pred = self.rf_model.predict(X_test)
            rf_proba_matrix = self.rf_model.predict_proba(X_test)
            rf_probs = rf_proba_matrix[:, 1] if rf_proba_matrix.shape[1] > 1 else rf_proba_matrix[:, 0]
            
            # Ensemble predictions
            ensemble_probs = 0.4 * lr_probs + 0.6 * rf_probs
            ensemble_pred = (ensemble_probs > 0.5).astype(int)
            
            # Calculate metrics for ensemble
            self.model_metrics = {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred, zero_division='warn'),
                'recall': recall_score(y_test, ensemble_pred, zero_division='warn'),
                'f1': f1_score(y_test, ensemble_pred, zero_division='warn'),
                'auc': roc_auc_score(y_test, ensemble_probs) if len(np.unique(y_test)) > 1 else 0.5,
                'n_samples': len(y_test)
            }
            
        except Exception as err:
            logger.exception("Model evaluation failed")
            self.model_metrics = {
                'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5,
                'f1': 0.5, 'auc': 0.5, 'n_samples': 0
            }
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        try:
            if hasattr(self.rf_model, 'feature_importances_') and len(self.feature_columns) > 0:
                importance_data = {
                    'feature': self.feature_columns,
                    'importance': self.rf_model.feature_importances_
                }
                self.feature_importance = pd.DataFrame(importance_data)
                self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
            
        except Exception as err:
            logger.exception("Feature importance calculation failed")
            self.feature_importance = pd.DataFrame()
    
    def is_trained(self):
        """Check if models are trained."""
        return self.is_trained_flag
    
    def get_model_accuracy(self):
        """Get model accuracy."""
        return self.model_metrics.get('accuracy', 0.5)
    
    def get_model_metrics(self):
        """Get all model metrics."""
        return self.model_metrics
    
    def get_feature_importance(self):
        """Get feature importance DataFrame."""
        return self.feature_importance
    
    def save_models(self, filepath):
        """Save trained models to file."""
        if self.is_trained_flag:
            model_data = {
                'logistic_model': self.logistic_model,
                'rf_model': self.rf_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_metrics': self.model_metrics,
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_data, filepath)
    
    def load_models(self, filepath):
        """Load trained models from file."""
        try:
            model_data = joblib.load(filepath)
            self.logistic_model = model_data['logistic_model']
            self.rf_model = model_data['rf_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data['model_metrics']
            self.feature_importance = model_data['feature_importance']
            self.is_trained_flag = True
            return True
        except Exception as err:
            logger.exception("Model loading failed")
            return False
