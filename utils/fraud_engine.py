import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FraudEngine:
    """Rule-based fraud detection engine with statistical analysis."""
    
    def __init__(self):
        self.rules = {
            'z_score_outlier': {'weight': 0.2, 'threshold': 2.5},
            'benford_anomaly': {'weight': 0.15, 'threshold': 0.3},
            'unusual_hour': {'weight': 0.1, 'threshold': 1},
            'round_amount': {'weight': 0.1, 'threshold': 1},
            'high_amount': {'weight': 0.15, 'threshold': 100000},
            'young_high_claim': {'weight': 0.1, 'threshold': 1},
            'no_witnesses_high': {'weight': 0.1, 'threshold': 1},
            'frequency_flag': {'weight': 0.1, 'threshold': 1}
        }
    
    def analyze_batch(self, df):
        """Analyze a batch of claims for fraud indicators."""
        results = df.copy()
        
        # Initialize fraud scoring
        results['fraud_score'] = 0.0
        results['triggered_rules'] = ''
        results['rule_details'] = ''
        
        # Apply each rule
        results = self._apply_z_score_rule(results)
        results = self._apply_benford_rule(results)
        results = self._apply_unusual_hour_rule(results)
        results = self._apply_round_amount_rule(results)
        results = self._apply_high_amount_rule(results)
        results = self._apply_young_high_claim_rule(results)
        results = self._apply_no_witnesses_rule(results)
        results = self._apply_frequency_rule(results)
        
        # Calculate final risk score (0-100)
        results['risk_score'] = (results['fraud_score'] * 100).clip(0, 100).round(0).astype(int)
        
        # Assign risk bands
        results['risk_band'] = pd.cut(
            results['risk_score'],
            bins=[-1, 30, 70, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        # Initial fraud prediction based on rules
        results['fraud_prediction'] = (results['risk_score'] >= 70).map({True: 'Y', False: 'N'})
        
        return results
    
    def analyze_single_claim(self, claim_data):
        """Analyze a single claim for fraud indicators."""
        result = claim_data.copy()
        
        # Convert to DataFrame if it's a Series
        if isinstance(result, pd.Series):
            result = result.to_frame().T
        
        # Apply fraud detection
        result = self.analyze_batch(result)
        
        # Return as dictionary for single claim
        if len(result) == 1:
            return result.iloc[0].to_dict()
        
        return result
    
    def combine_predictions(self, rule_results, ml_predictions):
        """Combine rule-based and ML predictions."""
        combined = rule_results.copy()
        
        # Weighted combination: 60% ML, 40% rules
        ml_weight = 0.6
        rule_weight = 0.4
        
        # Ensure ml_predictions has the same index
        if len(ml_predictions) == len(combined):
            combined['ml_fraud_prob'] = ml_predictions
            combined['combined_score'] = (
                rule_weight * (combined['risk_score'] / 100) + 
                ml_weight * combined['ml_fraud_prob']
            )
            
            # Update risk score and bands
            combined['risk_score'] = (combined['combined_score'] * 100).clip(0, 100).round(0).astype(int)
            combined['risk_band'] = pd.cut(
                combined['risk_score'],
                bins=[-1, 30, 70, 100],
                labels=['Low', 'Medium', 'High']
            )
            
            # Update fraud prediction
            combined['fraud_prediction'] = (combined['risk_score'] >= 70).map({True: 'Y', False: 'N'})
        
        return combined
    
    def combine_single_prediction(self, rule_result, ml_prediction):
        """Combine single claim rule and ML predictions."""
        if isinstance(ml_prediction, (list, np.ndarray)):
            ml_prediction = ml_prediction[0] if len(ml_prediction) > 0 else 0.5
        
        # Weighted combination
        ml_weight = 0.6
        rule_weight = 0.4
        
        rule_result['ml_fraud_prob'] = ml_prediction
        rule_result['combined_score'] = (
            rule_weight * (rule_result['risk_score'] / 100) + 
            ml_weight * ml_prediction
        )
        
        # Update risk score
        rule_result['risk_score'] = int((rule_result['combined_score'] * 100).clip(0, 100))
        
        # Update risk band
        if rule_result['risk_score'] < 30:
            rule_result['risk_band'] = 'Low'
        elif rule_result['risk_score'] < 70:
            rule_result['risk_band'] = 'Medium'
        else:
            rule_result['risk_band'] = 'High'
        
        # Update fraud prediction
        rule_result['fraud_prediction'] = 'Y' if rule_result['risk_score'] >= 70 else 'N'
        
        return rule_result
    
    def _apply_z_score_rule(self, df):
        """Apply Z-score outlier detection for claim amounts."""
        if 'total_claim_amount' in df.columns and len(df) > 1:
            try:
                z_scores = np.abs(stats.zscore(df['total_claim_amount'].astype(float)))
                threshold = self.rules['z_score_outlier']['threshold']
                weight = self.rules['z_score_outlier']['weight']
                
                outliers = z_scores > threshold
                df.loc[outliers, 'fraud_score'] += weight
                
                # Track triggered rules
                df.loc[outliers, 'triggered_rules'] += 'Z-Score Outlier, '
                if outliers.any():
                    max_z = float(z_scores[outliers].max())
                    df.loc[outliers, 'rule_details'] += f'Amount is {max_z:.1f} std devs from mean. '
            except Exception:
                pass  # Skip z-score if calculation fails
        
        return df
    
    def _apply_benford_rule(self, df):
        """Apply Benford's Law analysis."""
        if 'benford_anomaly_score' in df.columns:
            threshold = self.rules['benford_anomaly']['threshold']
            weight = self.rules['benford_anomaly']['weight']
            
            anomalies = df['benford_anomaly_score'] > threshold
            df.loc[anomalies, 'fraud_score'] += weight
            
            df.loc[anomalies, 'triggered_rules'] += 'Benford Anomaly, '
            df.loc[anomalies, 'rule_details'] += 'First digit distribution unusual. '
        
        return df
    
    def _apply_unusual_hour_rule(self, df):
        """Flag claims at unusual hours."""
        if 'unusual_hour' in df.columns:
            weight = self.rules['unusual_hour']['weight']
            
            unusual = df['unusual_hour'] == 1
            df.loc[unusual, 'fraud_score'] += weight
            
            df.loc[unusual, 'triggered_rules'] += 'Unusual Hour, '
            df.loc[unusual, 'rule_details'] += 'Incident occurred at suspicious time. '
        
        return df
    
    def _apply_round_amount_rule(self, df):
        """Flag suspiciously round amounts."""
        if 'is_round_amount' in df.columns:
            weight = self.rules['round_amount']['weight']
            
            round_amounts = df['is_round_amount'] == 1
            df.loc[round_amounts, 'fraud_score'] += weight
            
            df.loc[round_amounts, 'triggered_rules'] += 'Round Amount, '
            df.loc[round_amounts, 'rule_details'] += 'Claim amount is suspiciously round. '
        
        return df
    
    def _apply_high_amount_rule(self, df):
        """Flag unusually high claim amounts."""
        if 'total_claim_amount' in df.columns:
            threshold = self.rules['high_amount']['threshold']
            weight = self.rules['high_amount']['weight']
            
            high_amounts = df['total_claim_amount'] > threshold
            df.loc[high_amounts, 'fraud_score'] += weight
            
            df.loc[high_amounts, 'triggered_rules'] += 'High Amount, '
            df.loc[high_amounts, 'rule_details'] += f'Claim exceeds ₱{threshold:,}. '
        
        return df
    
    def _apply_young_high_claim_rule(self, df):
        """Flag high claims from young drivers."""
        if 'young_driver' in df.columns and 'total_claim_amount' in df.columns:
            weight = self.rules['young_high_claim']['weight']
            
            young_high = (df['young_driver'] == 1) & (df['total_claim_amount'] > 50000)
            df.loc[young_high, 'fraud_score'] += weight
            
            df.loc[young_high, 'triggered_rules'] += 'Young High Claim, '
            df.loc[young_high, 'rule_details'] += 'High claim amount for young driver. '
        
        return df
    
    def _apply_no_witnesses_rule(self, df):
        """Flag high claims with no witnesses."""
        if 'high_amount_no_witnesses' in df.columns:
            weight = self.rules['no_witnesses_high']['weight']
            
            no_witness_high = df['high_amount_no_witnesses'] == 1
            df.loc[no_witness_high, 'fraud_score'] += weight
            
            df.loc[no_witness_high, 'triggered_rules'] += 'No Witnesses High, '
            df.loc[no_witness_high, 'rule_details'] += 'High amount claim with no witnesses. '
        
        return df
    
    def _apply_frequency_rule(self, df):
        """Apply frequency analysis (simplified for batch processing)."""
        if 'claim_id' in df.columns and len(df) > 1:
            weight = self.rules['frequency_flag']['weight']
            
            # Check for duplicate patterns (simplified)
            # In a real system, this would check against historical data
            duplicated_amounts = df.duplicated(subset=['total_claim_amount'], keep=False)
            
            df.loc[duplicated_amounts, 'fraud_score'] += weight * 0.5  # Reduced weight for batch
            df.loc[duplicated_amounts, 'triggered_rules'] += 'Duplicate Amount, '
            df.loc[duplicated_amounts, 'rule_details'] += 'Similar claim amount in batch. '
        
        return df
    
    def get_rule_explanations(self):
        """Get explanations for all fraud detection rules."""
        explanations = {
            'Z-Score Outlier': 'Claim amount is statistically unusual compared to other claims',
            'Benford Anomaly': 'First digit distribution does not follow natural patterns (Benford\'s Law)',
            'Unusual Hour': 'Incident occurred during suspicious hours (late night/early morning)',
            'Round Amount': 'Claim amount is suspiciously round (e.g., exactly ₱50,000)',
            'High Amount': 'Claim amount exceeds normal thresholds',
            'Young High Claim': 'High claim amount for a young driver (under 25)',
            'No Witnesses High': 'High amount claim with no witnesses present',
            'Duplicate Amount': 'Similar claim amounts detected in the same batch'
        }
        return explanations
    
    def get_rule_weights(self):
        """Get current rule weights for transparency."""
        return {rule: config['weight'] for rule, config in self.rules.items()}
