import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import streamlit as st

class SessionManager:
    """Manages session state and local data persistence."""
    
    def __init__(self):
        self.session_key = 'fraud_detection_sessions'
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if 'total_claims_processed' not in st.session_state:
            st.session_state.total_claims_processed = 0
        
        if 'session_stats' not in st.session_state:
            st.session_state.session_stats = {
                'total_runtime': 0.0,
                'total_high_risk': 0,
                'session_start': datetime.now()
            }
    
    def store_analysis(self, results_df, runtime):
        """Store analysis results in session history."""
        analysis_record = {
            'timestamp': datetime.now(),
            'results': results_df.copy(),
            'runtime': runtime,
            'claims_count': len(results_df),
            'high_risk_count': len(results_df[results_df['risk_band'] == 'High']),
            'medium_risk_count': len(results_df[results_df['risk_band'] == 'Medium']),
            'low_risk_count': len(results_df[results_df['risk_band'] == 'Low']),
            'avg_risk_score': results_df['risk_score'].mean(),
            'fraud_predictions': len(results_df[results_df['fraud_prediction'] == 'Y'])
        }
        
        # Add to history
        st.session_state.analysis_history.append(analysis_record)
        
        # Update session stats
        st.session_state.total_claims_processed += len(results_df)
        st.session_state.session_stats['total_runtime'] += runtime
        st.session_state.session_stats['total_high_risk'] += analysis_record['high_risk_count']
        
        # Keep only last 50 analyses to prevent memory issues
        if len(st.session_state.analysis_history) > 50:
            st.session_state.analysis_history = st.session_state.analysis_history[-50:]
    
    def get_analysis_history(self):
        """Get the analysis history."""
        return st.session_state.analysis_history
    
    def get_recent_results(self, limit=10):
        """Get recent analysis results."""
        if not st.session_state.analysis_history:
            return []
        
        return st.session_state.analysis_history[-limit:]
    
    def get_total_claims(self):
        """Get total number of claims processed."""
        return st.session_state.total_claims_processed
    
    def get_high_risk_percentage(self):
        """Get percentage of high-risk claims."""
        if st.session_state.total_claims_processed == 0:
            return 0.0
        
        return (st.session_state.session_stats['total_high_risk'] / 
                st.session_state.total_claims_processed) * 100
    
    def get_avg_runtime(self):
        """Get average runtime per analysis."""
        history = st.session_state.analysis_history
        if not history:
            return 0.0
        
        total_runtime = sum(record['runtime'] for record in history)
        return total_runtime / len(history)
    
    def get_session_statistics(self):
        """Get comprehensive session statistics."""
        history = st.session_state.analysis_history
        
        if not history:
            return {
                'total_analyses': 0,
                'total_claims': 0,
                'avg_claims_per_analysis': 0,
                'total_runtime': 0,
                'avg_runtime': 0,
                'high_risk_rate': 0,
                'fraud_detection_rate': 0,
                'session_duration': 0
            }
        
        total_analyses = len(history)
        total_claims = sum(record['claims_count'] for record in history)
        total_runtime = sum(record['runtime'] for record in history)
        total_high_risk = sum(record['high_risk_count'] for record in history)
        total_fraud_predictions = sum(record['fraud_predictions'] for record in history)
        
        session_duration = (datetime.now() - st.session_state.session_stats['session_start']).total_seconds() / 3600
        
        return {
            'total_analyses': total_analyses,
            'total_claims': total_claims,
            'avg_claims_per_analysis': total_claims / total_analyses if total_analyses > 0 else 0,
            'total_runtime': total_runtime,
            'avg_runtime': total_runtime / total_analyses if total_analyses > 0 else 0,
            'high_risk_rate': (total_high_risk / total_claims * 100) if total_claims > 0 else 0,
            'fraud_detection_rate': (total_fraud_predictions / total_claims * 100) if total_claims > 0 else 0,
            'session_duration': session_duration
        }
    
    def get_time_series_data(self):
        """Get time series data for trends analysis."""
        history = st.session_state.analysis_history
        
        if not history:
            return pd.DataFrame()
        
        time_series_data = []
        for record in history:
            time_series_data.append({
                'timestamp': record['timestamp'],
                'claims_count': record['claims_count'],
                'high_risk_count': record['high_risk_count'],
                'avg_risk_score': record['avg_risk_score'],
                'runtime': record['runtime'],
                'fraud_predictions': record['fraud_predictions']
            })
        
        return pd.DataFrame(time_series_data)
    
    def clear_history(self):
        """Clear analysis history (for testing or reset purposes)."""
        st.session_state.analysis_history = []
        st.session_state.total_claims_processed = 0
        st.session_state.session_stats = {
            'total_runtime': 0.0,
            'total_high_risk': 0,
            'session_start': datetime.now()
        }
    
    def export_session_data(self):
        """Export session data for download."""
        session_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_statistics': self.get_session_statistics(),
            'analysis_history': []
        }
        
        # Convert analysis history to serializable format
        for record in st.session_state.analysis_history:
            serializable_record = {
                'timestamp': record['timestamp'].isoformat(),
                'claims_count': record['claims_count'],
                'high_risk_count': record['high_risk_count'],
                'medium_risk_count': record['medium_risk_count'],
                'low_risk_count': record['low_risk_count'],
                'avg_risk_score': float(record['avg_risk_score']),
                'fraud_predictions': record['fraud_predictions'],
                'runtime': record['runtime']
            }
            session_data['analysis_history'].append(serializable_record)
        
        return json.dumps(session_data, indent=2)
    
    def get_performance_metrics(self):
        """Get performance metrics for the session."""
        history = st.session_state.analysis_history
        
        if not history:
            return {}
        
        # Calculate various performance metrics
        runtimes = [record['runtime'] for record in history]
        claims_per_analysis = [record['claims_count'] for record in history]
        risk_scores = []
        
        for record in history:
            if 'results' in record:
                risk_scores.extend(record['results']['risk_score'].tolist())
        
        metrics = {
            'avg_runtime': np.mean(runtimes),
            'min_runtime': np.min(runtimes),
            'max_runtime': np.max(runtimes),
            'std_runtime': np.std(runtimes),
            'avg_claims_per_batch': np.mean(claims_per_analysis),
            'total_unique_claims': sum(claims_per_analysis),
            'avg_risk_score_overall': np.mean(risk_scores) if risk_scores else 0,
            'risk_score_std': np.std(risk_scores) if risk_scores else 0
        }
        
        return metrics
    
    def get_fraud_patterns(self):
        """Analyze patterns in fraud detection across sessions."""
        history = st.session_state.analysis_history
        
        if not history:
            return {}
        
        # Aggregate rule patterns
        all_triggered_rules = []
        all_risk_bands = []
        hourly_patterns = []
        amount_patterns = []
        
        for record in history:
            if 'results' in record:
                results = record['results']
                
                # Collect triggered rules
                for rules_str in results['triggered_rules']:
                    if pd.notna(rules_str):
                        rules = [rule.strip() for rule in str(rules_str).split(',') if rule.strip()]
                        all_triggered_rules.extend(rules)
                
                # Collect risk bands
                all_risk_bands.extend(results['risk_band'].tolist())
                
                # Collect time patterns if available
                if 'incident_hour_of_the_day' in results.columns:
                    hourly_patterns.extend(results['incident_hour_of_the_day'].tolist())
                
                # Collect amount patterns
                if 'total_claim_amount' in results.columns:
                    amount_patterns.extend(results['total_claim_amount'].tolist())
        
        patterns = {
            'most_common_rules': pd.Series(all_triggered_rules).value_counts().head(10).to_dict() if all_triggered_rules else {},
            'risk_distribution': pd.Series(all_risk_bands).value_counts().to_dict() if all_risk_bands else {},
            'peak_fraud_hours': pd.Series(hourly_patterns).value_counts().head(5).to_dict() if hourly_patterns else {},
            'amount_quartiles': {
                'q25': np.percentile(amount_patterns, 25) if amount_patterns else 0,
                'q50': np.percentile(amount_patterns, 50) if amount_patterns else 0,
                'q75': np.percentile(amount_patterns, 75) if amount_patterns else 0,
                'q95': np.percentile(amount_patterns, 95) if amount_patterns else 0
            } if amount_patterns else {}
        }
        
        return patterns
