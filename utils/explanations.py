import pandas as pd
import numpy as np
import re

class ExplanationEngine:
    """Converts technical ML outputs to plain-language explanations."""
    
    def __init__(self):
        self.explanation_templates = {
            'high_risk': [
                "This claim shows multiple red flags that are commonly associated with fraudulent activity.",
                "Several unusual patterns were detected that warrant further investigation.",
                "The claim exhibits characteristics that significantly deviate from normal patterns."
            ],
            'medium_risk': [
                "This claim shows some concerning patterns that require additional verification.",
                "While not immediately suspicious, certain aspects of this claim need closer examination.",
                "Some indicators suggest this claim may need additional documentation or review."
            ],
            'low_risk': [
                "This claim appears to follow normal patterns with minimal fraud indicators.",
                "The claim shows standard characteristics consistent with legitimate insurance claims.",
                "Most fraud detection checks passed with only minor concerns noted."
            ]
        }
        
        self.reason_explanations = {
            'Z-Score Outlier': [
                "The claim amount is unusually high or low compared to similar cases",
                "Statistical analysis shows this amount is significantly different from the norm",
                "The claim value falls outside typical ranges for similar incidents"
            ],
            'Benford Anomaly': [
                "The claim amount doesn't follow natural number patterns",
                "The first digits of amounts in this claim are statistically unusual",
                "Number patterns suggest potential manipulation of claim values"
            ],
            'Unusual Hour': [
                "The incident occurred at a time when fraud is more commonly reported",
                "Late night or early morning incidents are flagged for additional review",
                "The timing of this incident matches patterns seen in fraudulent claims"
            ],
            'Round Amount': [
                "The claim amount is suspiciously round (like exactly ₱50,000)",
                "Legitimate claims rarely result in perfectly round numbers",
                "The precise amount suggests it may have been predetermined"
            ],
            'High Amount': [
                "The claim amount exceeds normal thresholds for this type of incident",
                "Large claims require additional scrutiny to prevent fraud",
                "The requested amount is significantly higher than typical cases"
            ],
            'Young High Claim': [
                "Young drivers with high-value claims require additional verification",
                "The combination of driver age and claim amount raises concerns",
                "Statistical data shows higher fraud rates in this demographic for large claims"
            ],
            'No Witnesses High': [
                "High-value claims without witnesses are more likely to be fraudulent",
                "The lack of independent verification for such a large claim is concerning",
                "Claims with no witnesses and high amounts need thorough investigation"
            ],
            'Duplicate Amount': [
                "Similar claim amounts were found in the same batch, suggesting coordination",
                "Multiple claims with identical amounts may indicate organized fraud",
                "The pattern of matching amounts across claims requires investigation"
            ]
        }
        
        self.action_recommendations = {
            'high_risk': {
                'action': 'ESCALATE',
                'description': 'Immediate investigation required',
                'steps': [
                    'Assign to senior fraud investigator',
                    'Request additional documentation',
                    'Verify incident details with authorities',
                    'Consider claim suspension pending investigation',
                    'Review claimant history for patterns'
                ]
            },
            'medium_risk': {
                'action': 'VERIFY',
                'description': 'Additional verification recommended',
                'steps': [
                    'Contact claimant for additional information',
                    'Verify incident details with authorities',
                    'Review supporting documentation carefully',
                    'Check for similar recent claims',
                    'Consider field inspection if warranted'
                ]
            },
            'low_risk': {
                'action': 'APPROVE',
                'description': 'Standard processing recommended',
                'steps': [
                    'Process claim through normal workflow',
                    'Routine documentation review',
                    'Standard settlement procedures',
                    'Monitor for any additional red flags'
                ]
            }
        }
    
    def add_explanations(self, results_df):
        """Add plain-language explanations to batch results."""
        results = results_df.copy()
        
        # Add primary reasons
        results['reason_1'] = results.apply(self._get_primary_reason, axis=1)
        results['reason_2'] = results.apply(self._get_secondary_reason, axis=1)
        
        # Add detailed explanations
        results['explanation'] = results.apply(self._generate_explanation, axis=1)
        
        # Clean up triggered rules display
        results['triggered_rules'] = results['triggered_rules'].str.rstrip(', ')
        
        return results
    
    def add_single_explanation(self, result):
        """Add explanation to a single claim result."""
        # Get primary and secondary reasons
        result['reason_1'] = self._get_primary_reason(result)
        result['reason_2'] = self._get_secondary_reason(result)
        
        # Generate detailed explanation
        result['explanation'] = self._generate_explanation(result)
        
        # Clean up triggered rules
        if 'triggered_rules' in result:
            result['triggered_rules'] = result['triggered_rules'].rstrip(', ')
        
        return result
    
    def _get_primary_reason(self, row):
        """Get the primary reason for the fraud score."""
        triggered_rules = str(row.get('triggered_rules', ''))
        
        if not triggered_rules or triggered_rules == 'nan':
            return "Standard risk assessment based on claim characteristics"
        
        # Split rules and get the first one
        rules = [rule.strip() for rule in triggered_rules.split(',') if rule.strip()]
        
        if not rules:
            return "Standard risk assessment based on claim characteristics"
        
        primary_rule = rules[0]
        
        # Convert to user-friendly explanation
        explanations = self.reason_explanations.get(primary_rule, [
            f"Analysis detected: {primary_rule.lower()}"
        ])
        
        return np.random.choice(explanations)
    
    def _get_secondary_reason(self, row):
        """Get the secondary reason for the fraud score."""
        triggered_rules = str(row.get('triggered_rules', ''))
        
        if not triggered_rules or triggered_rules == 'nan':
            return "All standard verification checks completed"
        
        rules = [rule.strip() for rule in triggered_rules.split(',') if rule.strip()]
        
        if len(rules) < 2:
            # If only one rule, provide a generic secondary reason
            return "Additional risk factors considered in overall assessment"
        
        secondary_rule = rules[1]
        
        explanations = self.reason_explanations.get(secondary_rule, [
            f"Secondary concern: {secondary_rule.lower()}"
        ])
        
        return np.random.choice(explanations)
    
    def _generate_explanation(self, row):
        """Generate a comprehensive explanation for the fraud assessment."""
        risk_band = str(row.get('risk_band', 'Low')).lower()
        risk_score = row.get('risk_score', 0)
        triggered_rules = str(row.get('triggered_rules', ''))
        
        # Base explanation based on risk level
        base_explanations = self.explanation_templates.get(f"{risk_band}_risk", 
                                                          self.explanation_templates['low_risk'])
        base_explanation = np.random.choice(base_explanations)
        
        # Add specific details
        details = []
        
        # Risk score context
        if risk_score >= 80:
            details.append("The risk score of {}/100 indicates very high fraud likelihood.".format(risk_score))
        elif risk_score >= 60:
            details.append("The risk score of {}/100 suggests elevated fraud risk.".format(risk_score))
        elif risk_score >= 30:
            details.append("The risk score of {}/100 indicates moderate concern.".format(risk_score))
        else:
            details.append("The risk score of {}/100 is within normal ranges.".format(risk_score))
        
        # Rule-specific details
        if triggered_rules and triggered_rules != 'nan':
            rules = [rule.strip() for rule in triggered_rules.split(',') if rule.strip()]
            if len(rules) == 1:
                details.append("One fraud indicator was detected during analysis.")
            elif len(rules) > 1:
                details.append("{} fraud indicators were detected during analysis.".format(len(rules)))
        
        # ML model contribution if available
        if 'ml_fraud_prob' in row and row.get('ml_fraud_prob', 0) > 0:
            ml_prob = row['ml_fraud_prob']
            if ml_prob > 0.7:
                details.append("Machine learning models strongly suggest fraudulent activity.")
            elif ml_prob > 0.4:
                details.append("Machine learning models indicate some concern.")
            else:
                details.append("Machine learning models support legitimate classification.")
        
        # Combine all parts
        explanation_parts = [base_explanation] + details
        full_explanation = " ".join(explanation_parts)
        
        return full_explanation
    
    def get_action_recommendation(self, risk_band):
        """Get action recommendation based on risk band."""
        return self.action_recommendations.get(risk_band.lower(), 
                                               self.action_recommendations['low_risk'])
    
    def explain_rule(self, rule_name):
        """Get detailed explanation for a specific rule."""
        explanations = self.reason_explanations.get(rule_name, [
            f"Rule: {rule_name} - No detailed explanation available"
        ])
        return np.random.choice(explanations)
    
    def generate_summary_insights(self, results_df):
        """Generate summary insights for a batch of results."""
        if len(results_df) == 0:
            return "No claims were analyzed."
        
        total_claims = len(results_df)
        high_risk = len(results_df[results_df['risk_band'] == 'High'])
        medium_risk = len(results_df[results_df['risk_band'] == 'Medium'])
        low_risk = len(results_df[results_df['risk_band'] == 'Low'])
        
        avg_risk_score = results_df['risk_score'].mean()
        
        # Most common rules
        all_rules = []
        for rules_str in results_df['triggered_rules']:
            if pd.notna(rules_str):
                all_rules.extend([rule.strip() for rule in str(rules_str).split(',') if rule.strip()])
        
        rule_counts = pd.Series(all_rules).value_counts()
        top_rules = rule_counts.head(3).index.tolist()
        
        insights = []
        insights.append(f"Analyzed {total_claims} claims with an average risk score of {avg_risk_score:.1f}/100.")
        
        if high_risk > 0:
            insights.append(f"{high_risk} claims ({high_risk/total_claims*100:.1f}%) require immediate investigation.")
        
        if medium_risk > 0:
            insights.append(f"{medium_risk} claims ({medium_risk/total_claims*100:.1f}%) need additional verification.")
        
        insights.append(f"{low_risk} claims ({low_risk/total_claims*100:.1f}%) can proceed with standard processing.")
        
        if top_rules:
            top_rules_str = [str(rule) for rule in top_rules[:2]]
            insights.append(f"Most common fraud indicators: {', '.join(top_rules_str)}.")
        
        return " ".join(insights)
