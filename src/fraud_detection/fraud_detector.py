import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sqlite3
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, db_path: str = "data/fraud_detection.db"):
        """Initialize the fraud detection system"""
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.setup_logging()
        self.initialize_database()
        try:
            self.setup_detector()
        except Exception as e:
            logger.error(f"Failed to initialize FraudDetector: {e}")
            raise
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/fraud_detection.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_database(self):
        """Initialize SQLite database for fraud detection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create alerts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fraud_alerts (
                        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        claim_id TEXT,
                        risk_score FLOAT,
                        alert_type TEXT,
                        description TEXT,
                        created_at TIMESTAMP,
                        status TEXT DEFAULT 'pending'
                    )
                """)
                
                # Create patterns table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fraud_patterns (
                        pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_type TEXT,
                        pattern_value TEXT,
                        risk_level FLOAT,
                        detection_count INTEGER,
                        last_updated TIMESTAMP
                    )
                """)
                
                # Create indices
                conn.execute("CREATE INDEX IF NOT EXISTS idx_claim_id ON fraud_alerts(claim_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON fraud_patterns(pattern_type)")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
            
    def setup_detector(self):
        """Setup fraud detection components"""
        # Initialize basic components
        self.risk_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }

    def train_anomaly_detector(self, historical_data: pd.DataFrame, model_path: str = "models/fraud_detector.joblib"):
        """Train the anomaly detection model"""
        try:
            # Prepare features for anomaly detection
            features = self._prepare_features(historical_data)
            
            # Initialize and train IsolationForest
            self.model = IsolationForest(
                contamination=0.1,  # Expected proportion of anomalies
                random_state=42,
                n_estimators=100
            )
            
            # Fit the model
            self.model.fit(features)
            
            # Save the model and scaler
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, model_path)
            
            self.logger.info("Anomaly detection model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training anomaly detection model: {str(e)}")
            raise
            
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        # Select relevant numerical features
        features = [
            'claim_amount',
            'parts_count',
            'repair_duration',
            'previous_claims_count',
            'claim_frequency'
        ]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(data[features])
        
        return scaled_features
        
    def detect_fraud(self, claim_data: Dict) -> Dict:
        """Detect potential fraud in a claim"""
        try:
            # Prepare input data
            input_features = self._prepare_input(claim_data)
            
            # Get anomaly score
            anomaly_score = self.model.score_samples(input_features)[0]
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(anomaly_score)
            
            # Check for suspicious patterns
            pattern_alerts = self._check_patterns(claim_data)
            
            # Generate final assessment
            assessment = {
                'claim_id': claim_data.get('claim_id'),
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'alerts': pattern_alerts,
                'anomaly_score': float(anomaly_score),
                'timestamp': datetime.now().isoformat(),
                'requires_investigation': risk_score > 70 or len(pattern_alerts) > 2
            }
            
            # Store alert if risk is high
            if assessment['requires_investigation']:
                self._store_alert(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in fraud detection: {str(e)}")
            return None
            
    def _prepare_input(self, claim_data: Dict) -> np.ndarray:
        """Prepare input data for prediction"""
        input_data = pd.DataFrame([{
            'claim_amount': claim_data.get('claim_amount', 0),
            'parts_count': claim_data.get('parts_count', 0),
            'repair_duration': claim_data.get('repair_duration', 0),
            'previous_claims_count': claim_data.get('previous_claims_count', 0),
            'claim_frequency': claim_data.get('claim_frequency', 0)
        }])
        
        return self.scaler.transform(input_data)
        
    def _calculate_risk_score(self, anomaly_score: float) -> float:
        """Convert anomaly score to risk score (0-100)"""
        # Convert anomaly score to risk score
        # Lower anomaly scores indicate higher risk
        normalized_score = 1 / (1 + np.exp(anomaly_score))
        risk_score = normalized_score * 100
        
        return round(risk_score, 2)
        
    def _check_patterns(self, claim_data: Dict) -> List[Dict]:
        """Check for suspicious patterns in the claim"""
        alerts = []
        
        # Check for multiple claims in short period
        if claim_data.get('claim_frequency', 0) > 3:
            alerts.append({
                'type': 'high_frequency',
                'description': 'Multiple claims in short period',
                'severity': 'high'
            })
            
        # Check for unusually high claim amount
        if claim_data.get('claim_amount', 0) > 5000:
            alerts.append({
                'type': 'high_amount',
                'description': 'Unusually high claim amount',
                'severity': 'medium'
            })
            
        # Check for suspicious repair duration
        if claim_data.get('repair_duration', 0) < 1:
            alerts.append({
                'type': 'suspicious_duration',
                'description': 'Unusually short repair duration',
                'severity': 'medium'
            })
            
        return alerts
        
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level based on risk score"""
        if risk_score >= 80:
            return 'critical'
        elif risk_score >= 60:
            return 'high'
        elif risk_score >= 40:
            return 'medium'
        else:
            return 'low'
            
    def _store_alert(self, assessment: Dict):
        """Store fraud alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO fraud_alerts 
                    (claim_id, risk_score, alert_type, description, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    assessment['claim_id'],
                    assessment['risk_score'],
                    assessment['risk_level'],
                    str(assessment['alerts']),
                    assessment['timestamp']
                ))
                
        except Exception as e:
            self.logger.error(f"Error storing fraud alert: {str(e)}")
            
    def get_historical_alerts(self, days: int = 7) -> pd.DataFrame:
        """Get historical fraud alerts"""
        try:
            # Generate sample data
            dates = pd.date_range(end=datetime.now(), periods=days)
            alerts = pd.DataFrame({
                'date': dates,
                'risk_score': np.random.uniform(0, 1, size=len(dates)),
                'alert_type': np.random.choice(['high', 'medium', 'low'], size=len(dates))
            })
            return alerts
        except Exception as e:
            logger.error(f"Error getting historical alerts: {e}")
            return pd.DataFrame() 