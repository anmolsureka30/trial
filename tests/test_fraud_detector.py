import unittest
import pandas as pd
import os
import numpy as np
from datetime import datetime
from src.fraud_detection.fraud_detector import FraudDetector

class TestFraudDetector(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = "test_fraud_detection.db"
        self.test_model_path = "test_fraud_model.joblib"
        self.detector = FraudDetector(db_path=self.test_db_path)
        
        # Create test historical data
        self.historical_data = pd.DataFrame({
            'claim_id': [f'CLM{i:03d}' for i in range(100)],
            'claim_amount': np.random.uniform(100, 10000, 100),
            'parts_count': np.random.randint(1, 10, 100),
            'repair_duration': np.random.uniform(1, 30, 100),
            'previous_claims_count': np.random.randint(0, 5, 100),
            'claim_frequency': np.random.uniform(0, 5, 100)
        })
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
            
    def test_model_training(self):
        """Test anomaly detection model training"""
        self.detector.train_anomaly_detector(self.historical_data, self.test_model_path)
        self.assertTrue(os.path.exists(self.test_model_path))
        
    def test_fraud_detection(self):
        """Test fraud detection functionality"""
        # Train model first
        self.detector.train_anomaly_detector(self.historical_data, self.test_model_path)
        
        # Test detection
        test_claim = {
            'claim_id': 'TEST001',
            'claim_amount': 8000,
            'parts_count': 5,
            'repair_duration': 0.5,
            'previous_claims_count': 4,
            'claim_frequency': 4.5
        }
        
        result = self.detector.detect_fraud(test_claim)
        
        self.assertIsNotNone(result)
        self.assertIn('risk_score', result)
        self.assertIn('risk_level', result)
        self.assertIn('alerts', result)
        self.assertIn('requires_investigation', result)
        
    def test_pattern_checking(self):
        """Test suspicious pattern detection"""
        suspicious_claim = {
            'claim_frequency': 4,
            'claim_amount': 6000,
            'repair_duration': 0.5
        }
        
        alerts = self.detector._check_patterns(suspicious_claim)
        self.assertTrue(len(alerts) > 0)
        
    def test_risk_level_calculation(self):
        """Test risk level determination"""
        self.assertEqual(self.detector._get_risk_level(85), 'critical')
        self.assertEqual(self.detector._get_risk_level(65), 'high')
        self.assertEqual(self.detector._get_risk_level(45), 'medium')
        self.assertEqual(self.detector._get_risk_level(25), 'low')
        
    def test_historical_alerts(self):
        """Test historical alerts retrieval"""
        alerts = self.detector.get_historical_alerts(days=30)
        self.assertIsInstance(alerts, pd.DataFrame)

if __name__ == '__main__':
    unittest.main() 