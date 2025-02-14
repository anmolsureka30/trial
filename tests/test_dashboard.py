import unittest
import pandas as pd
from src.dashboard.app import DashboardApp
from datetime import datetime, timedelta

class TestDashboard(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.dashboard = DashboardApp()
        
    def test_metrics_retrieval(self):
        """Test basic metrics retrieval"""
        self.assertIsInstance(self.dashboard._get_active_claims_count(), int)
        self.assertIsInstance(self.dashboard._get_fraud_alerts_count(), int)
        self.assertIsInstance(self.dashboard._get_avg_processing_time(), float)
        self.assertIsInstance(self.dashboard._get_success_rate(), float)
        
    def test_date_range_validation(self):
        """Test date range validation for claims analysis"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # This should return a DataFrame
        claims_data = self.dashboard._get_claims_data(start_date, end_date)
        self.assertIsInstance(claims_data, pd.DataFrame)
        
    def test_risk_distribution(self):
        """Test risk distribution data retrieval"""
        risk_data = self.dashboard._get_risk_distribution()
        self.assertIsInstance(risk_data, pd.DataFrame)
        self.assertTrue('risk_level' in risk_data.columns)
        self.assertTrue('count' in risk_data.columns)

if __name__ == '__main__':
    unittest.main() 