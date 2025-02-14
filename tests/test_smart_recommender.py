import unittest
import pandas as pd
import os
import numpy as np
from datetime import datetime
from src.recommendations.smart_recommender import SmartRecommender
import sqlite3

class TestSmartRecommender(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = "test_recommendations.db"
        self.recommender = SmartRecommender(db_path=self.test_db_path)
        
        # Create test historical data
        self.historical_data = pd.DataFrame({
            'claim_id': np.repeat(range(50), 2),  # 50 claims, 2 parts each
            'damaged_part_code': np.random.choice(['1001', '1002', '1003', '1004'], 100),
            'damage_severity': np.random.choice(['minor', 'moderate', 'severe', 'critical'], 100),
            'repair_cost': np.random.uniform(100, 1000, 100),
            'repair_date': [datetime.now().isoformat() for _ in range(100)]
        })
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
            
    def test_pattern_analysis(self):
        """Test damage pattern analysis"""
        self.recommender.update_damage_patterns(self.historical_data)
        
        # Test pattern retrieval
        with sqlite3.connect(self.test_db_path) as conn:
            patterns = pd.read_sql_query("SELECT * FROM damage_patterns", conn)
            self.assertTrue(len(patterns) > 0)
            
    def test_recommendations(self):
        """Test recommendation generation"""
        # Update patterns first
        self.recommender.update_damage_patterns(self.historical_data)
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations('1001', 'severe')
        
        self.assertIsNotNone(recommendations)
        self.assertIn('associated_parts', recommendations)
        self.assertIn('repair_suggestions', recommendations)
        self.assertIn('cost_estimates', recommendations)
        self.assertIn('priority_level', recommendations)
        
    def test_cost_estimates(self):
        """Test cost estimation functionality"""
        estimates = self.recommender._get_cost_estimates('1001', 'severe')
        
        self.assertIn('estimated_cost', estimates)
        self.assertIn('confidence_level', estimates)
        self.assertIn('includes_labor', estimates)
        self.assertIn('currency', estimates)
        
    def test_priority_calculation(self):
        """Test priority calculation"""
        priority = self.recommender._calculate_priority('1001', 'critical')
        
        self.assertIn('level', priority)
        self.assertIn('score', priority)
        self.assertIn('requires_immediate_action', priority)
        self.assertTrue(priority['requires_immediate_action'])

if __name__ == '__main__':
    unittest.main() 