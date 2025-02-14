import unittest
import pandas as pd
import os
import numpy as np
from src.damage_assessment.damage_classifier import DamageAssessment

class TestDamageAssessment(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_model_path = "test_damage_model.joblib"
        self.damage_assessor = DamageAssessment(model_path=self.test_model_path)
        
        # Create test training data
        self.training_data = pd.DataFrame({
            'part_age': np.random.randint(0, 10, 100),
            'impact_severity': np.random.randint(1, 5, 100),
            'material_type': np.random.choice(['metal', 'plastic', 'glass'], 100),
            'location_score': np.random.uniform(0, 1, 100),
            'previous_repairs': np.random.randint(0, 3, 100),
            'damage_severity': np.random.choice(['minor', 'moderate', 'severe', 'critical'], 100)
        })
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
            
    def test_model_training(self):
        """Test model training functionality"""
        self.damage_assessor.train_model(self.training_data)
        self.assertTrue(os.path.exists(self.test_model_path))
        
    def test_damage_assessment(self):
        """Test damage assessment functionality"""
        # Train model first
        self.damage_assessor.train_model(self.training_data)
        
        # Test assessment
        test_data = {
            'part_age': 2,
            'impact_severity': 3,
            'material_type': 'metal',
            'location_score': 0.7,
            'previous_repairs': 1,
            'part_code': '1001'
        }
        
        result = self.damage_assessor.assess_damage(test_data)
        
        self.assertIsNotNone(result)
        self.assertIn('damage_severity', result)
        self.assertIn('confidence', result)
        self.assertIn('estimated_cost', result)
        self.assertIn('recommendations', result)
        
    def test_cost_estimation(self):
        """Test cost estimation functionality"""
        cost_info = self.damage_assessor._estimate_repair_cost('moderate', '1001', 85.0)
        
        self.assertIn('base_cost', cost_info)
        self.assertIn('adjusted_cost', cost_info)
        self.assertIn('confidence_factor', cost_info)
        self.assertIn('currency', cost_info)
        
    def test_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.damage_assessor._generate_recommendations('severe', 90.0)
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)

if __name__ == '__main__':
    unittest.main() 