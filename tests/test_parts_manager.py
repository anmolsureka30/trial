import unittest
import os
import pandas as pd
from src.data_management.parts_manager import PartsManager

class TestPartsManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = "test_parts.db"
        self.parts_manager = PartsManager(db_path=self.test_db_path)
        
        # Create test CSV
        self.test_csv_path = "test_parts.csv"
        test_data = {
            'Product': ['Private Car'] * 3,
            'Surveyor Part Code': ['1001', '1002', '1003'],
            'Surveyor Part Name': ['Bumper Front Assembly', 'Bonnet|Hood Assembly', 'Windshield Glass Front']
        }
        pd.DataFrame(test_data).to_csv(self.test_csv_path, index=False)
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_load_parts_catalog(self):
        """Test loading parts catalog"""
        self.parts_manager.load_parts_catalog(self.test_csv_path)
        part = self.parts_manager.get_part_by_code('1001')
        self.assertIsNotNone(part)
        self.assertEqual(part['part_name'], 'Bumper Front Assembly')

    def test_fuzzy_matching(self):
        """Test fuzzy matching functionality"""
        self.parts_manager.load_parts_catalog(self.test_csv_path)
        
        # Test exact match
        match = self.parts_manager.find_matching_part("Bumper Front Assembly")
        self.assertIsNotNone(match)
        self.assertEqual(match['part_code'], '1001')
        
        # Test fuzzy match
        match = self.parts_manager.find_matching_part("Front Bumper")
        self.assertIsNotNone(match)
        self.assertEqual(match['part_code'], '1001')
        
        # Test no match
        match = self.parts_manager.find_matching_part("NonexistentPart")
        self.assertIsNone(match)

if __name__ == '__main__':
    unittest.main() 