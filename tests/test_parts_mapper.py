import unittest
import pandas as pd
import numpy as np
from src.data_management.parts_mapper import PartsMapper
import os
import asyncio

class TestPartsMapper(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Use a test API key for testing
        self.api_key = os.getenv('GEMINI_API_KEY', 'test_key')
        self.mapper = PartsMapper(self.api_key)
        
        # Create test data
        self.surveyor_data = pd.DataFrame({
            'Surveyor Part Code': ['1001', '1002', '1003'],
            'Surveyor Part Name': [
                'Bumper Front Assembly',
                'Bonnet|Hood Assembly',
                'Windshield Glass Front'
            ]
        })
        
        self.garage_data = pd.DataFrame({
            'Garage Part Description': [
                'Front Bumper Complete',
                'Engine Hood',
                'Front Windscreen'
            ]
        })
        
    def test_embedding_generation(self):
        """Test embedding generation"""
        texts = ['test part one', 'test part two']
        embeddings = self.mapper._generate_embeddings(texts)
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(isinstance(embeddings, np.ndarray))
        
    def test_reference_data_loading(self):
        """Test reference data loading"""
        self.mapper.load_reference_data(self.surveyor_data, self.garage_data)
        self.assertIsNotNone(self.mapper.surveyor_embeddings)
        self.assertIsNotNone(self.mapper.garage_embeddings)
        
    async def test_part_mapping(self):
        """Test part description mapping"""
        self.mapper.load_reference_data(self.surveyor_data, self.garage_data)
        
        result = await self.mapper.map_part_description('Front Bumper Complete')
        self.assertIsNotNone(result)
        self.assertIn('part_code', result)
        self.assertIn('confidence', result)
        
    def test_prompt_creation(self):
        """Test Gemini prompt creation"""
        input_desc = "Front Bumper"
        matches = ['Bumper Front Assembly', 'Bumper Rear Assembly']
        prompt = self.mapper._create_mapping_prompt(input_desc, matches)
        self.assertIsInstance(prompt, str)
        self.assertIn(input_desc, prompt)
        
    def test_response_parsing(self):
        """Test Gemini response parsing"""
        test_response = """
        {
            "match_found": true,
            "part_code": "1001",
            "surveyor_name": "Bumper Front Assembly",
            "confidence": 95,
            "reasoning": "Direct match with standard terminology"
        }
        """
        result = self.mapper._parse_gemini_response(test_response)
        self.assertIsNotNone(result)
        self.assertEqual(result['part_code'], "1001")

def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

if __name__ == '__main__':
    unittest.main() 