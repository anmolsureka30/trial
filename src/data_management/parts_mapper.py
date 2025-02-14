import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
import json

class PartsMapper:
    def __init__(self, gemini_api_key: str):
        """Initialize the parts mapper with necessary models"""
        self.setup_logging()
        self.setup_gemini(gemini_api_key)
        self.setup_embeddings()
        self.parts_cache = {}
        self.embeddings_cache = {}
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/parts_mapper.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gemini(self, api_key: str):
        """Initialize Gemini API"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def setup_embeddings(self):
        """Initialize sentence transformer model"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_reference_data(self, surveyor_data: pd.DataFrame, garage_data: pd.DataFrame):
        """Load and process reference data"""
        try:
            self.surveyor_parts = surveyor_data
            self.garage_parts = garage_data
            
            # Generate embeddings for all parts
            self.surveyor_embeddings = self._generate_embeddings(
                self.surveyor_parts['Surveyor Part Name'].tolist()
            )
            self.garage_embeddings = self._generate_embeddings(
                self.garage_parts['Garage Part Description'].tolist()
            )
            
            self.logger.info("Reference data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading reference data: {str(e)}")
            raise
            
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text descriptions"""
        embeddings = []
        for text in texts:
            if text in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[text])
            else:
                embedding = self.embedding_model.encode(text)
                self.embeddings_cache[text] = embedding
                embeddings.append(embedding)
        return np.array(embeddings)
        
    async def map_part_description(self, part_description: str) -> Dict:
        """Map garage part description to surveyor part code"""
        try:
            # Check cache first
            if part_description in self.parts_cache:
                return self.parts_cache[part_description]
                
            # Generate embedding for input description
            input_embedding = self.embedding_model.encode(part_description)
            
            # Calculate similarities with surveyor parts
            similarities_surveyor = cosine_similarity(
                [input_embedding], 
                self.surveyor_embeddings
            )[0]
            
            # Get top matches
            top_indices = similarities_surveyor.argsort()[-3:][::-1]
            top_matches = self.surveyor_parts.iloc[top_indices]
            
            # Use Gemini for verification and disambiguation
            prompt = self._create_mapping_prompt(
                part_description,
                top_matches['Surveyor Part Name'].tolist()
            )
            
            response = await self.model.generate_content_async(prompt)
            verified_match = self._parse_gemini_response(response.text)
            
            if verified_match:
                result = {
                    'part_code': verified_match['part_code'],
                    'surveyor_name': verified_match['surveyor_name'],
                    'confidence': verified_match['confidence'],
                    'similarity_score': float(similarities_surveyor[top_indices[0]])
                }
                
                # Cache the result
                self.parts_cache[part_description] = result
                return result
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error mapping part description: {str(e)}")
            return None
            
    def _create_mapping_prompt(self, input_desc: str, potential_matches: List[str]) -> str:
        """Create prompt for Gemini API"""
        prompt = f"""
        Task: Match an auto part description to standardized part names.
        
        Input Description: "{input_desc}"
        
        Potential Standard Names:
        {json.dumps(potential_matches, indent=2)}
        
        Please analyze if the input description matches any of the standard names.
        Consider:
        1. Part functionality
        2. Common alternative names
        3. Regional variations
        4. Technical terms
        
        Return JSON format:
        {{
            "match_found": true/false,
            "part_code": "code if found",
            "surveyor_name": "matched standard name",
            "confidence": 0-100,
            "reasoning": "brief explanation"
        }}
        """
        return prompt
        
    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse and validate Gemini API response"""
        try:
            response_data = json.loads(response_text)
            if response_data.get('match_found'):
                return {
                    'part_code': response_data['part_code'],
                    'surveyor_name': response_data['surveyor_name'],
                    'confidence': response_data['confidence']
                }
            return None
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {str(e)}")
            return None
            
    def analyze_mapping_accuracy(self) -> Dict:
        """Analyze mapping accuracy using test data"""
        try:
            results = {
                'total_mappings': 0,
                'successful_mappings': 0,
                'high_confidence_mappings': 0,
                'low_confidence_mappings': 0,
                'failed_mappings': 0,
                'average_confidence': 0.0,
                'common_failures': []
            }
            
            # Implement accuracy analysis logic
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing mapping accuracy: {str(e)}")
            return {} 