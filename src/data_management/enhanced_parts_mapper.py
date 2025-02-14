import os
from typing import Dict, List, Optional
import google.generativeai as genai
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.data_management.data_loader import DataLoader

class EnhancedPartsMapper:
    def __init__(self):
        """Initialize the enhanced parts mapper"""
        self.setup_logging()
        self.setup_models()
        self.setup_vector_store()
        self.mapping_history = []
        self.data_loader = DataLoader()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/enhanced_parts_mapper.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """Initialize AI models and embeddings"""
        try:
            # Load environment variables
            load_dotenv()
            
            # Setup Gemini
            GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
                
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini = genai.GenerativeModel('gemini-pro')
            
            # Setup embeddings using sentence-transformers directly
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}  # Force CPU usage
            )
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up models: {str(e)}")
            raise
            
    def setup_vector_store(self):
        """Initialize vector store"""
        self.vector_store = None
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data structure"""
        required_columns = ['Surveyor Part Code', 'Surveyor Part Name']
        
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False
        
        if data.empty:
            self.logger.error("Empty dataset provided")
            return False
        
        if data.duplicated(subset=['Surveyor Part Code']).any():
            self.logger.warning("Duplicate part codes found")
        
        return True

    def load_reference_data(self, surveyor_data: pd.DataFrame = None):
        """Load and process reference data into vector store"""
        try:
            # Use provided data or load from DataLoader
            if surveyor_data is None:
                surveyor_data = self.data_loader.parts_data
            
            # Validate data
            if not self.validate_data(surveyor_data):
                raise ValueError("Invalid data format")
            
            # Create documents with enhanced metadata
            documents = []
            for _, row in surveyor_data.iterrows():
                base_name = row['normalized_name'].split()[0]
                related_parts = self.data_loader.parts_groups.get(base_name, [])
                
                doc = {
                    'page_content': row['Surveyor Part Name'],
                    'metadata': {
                        'part_code': row['Surveyor Part Code'],
                        'normalized_name': row['normalized_name'],
                        'related_parts': related_parts,
                        'location': next((loc for loc, parts in 
                                       self.data_loader.parts_relationships.items() 
                                       if base_name in parts), 'other')
                    }
                }
                documents.append(doc)
            
            # Create vector store with enhanced documents
            texts = [doc['page_content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            self.reference_data = surveyor_data
            self.logger.info(f"Loaded {len(documents)} parts into vector store")
            
        except Exception as e:
            self.logger.error(f"Error loading reference data: {str(e)}")
            raise
            
    async def map_part(self, part_description: str) -> Dict:
        """Map garage part description to standardized part"""
        try:
            # Step 1: Vector similarity search
            similar_parts = self.vector_store.similarity_search(
                part_description,
                k=3
            )
            
            # Step 2: Generate context for Gemini
            context = self._prepare_context(similar_parts)
            
            # Step 3: Use Gemini for verification
            prompt = self._create_mapping_prompt(part_description, context)
            response = await self.gemini.generate_content_async(prompt)
            
            # Step 4: Parse and validate response
            result = self._parse_gemini_response(response.text)
            
            if result:
                # Step 5: Store mapping history
                self._store_mapping_history(part_description, result)
                return result
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error mapping part: {str(e)}")
            return None
            
    def _prepare_context(self, similar_parts: List) -> List[Dict]:
        """Prepare context for Gemini from similar parts"""
        context = []
        for doc in similar_parts:
            part_data = self.reference_data[
                self.reference_data['Surveyor Part Name'] == doc.page_content
            ].iloc[0]
            
            context.append({
                'part_code': part_data['Surveyor Part Code'],
                'part_name': doc.page_content,
                'similarity_score': doc.metadata.get('score', 0)
            })
        return context
        
    def _create_mapping_prompt(self, input_desc: str, context: List[Dict]) -> str:
        """Create detailed prompt for Gemini"""
        prompt = f"""
        Task: Map an auto part description to a standardized part name.
        
        Input Description: "{input_desc}"
        
        Potential Matches:
        {json.dumps(context, indent=2)}
        
        Analyze the input description and potential matches considering:
        1. Part functionality and purpose
        2. Common alternative names and industry terminology
        3. Regional variations and synonyms
        4. Technical specifications and compatibility
        5. Part relationships and hierarchy
        
        Return a JSON response with:
        {{
            "match_found": true/false,
            "part_code": "matched part code if found",
            "standardized_name": "matched part name",
            "confidence_score": 0-100,
            "reasoning": "detailed explanation of the match",
            "alternative_suggestions": ["list of alternative matches if relevant"]
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
                    'standardized_name': response_data['standardized_name'],
                    'confidence_score': response_data['confidence_score'],
                    'reasoning': response_data['reasoning'],
                    'alternatives': response_data.get('alternative_suggestions', [])
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {str(e)}")
            return None
            
    def _store_mapping_history(self, input_desc: str, result: Dict):
        """Store mapping history for learning and analysis"""
        self.mapping_history.append({
            'timestamp': datetime.now().isoformat(),
            'input_description': input_desc,
            'mapped_result': result
        })
        
    def analyze_mapping_performance(self) -> Dict:
        """Analyze mapping performance metrics"""
        try:
            total_mappings = len(self.mapping_history)
            if total_mappings == 0:
                return {}
                
            confidence_scores = [
                m['mapped_result']['confidence_score'] 
                for m in self.mapping_history
            ]
            
            return {
                'total_mappings': total_mappings,
                'average_confidence': np.mean(confidence_scores),
                'high_confidence_rate': np.mean([s >= 90 for s in confidence_scores]),
                'low_confidence_rate': np.mean([s < 70 for s in confidence_scores]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {} 