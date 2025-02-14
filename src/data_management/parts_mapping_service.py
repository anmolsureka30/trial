from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class PartsMapperRAG:
    def __init__(self):
        """Initialize the parts mapping system with RAG"""
        try:
            self.setup_components()
            self.load_reference_data()
            self.create_knowledge_base()
        except Exception as e:
            logger.error(f"Failed to initialize PartsMapperRAG: {e}")
            raise

    def setup_components(self):
        """Setup RAG components"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            
            # Initialize Gemini
            self.llm = GoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.3
            )
            
            # Setup text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            
            # Initialize mapping prompts
            self.setup_prompts()
            
            logger.info("RAG components initialized successfully")
        except Exception as e:
            logger.error(f"Component setup failed: {e}")
            raise

    def setup_prompts(self):
        """Setup mapping prompts"""
        self.mapping_prompt = PromptTemplate(
            input_variables=["part_info", "context", "reference_data"],
            template="""
            Task: Map automotive part descriptions between garage and surveyor datasets.
            
            Part Information:
            {part_info}
            
            Relevant Context:
            {context}
            
            Reference Data:
            {reference_data}
            
            Please provide:
            1. Best matching standard part code and name
            2. Confidence score (0-100)
            3. Reasoning for the match
            4. Alternative matches if applicable
            
            Format the response as a structured JSON.
            """
        )
        
        self.mapping_chain = LLMChain(
            llm=self.llm,
            prompt=self.mapping_prompt
        )

    def load_reference_data(self):
        """Load and process reference data"""
        try:
            # Load standard parts catalog
            self.parts_catalog = pd.read_csv("data/Primary_Parts_Code.csv")
            
            # Load historical mappings if available
            mappings_path = Path("data/historical_mappings.csv")
            if mappings_path.exists():
                self.historical_mappings = pd.read_csv(mappings_path)
            else:
                self.historical_mappings = pd.DataFrame()
                
            logger.info("Reference data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            raise

    def create_knowledge_base(self):
        """Create knowledge base from parts catalog"""
        try:
            # Load parts catalog
            self.parts_catalog = pd.read_csv("data/Primary_Parts_Code.csv")
            
            # Create knowledge base documents
            documents = []
            for _, row in self.parts_catalog.iterrows():
                doc = f"""
                Part Information:
                Part Code: {row['Surveyor Part Code']}  # Changed from 'Part_Code'
                Part Name: {row['Surveyor Part Name']}  # Changed from 'Part_Name'
                Category: {row['Category']}
                Average Cost: ${row['Average_Cost']:.2f}
                
                Description:
                Standard automotive part used in vehicle repairs.
                Category: {row['Category']} component
                Typical applications include repair and replacement work.
                """
                documents.append(doc)
            
            # Create text chunks
            text_chunks = self.text_splitter.create_documents(documents)
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=text_chunks,
                embedding=self.embeddings,
                persist_directory="data/vectors"
            )
            
            logger.info(f"Created knowledge base with {len(documents)} parts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {e}")
            return False

    async def map_part(self, part_description: str, additional_info: Dict = None) -> Dict:
        """Map a part description to standard catalog"""
        try:
            # Get relevant context from vector store
            context = self._get_relevant_context(part_description)
            
            # Prepare mapping input
            part_info = {
                "description": part_description,
                "additional_info": additional_info or {}
            }
            
            # Get reference data
            reference_data = self._get_reference_data(part_description)
            
            # Generate mapping
            mapping_result = await self.mapping_chain.arun(
                part_info=str(part_info),
                context=context,
                reference_data=str(reference_data)
            )
            
            # Process and validate result
            processed_result = self._process_mapping_result(mapping_result, part_description)
            
            # Update historical mappings
            self._update_historical_mappings(processed_result)
            
            return processed_result
        except Exception as e:
            logger.error(f"Part mapping failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context from vector store"""
        docs = self.vector_store.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_reference_data(self, query: str) -> Dict:
        """Get relevant reference data"""
        # Get similar parts from catalog
        similar_parts = self.parts_catalog[
            self.parts_catalog['Description'].str.contains(query, case=False, na=False)
        ]
        
        # Get historical mappings if available
        historical = self.historical_mappings[
            self.historical_mappings['Original_Description'].str.contains(query, case=False, na=False)
        ] if not self.historical_mappings.empty else pd.DataFrame()
        
        return {
            "similar_parts": similar_parts.to_dict('records'),
            "historical_mappings": historical.to_dict('records')
        }

    def _process_mapping_result(self, result: str, original_description: str) -> Dict:
        """Process and validate mapping result"""
        try:
            # Parse JSON result
            mapping = eval(result)  # In production, use proper JSON parsing
            
            # Validate mapping
            if mapping['confidence'] < 50:
                mapping['needs_review'] = True
                logger.warning(f"Low confidence mapping for: {original_description}")
            
            mapping['original_description'] = original_description
            mapping['timestamp'] = pd.Timestamp.now()
            
            return mapping
        except Exception as e:
            logger.error(f"Failed to process mapping result: {e}")
            raise

    def _update_historical_mappings(self, mapping: Dict):
        """Update historical mappings database"""
        try:
            new_mapping = pd.DataFrame([{
                'Original_Description': mapping['original_description'],
                'Mapped_Part': mapping['standard_part_code'],
                'Confidence': mapping['confidence'],
                'Timestamp': mapping['timestamp']
            }])
            
            if self.historical_mappings.empty:
                self.historical_mappings = new_mapping
            else:
                self.historical_mappings = pd.concat([self.historical_mappings, new_mapping])
            
            # Save to file
            self.historical_mappings.to_csv("data/historical_mappings.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to update historical mappings: {e}") 