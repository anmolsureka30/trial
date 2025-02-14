import google.generativeai as genai
import os
from typing import List, Dict
import logging
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        """Initialize Gemini AI service with RAG capabilities"""
        try:
            self.setup_gemini()
            self.setup_rag()
            self.load_knowledge_base()
        except Exception as e:
            logger.error(f"Failed to initialize GeminiService: {e}")
            raise

    def setup_gemini(self):
        """Setup Gemini API"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini API setup successful")
        except Exception as e:
            logger.error(f"Gemini API setup failed: {e}")
            raise

    def setup_rag(self):
        """Setup RAG components"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            self.vector_store = None  # Will be initialized when loading knowledge base
            logger.info("RAG components setup successful")
        except Exception as e:
            logger.error(f"RAG setup failed: {e}")
            raise

    def load_knowledge_base(self):
        """Load and process knowledge base documents"""
        try:
            kb_dir = Path("data/knowledge_base")
            if not kb_dir.exists():
                kb_dir.mkdir(parents=True)
                self._create_sample_knowledge_base(kb_dir)

            documents = []
            for file in kb_dir.glob("*.txt"):
                with open(file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty documents
                        documents.append(content)

            # Only create vector store if we have documents
            if documents:
                # Split documents into chunks
                texts = self.text_splitter.create_documents(documents)
                
                if texts:  # Verify we have texts after splitting
                    # Create vector store
                    self.vector_store = Chroma.from_documents(
                        documents=texts,
                        embedding=self.embeddings,
                        persist_directory="data/vectors"
                    )
                    logger.info(f"Loaded {len(texts)} document chunks into vector store")
                else:
                    logger.warning("No valid text chunks created from documents")
                    self.vector_store = None
            else:
                logger.warning("No documents found in knowledge base")
                self.vector_store = None
            
        except Exception as e:
            logger.error(f"Knowledge base loading failed: {e}")
            self.vector_store = None

    def _create_sample_knowledge_base(self, kb_dir: Path):
        """Create sample knowledge base documents"""
        sample_docs = {
            "parts_guide.txt": """
            Comprehensive guide to automotive parts and their classifications.
            Common parts include: bumpers, fenders, doors, hood, trunk, etc.
            Each part has specific damage assessment criteria and repair guidelines.
            """,
            "repair_procedures.txt": """
            Standard repair procedures for various types of damage.
            Includes safety guidelines, quality standards, and best practices.
            Repair vs. replace decision criteria and cost considerations.
            """,
            "fraud_patterns.txt": """
            Common fraud patterns in insurance claims:
            1. Multiple claims for same damage
            2. Inflated repair costs
            3. Pre-existing damage claims
            4. Staged accidents
            """
        }
        
        for filename, content in sample_docs.items():
            with open(kb_dir / filename, 'w') as f:
                f.write(content)

    async def analyze_claim(self, claim_data: Dict) -> Dict:
        """Analyze claim using Gemini and RAG"""
        try:
            # Prepare claim context
            claim_context = self._prepare_claim_context(claim_data)
            
            # Get relevant knowledge base information
            relevant_info = self._retrieve_relevant_info(claim_context)
            
            # Generate analysis using Gemini
            prompt = self._create_analysis_prompt(claim_data, relevant_info)
            response = await self.model.generate_content_async(prompt)
            
            # Process and structure the response
            analysis = self._process_response(response)
            
            return analysis
        except Exception as e:
            logger.error(f"Claim analysis failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _prepare_claim_context(self, claim_data: Dict) -> str:
        """Prepare claim context for analysis"""
        return f"""
        Claim ID: {claim_data.get('claim_id')}
        Vehicle: {claim_data.get('vehicle_info')}
        Damage Description: {claim_data.get('damage_description')}
        Estimated Cost: {claim_data.get('estimated_cost')}
        Parts Affected: {', '.join(claim_data.get('affected_parts', []))}
        """

    def _retrieve_relevant_info(self, query: str) -> str:
        """Retrieve relevant information from knowledge base"""
        if not self.vector_store:
            return ""
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        docs = retriever.get_relevant_documents(query)
        return "\n".join(doc.page_content for doc in docs)

    def _create_analysis_prompt(self, claim_data: Dict, relevant_info: str) -> str:
        """Create analysis prompt for Gemini"""
        return f"""
        Analyze the following insurance claim based on provided information and knowledge base:

        Claim Information:
        {self._prepare_claim_context(claim_data)}

        Relevant Knowledge Base Information:
        {relevant_info}

        Please provide:
        1. Risk Assessment
        2. Repair Recommendations
        3. Cost Analysis
        4. Potential Fraud Indicators (if any)
        5. Next Steps

        Format the response in a structured way.
        """

    def _process_response(self, response) -> Dict:
        """Process and structure Gemini's response"""
        # Add response processing logic here
        return {
            "analysis": response.text,
            "status": "success",
            "timestamp": pd.Timestamp.now()
        }