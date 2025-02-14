from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class PartsMapper:
    def __init__(self):
        """Initialize the parts mapping service"""
        self.setup_components()
        self.load_reference_data()
        self.create_knowledge_base()

    def setup_components(self):
        """Setup necessary components"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )

    def load_reference_data(self):
        """Load reference data"""
        self.standard_parts = pd.read_csv('data/Primary_Parts_Code.csv')
        self.garage_data = pd.read_csv('data/garage.csv')

    def create_knowledge_base(self):
        """Create knowledge base from standard parts"""
        documents = []
        for _, row in self.standard_parts.iterrows():
            doc = f"""
            Part Code: {row['Surveyor Part Code']}
            Standard Name: {row['Surveyor Part Name']}
            Common Variations:
            - {row['Surveyor Part Name'].replace(' ', '-')}
            - {row['Surveyor Part Name'].lower()}
            - {row['Surveyor Part Name'].upper()}
            """
            documents.append(doc)

        texts = self.text_splitter.create_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="data/vectors/parts"
        )

    async def map_part(self, description: str) -> Dict:
        """Map a garage part description to standard part"""
        try:
            # Get similar parts from vector store
            similar_docs = self.vectorstore.similarity_search(description, k=3)
            context = "\n".join(doc.page_content for doc in similar_docs)

            # Create prompt
            prompt = PromptTemplate.from_template("""
            Given the following part description from a garage:
            {description}

            And these potential standard parts:
            {context}

            Please identify the most likely matching standard part.
            Return your response in JSON format with:
            - mapped_code: the matching part code
            - confidence: matching confidence (0-100)
            - reasoning: brief explanation of the match
            """)

            # Generate mapping
            response = await self.llm.agenerate([prompt.format(
                description=description,
                context=context
            )])

            return response.generations[0].text

        except Exception as e:
            logging.error(f"Mapping failed: {e}")
            return None

    def get_mapping_statistics(self) -> Dict:
        """Get mapping performance statistics"""
        stats = {
            'total_mappings': len(self.garage_data),
            'successful_mappings': len(self.garage_data[self.garage_data['mapped_part_code'].notna()]),
            'average_confidence': self.garage_data['mapping_confidence'].mean(),
            'error_distribution': self.garage_data['error_type'].value_counts().to_dict()
        }
        return stats 