import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import logging

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path('data')
        self.parts_data = None
        self.load_data()
    
    def load_data(self):
        """Load all necessary data files"""
        try:
            # Load parts catalog
            parts_file = Path('Primary_Parts_Code.csv')
            if parts_file.exists():
                self.parts_data = pd.read_csv(parts_file)
                self.logger.info(f"Loaded {len(self.parts_data)} parts from catalog")
            else:
                self.logger.warning("Parts catalog not found, using sample data")
                from src.data.create_sample_data import create_sample_data
                self.parts_data = create_sample_data()
            
            # Clean and preprocess data
            self.parts_data['Surveyor Part Name'] = self.parts_data['Surveyor Part Name'].str.strip()
            self.parts_data['normalized_name'] = self.parts_data['Surveyor Part Name'].str.lower()
            
            # Create additional metadata
            self.create_parts_metadata()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_parts_metadata(self):
        """Create additional metadata for parts"""
        # Group similar parts
        self.parts_groups = {}
        for name in self.parts_data['normalized_name'].unique():
            base_name = name.split()[0]  # Get first word as base
            if base_name not in self.parts_groups:
                self.parts_groups[base_name] = []
            self.parts_groups[base_name].append(name)
        
        # Create relationships
        self.parts_relationships = {
            'front': ['bumper', 'headlight', 'hood', 'grille'],
            'rear': ['bumper', 'tail light', 'trunk'],
            'side': ['door', 'fender', 'mirror']
        } 