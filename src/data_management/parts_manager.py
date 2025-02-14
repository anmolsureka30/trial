import pandas as pd
from typing import Dict, List, Optional
from rapidfuzz import fuzz, process
import sqlite3
import logging

class PartsManager:
    def __init__(self, db_path: str = "data/parts.db"):
        """Initialize the parts manager with database connection"""
        self.db_path = db_path
        self.parts_df = None
        self.parts_cache = {}
        self.setup_logging()
        self.initialize_database()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/parts_manager.log'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_database(self):
        """Initialize SQLite database and create necessary tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS parts (
                        part_code TEXT PRIMARY KEY,
                        product TEXT,
                        part_name TEXT,
                        normalized_name TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_normalized_name 
                    ON parts(normalized_name)
                """)
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    def load_parts_catalog(self, csv_path: str):
        """Load and process the parts catalog from CSV"""
        try:
            self.parts_df = pd.read_csv(csv_path)
            
            # Normalize column names
            self.parts_df.columns = [col.strip().lower() for col in self.parts_df.columns]
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM parts")
                
                # Insert new data
                for _, row in self.parts_df.iterrows():
                    normalized_name = self._normalize_part_name(row['surveyor part name'])
                    conn.execute(
                        "INSERT INTO parts (part_code, product, part_name, normalized_name) VALUES (?, ?, ?, ?)",
                        (str(row['surveyor part code']), row['product'], row['surveyor part name'], normalized_name)
                    )
                
            self.logger.info(f"Successfully loaded {len(self.parts_df)} parts from catalog")
            
        except Exception as e:
            self.logger.error(f"Failed to load parts catalog: {str(e)}")
            raise

    def _normalize_part_name(self, part_name: str) -> str:
        """Normalize part name for better matching"""
        # Convert to lowercase and remove special characters
        normalized = part_name.lower()
        
        # Replace common separators with spaces
        normalized = normalized.replace('|', ' ').replace('/', ' ').replace('-', ' ')
        
        # Remove multiple spaces
        normalized = ' '.join(normalized.split())
        
        return normalized

    def find_matching_part(self, query: str, threshold: float = 80.0) -> Optional[Dict]:
        """Find matching part using fuzzy matching with caching"""
        # Check cache first
        cache_key = f"{query}_{threshold}"
        if cache_key in self.parts_cache:
            return self.parts_cache[cache_key]

        try:
            normalized_query = self._normalize_part_name(query)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all parts for matching
                parts = pd.read_sql_query("SELECT * FROM parts", conn)
                
                # Perform fuzzy matching
                matches = process.extractBests(
                    normalized_query,
                    parts['normalized_name'].tolist(),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=threshold,
                    limit=1
                )

                if matches:
                    best_match = matches[0]
                    matched_part = parts[parts['normalized_name'] == best_match[0]].iloc[0]
                    
                    result = {
                        'part_code': matched_part['part_code'],
                        'part_name': matched_part['part_name'],
                        'confidence': best_match[1],
                        'normalized_query': normalized_query
                    }
                    
                    # Cache the result
                    self.parts_cache[cache_key] = result
                    return result
                
                return None

        except Exception as e:
            self.logger.error(f"Error in part matching: {str(e)}")
            return None

    def get_part_by_code(self, part_code: str) -> Optional[Dict]:
        """Retrieve part details by part code"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM parts WHERE part_code = ?"
                result = pd.read_sql_query(query, conn, params=(part_code,))
                
                if not result.empty:
                    return result.iloc[0].to_dict()
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving part by code: {str(e)}")
            return None 