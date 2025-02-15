import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sqlite3
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class SmartRecommender:
    def __init__(self, db_path: str = ":memory:"):
        """Initialize the smart recommendation system"""
        self.db_path = db_path
        self.setup_logging()
        self.initialize_database()
        self.pattern_cache = {}
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_database(self):
        """Initialize SQLite database for storing patterns and recommendations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create patterns table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS damage_patterns (
                        pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        primary_part_code TEXT,
                        associated_part_code TEXT,
                        confidence FLOAT,
                        occurrence_count INTEGER,
                        last_updated TIMESTAMP
                    )
                """)
                
                # Create recommendations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS part_recommendations (
                        recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        part_code TEXT,
                        recommendation_type TEXT,
                        description TEXT,
                        priority INTEGER,
                        created_at TIMESTAMP
                    )
                """)
                
                # Create indices
                conn.execute("CREATE INDEX IF NOT EXISTS idx_primary_part ON damage_patterns(primary_part_code)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_part_code ON part_recommendations(part_code)")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
            
    def update_damage_patterns(self, historical_data: pd.DataFrame):
        """Update damage patterns based on historical claims data"""
        try:
            patterns = self._analyze_patterns(historical_data)
            
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing patterns
                conn.execute("DELETE FROM damage_patterns")
                
                # Insert new patterns
                for pattern in patterns:
                    conn.execute("""
                        INSERT INTO damage_patterns 
                        (primary_part_code, associated_part_code, confidence, occurrence_count, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        pattern['primary_part'],
                        pattern['associated_part'],
                        pattern['confidence'],
                        pattern['occurrences'],
                        datetime.now().isoformat()
                    ))
                    
            self.logger.info(f"Updated {len(patterns)} damage patterns")
            
        except Exception as e:
            self.logger.error(f"Error updating damage patterns: {str(e)}")
            raise
            
    def _analyze_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Analyze historical data to identify damage patterns"""
        patterns = []
        
        # Group claims by incident
        grouped_claims = data.groupby('claim_id')['damaged_part_code'].agg(list)
        
        # Count co-occurrences
        co_occurrences = defaultdict(int)
        total_claims = len(grouped_claims)
        
        for parts in grouped_claims:
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    pair = tuple(sorted([parts[i], parts[j]]))
                    co_occurrences[pair] += 1
        
        # Calculate confidence scores
        for (part1, part2), count in co_occurrences.items():
            confidence = (count / total_claims) * 100
            if confidence >= 5.0:  # Min threshold for pattern recognition
                patterns.append({
                    'primary_part': part1,
                    'associated_part': part2,
                    'confidence': confidence,
                    'occurrences': count
                })
        
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)
        
    def get_recommendations(self, part_code: str, damage_severity: str) -> Dict:
        """Get smart recommendations for a damaged part"""
        try:
            recommendations = {
                'associated_parts': self._get_associated_parts(part_code),
                'repair_suggestions': self._get_repair_suggestions(part_code, damage_severity),
                'cost_estimates': self._get_cost_estimates(part_code, damage_severity),
                'priority_level': self._calculate_priority(part_code, damage_severity),
                'timestamp': datetime.now().isoformat()
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return None
            
    def _get_associated_parts(self, part_code: str) -> List[Dict]:
        """Get frequently associated damaged parts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT associated_part_code, confidence, occurrence_count
                    FROM damage_patterns
                    WHERE primary_part_code = ?
                    ORDER BY confidence DESC
                    LIMIT 5
                """
                
                results = pd.read_sql_query(query, conn, params=(part_code,))
                
                associated_parts = []
                for _, row in results.iterrows():
                    associated_parts.append({
                        'part_code': row['associated_part_code'],
                        'confidence': row['confidence'],
                        'occurrences': row['occurrence_count']
                    })
                    
                return associated_parts
                
        except Exception as e:
            self.logger.error(f"Error retrieving associated parts: {str(e)}")
            return []
            
    def _get_repair_suggestions(self, part_code: str, damage_severity: str) -> List[str]:
        """Get repair suggestions based on part and damage severity"""
        severity_suggestions = {
            'minor': [
                "Visual inspection recommended",
                "Consider paintless dent repair if applicable",
                "Check for paint damage"
            ],
            'moderate': [
                "Detailed damage assessment required",
                "Consider partial replacement",
                "Check structural integrity"
            ],
            'severe': [
                "Full replacement recommended",
                "Check surrounding components",
                "Assess frame alignment"
            ],
            'critical': [
                "Immediate replacement required",
                "Full structural inspection needed",
                "Safety systems check mandatory"
            ]
        }
        
        return severity_suggestions.get(damage_severity.lower(), [])
        
    def _get_cost_estimates(self, part_code: str, damage_severity: str) -> Dict:
        """Generate cost estimates for repair/replacement"""
        # Basic cost estimation logic
        severity_multipliers = {
            'minor': 0.3,
            'moderate': 0.6,
            'severe': 0.9,
            'critical': 1.0
        }
        
        base_cost = 1000  # Example base cost
        multiplier = severity_multipliers.get(damage_severity.lower(), 0.5)
        
        return {
            'estimated_cost': round(base_cost * multiplier, 2),
            'confidence_level': 'high' if damage_severity in ['severe', 'critical'] else 'medium',
            'includes_labor': True,
            'currency': 'USD'
        }
        
    def _calculate_priority(self, part_code: str, damage_severity: str) -> Dict:
        """Calculate priority level for repair/replacement"""
        severity_scores = {
            'minor': 1,
            'moderate': 2,
            'severe': 3,
            'critical': 4
        }
        
        score = severity_scores.get(damage_severity.lower(), 1)
        
        priority_levels = {
            1: 'low',
            2: 'medium',
            3: 'high',
            4: 'urgent'
        }
        
        return {
            'level': priority_levels[score],
            'score': score,
            'requires_immediate_action': score >= 3
        } 