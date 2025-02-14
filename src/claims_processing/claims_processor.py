import logging
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
import asyncio
import numpy as np

from src.damage_assessment.damage_classifier import DamageAssessment
from src.data_management.parts_mapping_service import PartsMapperRAG
from src.fraud_detection.fraud_detector import FraudDetector
from src.ai_services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class ClaimsProcessor:
    def __init__(self):
        """Initialize the claims processing system"""
        try:
            self.setup_components()
            self.load_reference_data()
        except Exception as e:
            logger.error(f"Failed to initialize ClaimsProcessor: {e}")
            raise

    def setup_components(self):
        """Setup all processing components"""
        try:
            self.damage_assessor = DamageAssessment()
            self.parts_mapper = PartsMapperRAG()
            self.fraud_detector = FraudDetector()
            self.ai_service = GeminiService()
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Component setup failed: {e}")
            raise

    def load_reference_data(self):
        """Load reference data and historical claims"""
        try:
            # Load historical claims if available
            claims_path = Path("data/historical_claims.csv")
            if claims_path.exists():
                self.historical_claims = pd.read_csv(claims_path)
            else:
                self.historical_claims = pd.DataFrame()
            
            logger.info("Reference data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            raise

    async def process_claim(self, claim_data: Dict, images: List[str] = None) -> Dict:
        """Process a complete insurance claim"""
        try:
            claim_id = claim_data.get('claim_id', f"CLM{datetime.now().strftime('%Y%m%d%H%M%S')}")
            logger.info(f"Processing claim: {claim_id}")

            # Initialize result dictionary
            result = {
                'claim_id': claim_id,
                'status': 'processing',
                'timestamp': datetime.now(),
                'assessments': {}
            }

            # Parallel processing tasks
            tasks = []
            
            # 1. Damage Assessment
            if images:
                tasks.append(self._assess_damage(images[0], claim_data))
            
            # 2. Parts Mapping
            if 'parts' in claim_data:
                tasks.append(self._map_parts(claim_data['parts']))
            
            # 3. Fraud Detection
            tasks.append(self._check_fraud(claim_data))
            
            # 4. AI Analysis
            tasks.append(self._get_ai_analysis(claim_data))

            # Execute all tasks concurrently
            assessments = await asyncio.gather(*tasks)
            
            # Combine results
            result['assessments'] = {
                'damage': assessments[0] if images else None,
                'parts_mapping': assessments[1] if 'parts' in claim_data else None,
                'fraud_check': assessments[2] if len(assessments) > 2 else None,
                'ai_analysis': assessments[3] if len(assessments) > 3 else None
            }

            # Determine final recommendation
            result.update(self._generate_recommendation(result['assessments']))
            
            # Update historical data
            self._update_historical_data(result)
            
            return result

        except Exception as e:
            logger.error(f"Claim processing failed: {e}")
            return {
                'claim_id': claim_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now()
            }

    async def _assess_damage(self, image_path: str, claim_data: Dict) -> Dict:
        """Perform damage assessment"""
        try:
            assessment = await self.damage_assessor.assess_damage_from_image(
                image_path=image_path,
                claim_data=claim_data
            )
            return assessment
        except Exception as e:
            logger.error(f"Damage assessment failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def _map_parts(self, parts_data: List[Dict]) -> Dict:
        """Map parts descriptions"""
        try:
            mapped_parts = []
            for part in parts_data:
                mapping = await self.parts_mapper.map_part(
                    part_description=part['description'],
                    additional_info=part.get('additional_info')
                )
                mapped_parts.append(mapping)
            
            return {
                'status': 'success',
                'mapped_parts': mapped_parts,
                'confidence': np.mean([p['confidence'] for p in mapped_parts])
            }
        except Exception as e:
            logger.error(f"Parts mapping failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def _check_fraud(self, claim_data: Dict) -> Dict:
        """Perform fraud detection"""
        try:
            fraud_check = await self.fraud_detector.analyze_claim(claim_data)
            return fraud_check
        except Exception as e:
            logger.error(f"Fraud detection failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def _get_ai_analysis(self, claim_data: Dict) -> Dict:
        """Get AI analysis of the claim"""
        try:
            analysis = await self.ai_service.analyze_claim(claim_data)
            return analysis
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _generate_recommendation(self, assessments: Dict) -> Dict:
        """Generate final recommendation based on all assessments"""
        try:
            # Extract key metrics
            fraud_risk = assessments['fraud_check'].get('risk_score', 0)
            damage_confidence = assessments['damage'].get('confidence', 0) if assessments['damage'] else 100
            parts_confidence = assessments['parts_mapping'].get('confidence', 0) if assessments['parts_mapping'] else 100
            
            # Decision logic
            if fraud_risk > 80:
                recommendation = "reject"
                reason = "High fraud risk detected"
            elif fraud_risk > 60:
                recommendation = "review"
                reason = "Elevated fraud risk"
            elif damage_confidence < 50 or parts_confidence < 50:
                recommendation = "review"
                reason = "Low confidence in assessment"
            else:
                recommendation = "approve"
                reason = "All checks passed"
            
            return {
                'recommendation': recommendation,
                'reason': reason,
                'confidence_scores': {
                    'fraud_risk': fraud_risk,
                    'damage_confidence': damage_confidence,
                    'parts_confidence': parts_confidence
                }
            }
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {
                'recommendation': 'review',
                'reason': f'Error in processing: {str(e)}'
            }

    def _update_historical_data(self, claim_result: Dict):
        """Update historical claims database"""
        try:
            new_claim = pd.DataFrame([{
                'claim_id': claim_result['claim_id'],
                'timestamp': claim_result['timestamp'],
                'recommendation': claim_result['recommendation'],
                'fraud_risk': claim_result['assessments']['fraud_check'].get('risk_score', 0),
                'total_amount': claim_result.get('total_amount', 0)
            }])
            
            if self.historical_claims.empty:
                self.historical_claims = new_claim
            else:
                self.historical_claims = pd.concat([self.historical_claims, new_claim])
            
            # Save to file
            self.historical_claims.to_csv("data/historical_claims.csv", index=False)
            
        except Exception as e:
            logger.error(f"Failed to update historical data: {e}")