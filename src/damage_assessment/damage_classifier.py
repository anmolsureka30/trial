import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from src.ai_services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class DamageAssessment:
    def __init__(self, model_path: str = "models/damage_model.joblib"):
        """Initialize the damage assessment system"""
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.setup_logging()
        self.gemini_service = GeminiService()
        self.load_damage_categories()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/damage_assessment.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def train_model(self, training_data: pd.DataFrame):
        """Train the damage assessment model"""
        try:
            # Prepare features
            X = self._prepare_features(training_data)
            y = training_data['damage_severity']
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X, y_encoded)
            
            # Save the model
            joblib.dump({
                'model': self.model,
                'label_encoder': self.label_encoder
            }, self.model_path)
            
            self.logger.info("Model trained and saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model training/prediction"""
        # Extract relevant features
        features = [
            'part_age',
            'impact_severity',
            'material_type',
            'location_score',
            'previous_repairs'
        ]
        
        # Convert categorical variables to numeric
        numeric_data = pd.get_dummies(data[features], columns=['material_type'])
        
        return numeric_data.values
        
    def load_model(self):
        """Load the trained model"""
        try:
            saved_model = joblib.load(self.model_path)
            self.model = saved_model['model']
            self.label_encoder = saved_model['label_encoder']
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    def assess_damage(self, damage_data: Dict) -> Dict:
        """Assess damage and provide classification with confidence"""
        try:
            if self.model is None:
                self.load_model()
                
            # Prepare input data
            input_features = self._prepare_input(damage_data)
            
            # Get prediction and probability
            prediction = self.model.predict(input_features)
            probabilities = self.model.predict_proba(input_features)
            
            # Get the predicted class and confidence
            predicted_class = self.label_encoder.inverse_transform(prediction)[0]
            confidence = float(np.max(probabilities[0]) * 100)
            
            # Calculate estimated cost
            estimated_cost = self._estimate_repair_cost(
                predicted_class,
                damage_data['part_code'],
                confidence
            )
            
            return {
                'damage_severity': predicted_class,
                'confidence': confidence,
                'estimated_cost': estimated_cost,
                'assessment_timestamp': datetime.now().isoformat(),
                'recommendations': self._generate_recommendations(predicted_class, confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error in damage assessment: {str(e)}")
            return None
            
    def _prepare_input(self, damage_data: Dict) -> np.ndarray:
        """Prepare input data for prediction"""
        # Convert input data to feature array
        input_df = pd.DataFrame([{
            'part_age': damage_data.get('part_age', 0),
            'impact_severity': damage_data.get('impact_severity', 0),
            'material_type': damage_data.get('material_type', 'unknown'),
            'location_score': damage_data.get('location_score', 0),
            'previous_repairs': damage_data.get('previous_repairs', 0)
        }])
        
        return self._prepare_features(input_df)
        
    def _estimate_repair_cost(self, severity: str, part_code: str, confidence: float) -> Dict:
        """Estimate repair cost based on damage severity and part information"""
        # Basic cost estimation logic
        base_costs = {
            'minor': 100,
            'moderate': 300,
            'severe': 500,
            'critical': 1000
        }
        
        base_cost = base_costs.get(severity.lower(), 200)
        
        # Adjust cost based on confidence
        confidence_factor = confidence / 100
        adjusted_cost = base_cost * confidence_factor
        
        return {
            'base_cost': base_cost,
            'adjusted_cost': round(adjusted_cost, 2),
            'confidence_factor': confidence_factor,
            'currency': 'USD'
        }
        
    def _generate_recommendations(self, severity: str, confidence: float) -> List[str]:
        """Generate recommendations based on damage assessment"""
        recommendations = []
        
        if confidence < 70:
            recommendations.append("Manual inspection recommended due to low confidence")
            
        severity_recommendations = {
            'minor': [
                "Simple repair possible",
                "Consider paintless dent repair",
                "Check for surface scratches"
            ],
            'moderate': [
                "Partial repair recommended",
                "Check surrounding parts for damage",
                "Consider paint matching requirements"
            ],
            'severe': [
                "Full part replacement recommended",
                "Inspect structural integrity",
                "Check adjacent parts for damage"
            ],
            'critical': [
                "Immediate replacement required",
                "Safety inspection mandatory",
                "Check for frame damage"
            ]
        }
        
        recommendations.extend(severity_recommendations.get(severity.lower(), []))
        return recommendations

    def load_damage_categories(self) -> List[str]:
        """Load damage categories"""
        self.damage_categories = [
            'Minor_Scratch',
            'Deep_Scratch',
            'Dent',
            'Crack',
            'Break',
            'Misalignment',
            'Paint_Damage',
            'Structural_Damage'
        ]
        return self.damage_categories

    async def assess_damage_from_image(self, image_path: str, claim_data: Dict) -> Dict:
        """Assess damage from image and claim data"""
        try:
            # Process image
            image_features = self._process_image(image_path)
            
            # Get model predictions
            damage_predictions = self._get_predictions(image_features)
            
            # Get AI analysis
            ai_analysis = await self._get_ai_analysis(image_path, damage_predictions, claim_data)
            
            # Combine results
            assessment = self._combine_results(damage_predictions, ai_analysis)
            
            return assessment
        except Exception as e:
            logger.error(f"Damage assessment failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _process_image(self, image_path: str) -> torch.Tensor:
        """Process image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image)
            return image_tensor.unsqueeze(0)
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise

    def _get_predictions(self, image_features: torch.Tensor) -> List[Tuple[str, float]]:
        """Get model predictions"""
        with torch.no_grad():
            outputs = self.model(image_features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            predictions = [
                (self.damage_categories[idx], prob.item())
                for prob, idx in zip(top_probs[0], top_indices[0])
            ]
            
            return predictions

    async def _get_ai_analysis(self, image_path: str, predictions: List[Tuple[str, float]], 
                             claim_data: Dict) -> Dict:
        """Get AI analysis of damage"""
        try:
            # Prepare context for Gemini
            damage_context = {
                "image_path": image_path,
                "detected_damage": predictions,
                **claim_data
            }
            
            # Get AI analysis
            analysis = await self.gemini_service.analyze_claim(damage_context)
            return analysis
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {"error": str(e)}

    def _combine_results(self, predictions: List[Tuple[str, float]], 
                        ai_analysis: Dict) -> Dict:
        """Combine model predictions and AI analysis"""
        return {
            "damage_detection": {
                "predictions": predictions,
                "confidence": np.mean([prob for _, prob in predictions])
            },
            "ai_analysis": ai_analysis,
            "timestamp": pd.Timestamp.now(),
            "status": "success"
        }

    def setup_transforms(self):
        """Setup image transformations"""
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def train_model(self, train_data_dir: str, epochs: int = 10):
        """Train the damage classification model"""
        try:
            # Training implementation
            logger.info("Model training started")
            # Add training logic here
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise 