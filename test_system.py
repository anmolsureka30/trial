import logging
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create sample data for testing"""
    logger.info("Creating test data...")
    
    # Create sample claims data
    dates = pd.date_range(start='2023-01-01', end='2024-02-14', freq='D')
    n_claims = len(dates)
    
    claims_data = pd.DataFrame({
        'claim_id': [f'CLM{i:05d}' for i in range(n_claims)],
        'timestamp': dates,
        'total_amount': np.random.normal(5000, 1000, n_claims),
        'fraud_risk': np.random.uniform(0, 100, n_claims),
        'processing_time': np.random.uniform(1, 10, n_claims),
        'status': np.random.choice(['approved', 'rejected', 'pending'], n_claims),
        'vehicle_info': ['Test Vehicle'] * n_claims,
        'damage_description': ['Test Damage'] * n_claims
    })
    
    claims_data.to_csv('data/historical_claims.csv', index=False)
    logger.info(f"Created {n_claims} sample claims")
    
    # Create necessary directories
    directories = [
        'data/knowledge_base',
        'data/uploads',
        'data/vectors',
        'logs',
        'models',
        'temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return claims_data

def test_components():
    """Test all system components"""
    try:
        logger.info("Testing system components...")
        
        # Test imports
        from src.claims_processing.claims_processor import ClaimsProcessor
        from src.dashboard.enhanced_dashboard import EnhancedDashboard
        from src.analytics.advanced_predictions import AdvancedPredictions
        from src.reports.report_generator import ReportGenerator
        
        # Create test data
        claims_data = create_test_data()
        
        # Test predictions
        logger.info("Testing predictions...")
        predictions = AdvancedPredictions(claims_data)
        predictions.generate_predictions()
        
        # Test report generation
        logger.info("Testing report generation...")
        report_gen = ReportGenerator(claims_data)
        report_gen.generate_pdf_report("Test Report", ["Claims Count", "Total Amount"])
        
        logger.info("All components tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Component testing failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting system test...")
    
    if not test_components():
        logger.error("System test failed!")
        sys.exit(1)
    
    logger.info("System test completed successfully!")
    
    # Start the dashboard
    logger.info("Starting dashboard...")
    from src.dashboard.enhanced_dashboard import EnhancedDashboard
    dashboard = EnhancedDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 