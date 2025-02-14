import os
import subprocess
import sys
from pathlib import Path
import logging
from src.config.config import LOG_CONFIG, DB_CONFIG

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data',
        'logs',
        'models',
        'src/data_management',
        'src/fraud_detection',
        'src/recommendations',
        'src/dashboard',
        'src/config',
        'vectors'  # For storing vector indexes
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py in each src subdirectory
        if directory.startswith('src'):
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Package initialization\n')

    # Create root src/__init__.py
    with open('src/__init__.py', 'w') as f:
        f.write('# Root package initialization\n')

def setup_logging():
    """Configure logging"""
    log_dir = LOG_CONFIG['log_dir']
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=LOG_CONFIG['log_level'],
        format=LOG_CONFIG['log_format'],
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'deployment.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'scikit-learn',
        'rapidfuzz',
        'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {package}: {str(e)}")
            return False
    return True

def initialize_databases():
    """Initialize SQLite databases"""
    try:
        # Import necessary classes
        from src.data_management.parts_manager import PartsManager
        from src.fraud_detection.fraud_detector import FraudDetector
        from src.recommendations.smart_recommender import SmartRecommender
        
        # Initialize components
        PartsManager(db_path=DB_CONFIG['parts_db'])
        FraudDetector(db_path=DB_CONFIG['fraud_db'])
        SmartRecommender(db_path=DB_CONFIG['recommendations_db'])
        
        return True
    except Exception as e:
        logging.error(f"Database initialization failed: {str(e)}")
        return False

def main():
    """Main deployment function"""
    print("Starting local deployment...")
    
    # Setup logging
    setup_logging()
    logging.info("Deployment started")
    
    # Create directories
    setup_directories()
    logging.info("Directories created")
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        logging.info(f"Installing missing packages: {missing_packages}")
        if not install_dependencies(missing_packages):
            logging.error("Failed to install dependencies")
            return False
    
    # Initialize databases
    if not initialize_databases():
        logging.error("Failed to initialize databases")
        return False
    
    # Create run script
    run_script = """
import streamlit as st
from src.dashboard.app import DashboardApp

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run()
    """
    
    with open('run_app.py', 'w') as f:
        f.write(run_script)
    
    logging.info("Deployment completed successfully")
    print("\nDeployment completed!")
    print("\nTo run the application:")
    print("1. Execute: streamlit run run_app.py")
    print("2. Open browser and navigate to: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    main() 