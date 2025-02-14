import streamlit as st
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
import traceback

# Setup logging with both file and console handlers
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_streamlit():
    """Configure Streamlit page settings"""
    try:
        st.set_page_config(
            page_title="Insurance Claims Dashboard",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        logger.error(f"Failed to setup Streamlit: {e}")
        raise

def setup_environment():
    """Setup environment and dependencies"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Verify API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return False
        
        # Add project root to Python path
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        
        # Create necessary directories
        directories = ['data', 'logs', 'models', 'vectors', 'src/data']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        return False

def verify_dependencies():
    """Verify all required dependencies are available"""
    try:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import google.generativeai
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def main():
    """Main application entry point"""
    try:
        logger.info("Starting application...")
        
        # Setup environment
        if not setup_environment():
            st.error("Failed to setup environment. Check logs for details.")
            return
        
        # Verify dependencies
        if not verify_dependencies():
            st.error("Missing required dependencies. Please install all requirements.")
            return
        
        # Setup Streamlit
        setup_streamlit()
        
        # Import dashboard after environment setup
        from src.dashboard.app import DashboardApp
        
        # Initialize and run dashboard
        dashboard = DashboardApp()
        dashboard.run()
        
        logger.info("Application started successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(
            "Application failed to start. Please check logs for details.\n\n"
            f"Error: {str(e)}"
        )

if __name__ == "__main__":
    main() 