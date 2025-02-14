import os
import sys
from pathlib import Path
import streamlit as st
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/app.log'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and paths"""
    try:
        # Try importing dotenv
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not found, attempting to install...")
            import subprocess
            subprocess.run([
                sys.executable, "-m", "pip", "install", "python-dotenv==1.0.0"
            ], check=True)
            from dotenv import load_dotenv
            load_dotenv()
        
        # Add project root to Python path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        # Create necessary directories
        for directory in ['data', 'logs', 'models', 'vectors']:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        return True
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        return False

def main():
    """Main application entry point"""
    try:
        # Setup environment
        if not setup_environment():
            st.error("Failed to setup environment. Check logs for details.")
            return
        
        # Import after environment setup
        from src.dashboard.app import DashboardApp
        
        # Initialize and run dashboard
        dashboard = DashboardApp()
        dashboard.run()
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        st.error("Application failed to start. Please check logs for details.")

if __name__ == "__main__":
    main()
    