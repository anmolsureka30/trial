import subprocess
import sys
import os
from pathlib import Path
import logging
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Deployer:
    def __init__(self):
        self.python_cmd = sys.executable
        self.venv_path = Path('.venv')
        self.pip_cmd = str(self.venv_path / 'bin' / 'pip') if os.name != 'nt' else str(self.venv_path / 'Scripts' / 'pip')

    def check_python_version(self):
        """Check Python version compatibility"""
        if sys.version_info < (3, 9):
            logger.error("Python 3.9 or higher is required")
            return False
        return True

    def create_virtual_environment(self):
        """Create virtual environment"""
        try:
            if self.venv_path.exists():
                logger.info("Removing existing virtual environment...")
                shutil.rmtree(self.venv_path)
            
            logger.info("Creating virtual environment...")
            subprocess.run([self.python_cmd, '-m', 'venv', '.venv'], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False

    def install_requirements(self):
        """Install required packages"""
        try:
            # Upgrade pip
            logger.info("Upgrading pip...")
            subprocess.run([self.pip_cmd, 'install', '--upgrade', 'pip'], check=True)
            
            # Install requirements
            logger.info("Installing requirements...")
            subprocess.run([self.pip_cmd, 'install', '-r', 'requirements.txt'], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False

    def setup_project_structure(self):
        """Setup project directories and files"""
        try:
            # Create directories
            directories = [
                'data', 'logs', 'models', 'vectors',
                'src/data', 'src/dashboard', 'src/fraud_detection',
                'src/recommendations', 'src/data_management', 'src/config'
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                if directory.startswith('src'):
                    (Path(directory) / '__init__.py').touch()

            # Create .env if not exists
            if not Path('.env').exists():
                with open('.env', 'w') as f:
                    f.write('GEMINI_API_KEY=your_api_key_here\n')

            return True
        except Exception as e:
            logger.error(f"Failed to setup project structure: {e}")
            return False

    def verify_installation(self):
        """Verify installation"""
        try:
            # Test imports
            subprocess.run([
                self.python_cmd, '-c',
                'import streamlit; import torch; import langchain; import google.generativeai'
            ], check=True)
            return True
        except subprocess.CalledProcessError:
            logger.error("Installation verification failed")
            return False

    def deploy(self):
        """Run deployment process"""
        logger.info("Starting deployment process...")

        if not self.check_python_version():
            return False

        steps = [
            (self.create_virtual_environment, "Creating virtual environment"),
            (self.install_requirements, "Installing requirements"),
            (self.setup_project_structure, "Setting up project structure"),
            (self.verify_installation, "Verifying installation")
        ]

        for step_func, step_name in steps:
            logger.info(f"\nExecuting: {step_name}")
            if not step_func():
                logger.error(f"Deployment failed at: {step_name}")
                return False

        logger.info("\nDeployment completed successfully!")
        logger.info("\nTo run the application:")
        if os.name == 'nt':
            logger.info("1. .\\venv\\Scripts\\activate")
        else:
            logger.info("1. source .venv/bin/activate")
        logger.info("2. streamlit run run_app.py")
        return True

if __name__ == "__main__":
    deployer = Deployer()
    deployer.deploy() 