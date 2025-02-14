import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Setup the development environment"""
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        
        # Get pip path
        if sys.platform == "win32":
            pip_path = ".venv\\Scripts\\pip"
        else:
            pip_path = ".venv/bin/pip"
        
        # Upgrade pip
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_path, "install", "-e", "."], check=True)
        
        print("Environment setup completed successfully!")
        return True
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/knowledge_base",
        "data/vectors",
        "logs",
        "models",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    setup_environment() 