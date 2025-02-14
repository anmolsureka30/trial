import subprocess
import sys
import platform
import os
from pathlib import Path

def install_dependencies():
    """Install all required dependencies"""
    try:
        # Upgrade pip and setuptools first
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "pip", "setuptools", "wheel"
        ], check=True)
        
        # Install dependencies using pip's --prefer-binary flag
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--prefer-binary",  # Prefer pre-built wheels
            "-r", "requirements.txt"
        ], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    try:
        directories = [
            'data',
            'data/knowledge_base',
            'data/uploads',
            'data/vectors',
            'logs',
            'models',
            'temp',
            'src/data',
            'src/dashboard',
            'src/fraud_detection',
            'src/recommendations',
            'src/data_management',
            'src/config'
        ]
        
        for directory in directories:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files in src directories
            if directory.startswith('src'):
                init_file = path / '__init__.py'
                init_file.touch(exist_ok=True)
        
        print("Directory setup completed successfully")
        return True
    except Exception as e:
        print(f"Error setting up directories: {e}")
        return False

def create_virtual_environment():
    """Create and setup virtual environment"""
    try:
        venv_path = Path('.venv')
        if not venv_path.exists():
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
            print("Virtual environment created successfully")
        else:
            print("Virtual environment already exists")
        return True
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        return False

def main():
    """Main installation process"""
    print("Starting installation process...")
    
    # Create virtual environment
    if not create_virtual_environment():
        return
    
    # Get the correct python and pip paths
    if platform.system() == "Windows":
        python_path = ".venv\\Scripts\\python"
        pip_path = ".venv\\Scripts\\pip"
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        python_path = ".venv/bin/python"
        pip_path = ".venv/bin/pip"
        activate_cmd = "source .venv/bin/activate"
    
    print(f"\nPlease activate the virtual environment:")
    print(f"  {activate_cmd}")
    
    # Setup directories
    if not setup_directories():
        return
    
    print("\nInstalling dependencies...")
    success = install_dependencies()
    
    if success:
        print("\nInstallation completed successfully!")
        print("\nTo run the application:")
        print(f"1. {activate_cmd}")
        print("2. python test_system.py")
    else:
        print("\nInstallation failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 