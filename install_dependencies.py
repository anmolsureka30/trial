import subprocess
import sys
import os
from pathlib import Path

def install_base_requirements():
    """Install base requirements needed for setup"""
    base_packages = [
        "setuptools>=65.5.1",
        "wheel>=0.38.4",
        "pip>=23.0.1",
        "python-dotenv==1.0.0"
    ]
    
    print("Installing base requirements...")
    for package in base_packages:
        try:
            subprocess.run([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--upgrade",
                package
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path('.venv')
    
    # Create virtual environment
    if not venv_path.exists():
        try:
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
            print("Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False
    
    # Get the pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / 'Scripts' / 'pip'
    else:  # Unix/MacOS
        pip_path = venv_path / 'bin' / 'pip'
    
    # Upgrade pip in virtual environment
    try:
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e}")
        return False
    
    return True

def install_project_requirements():
    """Install project requirements"""
    if os.name == 'nt':  # Windows
        pip_path = '.venv/Scripts/pip'
    else:  # Unix/MacOS
        pip_path = '.venv/bin/pip'
    
    try:
        # Install requirements
        subprocess.run([
            pip_path,
            "install",
            "-r",
            "requirements.txt"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write('GEMINI_API_KEY=your_api_key_here\n')
        print(".env file created")

def main():
    print("Starting installation process...")
    
    if not install_base_requirements():
        print("Failed to install base requirements")
        return
    
    if not setup_virtual_environment():
        print("Failed to setup virtual environment")
        return
    
    if not install_project_requirements():
        print("Failed to install project requirements")
        return
    
    create_env_file()
    
    print("\nInstallation completed successfully!")
    print("\nTo run the application:")
    if os.name == 'nt':
        print("1. .\\venv\\Scripts\\activate")
    else:
        print("1. source .venv/bin/activate")
    print("2. streamlit run run_app.py")

if __name__ == "__main__":
    main() 