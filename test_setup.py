import os
import sys
from pathlib import Path
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def run_tests():
    """Run basic setup tests"""
    # Check required packages
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'dotenv',
        'torch',
        'langchain'
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    # Check directories
    required_dirs = ['data', 'logs', 'models', 'vectors']
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    
    # Check .env file
    if not Path('.env').exists():
        print(".env file missing")
        return False
    
    print("All setup tests passed successfully!")
    return True

if __name__ == "__main__":
    run_tests() 