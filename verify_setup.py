import importlib
import sys
from pathlib import Path

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    required_modules = [
        'dotenv',
        'streamlit',
        'pandas',
        'numpy',
        'setuptools'
    ]
    
    all_good = True
    for module in required_modules:
        if check_import(module):
            print(f"✅ {module} is installed")
        else:
            print(f"❌ {module} is NOT installed")
            all_good = False
    
    if not all_good:
        print("\nSome modules are missing. Please run:")
        print("python install_dependencies.py")
        sys.exit(1)
    
    print("\nAll required modules are installed!")

if __name__ == "__main__":
    main() 