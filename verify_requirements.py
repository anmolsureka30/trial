import subprocess
import pkg_resources
import sys
from pathlib import Path

def get_installed_packages():
    """Get dictionary of installed packages and their versions"""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def check_requirements():
    """Check and verify all required packages"""
    requirements = {
        # Core dependencies
        'streamlit': '1.24.0',
        'pandas': '1.5.3',
        'numpy': '1.24.3',
        
        # Visualization
        'plotly': '5.13.1',
        'matplotlib': '3.7.1',
        'seaborn': '0.12.2',
        'wordcloud': '1.9.2',
        'networkx': '3.0',
        
        # Machine Learning & AI
        'scikit-learn': '1.2.2',
        'google-generativeai': '0.3.2',
        'sentence-transformers': '2.2.2',
        'transformers': '4.36.2',
        
        # LangChain & Vector Store
        'langchain': '0.1.0',
        'langchain-community': '0.0.10',
        'chromadb': '0.4.22',
        
        # Vector Search
        'faiss-cpu': '1.7.4',
        
        # Text Processing
        'rapidfuzz': '3.0.0',
        'nltk': '3.8.1',
        
        # Database
        'sqlalchemy': '2.0.25',
        
        # Utilities
        'python-dotenv': '1.0.0',
        'joblib': '1.2.0',
        'pyyaml': '6.0.1',
        
        # PyTorch
        'torch': '2.0.0'
    }
    
    installed_packages = get_installed_packages()
    missing_packages = []
    version_mismatch = []
    
    print("\nChecking package requirements...")
    print("-" * 50)
    
    for package, required_version in requirements.items():
        if package not in installed_packages:
            missing_packages.append(package)
            print(f"❌ {package}: Not installed (Required: {required_version})")
        else:
            installed_version = installed_packages[package]
            if installed_version != required_version:
                version_mismatch.append((package, installed_version, required_version))
                print(f"⚠️  {package}: Version mismatch (Installed: {installed_version}, Required: {required_version})")
            else:
                print(f"✅ {package}: {installed_version}")
    
    print("\nSummary:")
    print("-" * 50)
    if missing_packages:
        print(f"\nMissing Packages ({len(missing_packages)}):")
        for package in missing_packages:
            print(f"- {package}")
    
    if version_mismatch:
        print(f"\nVersion Mismatches ({len(version_mismatch)}):")
        for package, installed, required in version_mismatch:
            print(f"- {package}: Installed={installed}, Required={required}")
    
    if not (missing_packages or version_mismatch):
        print("All packages are correctly installed!")
    
    return not (missing_packages or version_mismatch)

def install_missing_packages():
    """Install missing packages"""
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def main():
    """Main verification function"""
    print("Starting package verification...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("Error: requirements.txt not found!")
        return False
    
    # First check
    if check_requirements():
        print("\nAll requirements are satisfied!")
        return True
    
    # Ask to install missing packages
    response = input("\nWould you like to install/update missing packages? (y/n): ")
    if response.lower() == 'y':
        print("\nInstalling missing packages...")
        if install_missing_packages():
            print("\nRe-checking requirements...")
            if check_requirements():
                print("\nAll packages have been successfully installed!")
                return True
            else:
                print("\nSome packages could not be installed correctly.")
                return False
    
    return False

if __name__ == "__main__":
    main() 