import subprocess
import sys
from pathlib import Path

def fix_environment():
    """Fix common environment issues"""
    print("Starting environment fix...")
    
    # Install python-dotenv explicitly
    print("Installing python-dotenv...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "python-dotenv==1.0.0"
    ], check=True)
    
    # Verify installation
    try:
        import dotenv
        print("python-dotenv installed successfully!")
    except ImportError:
        print("Error: Failed to import dotenv after installation")
        return False
    
    # Create/update .env file if needed
    env_file = Path('.env')
    if not env_file.exists():
        print("Creating .env file...")
        with open(env_file, 'w') as f:
            f.write('GEMINI_API_KEY=your_api_key_here\n')
    
    print("Environment fix completed!")
    return True

if __name__ == "__main__":
    fix_environment() 