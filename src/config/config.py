import os
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'insurance_claims',
    'user': os.getenv('DB_USER', 'default_user'),
    'password': os.getenv('DB_PASSWORD', 'default_password')
}

# Path configurations
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API configurations
API_TIMEOUT = 30
MAX_RETRIES = 3

# Model configurations
MODEL_PARAMS = {
    'confidence_threshold': 0.8,
    'max_sequence_length': 512,
    'batch_size': 32
}

# Model paths
MODEL_PATHS = {
    'fraud_detector': os.path.join(MODELS_DIR, 'fraud_detector.joblib'),
    'damage_model': os.path.join(MODELS_DIR, 'damage_model.joblib')
}

# Logging configuration
LOG_CONFIG = {
    'log_dir': LOGS_DIR,
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Application settings
APP_CONFIG = {
    'host': 'localhost',
    'port': 8501,
    'debug': True
} 