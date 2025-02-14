#!/bin/bash

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python install.py
    source .venv/bin/activate
fi

# Install package in development mode
pip install -e .

# Set library path for MacOS
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Kill any existing processes on ports 8000 and 8501
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null

# Create necessary directories
mkdir -p data/knowledge_base data/vectors logs models temp

# Initialize data files
python src/data/init_data.py

# Create sample knowledge base if empty
if [ ! "$(ls -A data/knowledge_base)" ]; then
    echo "Creating sample knowledge base..."
    python -c "
from pathlib import Path
from src.ai_services.gemini_service import GeminiService
service = GeminiService()
service._create_sample_knowledge_base(Path('data/knowledge_base'))
"
fi

# Start FastAPI server and wait for it to be ready
python -m uvicorn src.api.claims_api:app --reload --port 8000 &
sleep 5  # Give the server time to start

# Start Streamlit dashboard
streamlit run src/dashboard/enhanced_dashboard.py 