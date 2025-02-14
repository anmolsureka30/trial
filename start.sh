#!/bin/bash

# Ensure script fails on any error
set -e

echo "Starting Insurance Claims System..."

# Function to cleanup processes
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "uvicorn" 2>/dev/null || true
    pkill -f "streamlit" 2>/dev/null || true
    rm -f api.log
}

# Set trap for cleanup on script exit
trap cleanup EXIT

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $port is in use. Stopping existing process..."
        lsof -ti :$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Activate virtual environment
echo "Activating virtual environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Running initial setup..."
    python init_setup.py
    source .venv/bin/activate
fi

# Set environment variables
echo "Setting environment variables..."
export PYTHONPATH=.
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Create necessary directories
echo "Setting up directories..."
mkdir -p data/{knowledge_base,vectors,uploads} logs models temp

# Initialize data if needed
echo "Checking data files..."
if [ ! -f "data/Primary_Parts_Code.csv" ] || [ ! -f "data/historical_claims.csv" ]; then
    echo "Initializing data files..."
    python src/data/init_data.py
fi

# Check and free required ports
check_port 8000
check_port 8501

# Start FastAPI server
echo "Starting API server..."
uvicorn src.api.claims_api:app --reload --port 8000 > api.log 2>&1 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API server to start..."
max_retries=30
count=0
while ! curl -s http://localhost:8000/health > /dev/null && [ $count -lt $max_retries ]; do
    if ! ps -p $API_PID > /dev/null; then
        echo "API process died. Check api.log for details:"
        cat api.log
        exit 1
    fi
    sleep 1
    count=$((count + 1))
    echo "Waiting... ($count/$max_retries)"
done

if [ $count -eq $max_retries ]; then
    echo "API failed to start properly. Check api.log for details:"
    cat api.log
    kill $API_PID 2>/dev/null || true
    exit 1
fi

echo "API is ready!"

# Start Streamlit dashboard
echo "Starting dashboard..."
streamlit run src/dashboard/enhanced_dashboard.py 