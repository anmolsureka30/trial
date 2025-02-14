#!/bin/bash

echo "Stopping Insurance Claims System..."

# Kill processes
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true

# Remove temporary files
rm -f api.log

echo "System stopped successfully" 