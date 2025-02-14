#!/bin/bash

echo "Restarting Insurance Claims System..."

# Stop the system
./stop.sh

# Wait a moment
sleep 2

# Start the system
./start.sh 