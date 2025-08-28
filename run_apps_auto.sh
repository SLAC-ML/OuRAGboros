#!/bin/bash

# Auto-detect environment and run with appropriate worker count
# Production nodes have 64 cores, local machines typically have 4-16

# Get CPU count
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Calculate worker count (reduced for better worker management)
if [ "$CPU_COUNT" -ge 32 ]; then
    # Production environment with many cores - reduced from 32 to 8 workers
    WORKER_COUNT=8
    echo "🚀 Production mode: Starting with $WORKER_COUNT workers on $CPU_COUNT cores (optimized)"
else
    # Local development with fewer cores
    WORKER_COUNT=$(( (CPU_COUNT + 1) / 2 ))  # Use half the cores
    echo "💻 Development mode: Starting with $WORKER_COUNT workers on $CPU_COUNT cores"
fi

# Start REST API with calculated workers
uv run uvicorn app_api:app --host 0.0.0.0 --port 8001 --workers $WORKER_COUNT &

# Start the Streamlit app (single process)
uv run streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0 &

# Wait for any process to exit
wait -n