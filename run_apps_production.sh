#!/bin/bash

# Production configuration with high worker count for massive single-pod scaling

# Start REST API with many workers (utilize all cores on node)
# Each worker can handle ~1.5-2 req/sec, so 32 workers = 48-64 req/sec
uv run uvicorn app_api:app --host 0.0.0.0 --port 8001 --workers 32 &

# Start the Streamlit app (single process is fine for UI)
uv run streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0 &

# Wait for any process to exit
wait -n