#!/bin/bash



# Start your REST API 
uv run uvicorn app_api:app --host 0.0.0.0 --port 8001 --reload &

# Start the Streamlit app
uv run streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0 &

# Wait for any process to exit
wait -n

# Exit with the status of the process that exited first
#exit $?