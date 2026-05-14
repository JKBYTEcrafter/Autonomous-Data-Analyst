#!/bin/bash
# Start FastAPI backend in the background
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for backend to start
sleep 3

# Start Streamlit frontend on port 7860 (HF Spaces default)
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
