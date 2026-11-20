#!/bin/bash
echo "Starting Smart Crowd AI in Streamlit..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
streamlit run app.py --server.port 8501
