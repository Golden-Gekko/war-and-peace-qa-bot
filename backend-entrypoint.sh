#!/bin/sh
set -e

cd /app

if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db)" ]; then
    echo "Chroma DB not found or empty. Running db_filling.py..."
    python db_filling.py
else
    echo "Chroma DB already exists. Skipping db_filling."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000