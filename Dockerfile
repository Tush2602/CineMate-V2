FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy source code only — data/models mounted separately
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY db/ ./db/

EXPOSE 8000 8501

CMD sh -c "uvicorn api.app:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0"
