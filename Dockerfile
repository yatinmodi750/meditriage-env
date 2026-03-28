FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/yatinmodi750/meditriage-env"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging files first
COPY pyproject.toml .
COPY setup.py .

# Install all dependencies explicitly before package install
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy>=1.24
RUN pip install --no-cache-dir pydantic>=2.0
RUN pip install --no-cache-dir openai>=1.0
RUN pip install --no-cache-dir openenv-core>=0.2.0
RUN pip install --no-cache-dir fastapi uvicorn pyyaml gradio

# Copy source code
COPY meditriage_env/ ./meditriage_env/
COPY graders/ ./graders/
COPY server/ ./server/
COPY server.py .
COPY scripts/ ./scripts/
COPY openenv.yaml .
COPY README.md .

# Install the package itself
RUN pip install --no-cache-dir .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]