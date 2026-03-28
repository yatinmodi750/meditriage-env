FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/yatinmodi750/meditriage-env"

WORKDIR /app

# Copy all project files
COPY setup.py .
COPY meditriage_env/ ./meditriage_env/
COPY graders/ ./graders/
COPY scripts/ ./scripts/
COPY server.py .
COPY openenv.yaml .
COPY README.md .

# Install all dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    pydantic>=2.0 \
    openai>=1.0 \
    pyyaml \
    gradio \
    fastapi \
    uvicorn

# Install the package
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 7860

# Run FastAPI server (OpenEnv API) on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]