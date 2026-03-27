FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/yatinmodi750/meditriage-env"

WORKDIR /app

# Install dependencies
COPY setup.py .
COPY meditriage_env/ ./meditriage_env/
COPY graders/ ./graders/
COPY scripts/ ./scripts/
COPY openenv.yaml .
COPY README.md .

RUN pip install --no-cache-dir numpy>=1.24 pyyaml gradio

# Install the package in editable mode
RUN pip install -e .

# Expose Gradio port (HF Spaces default)
EXPOSE 7860

# Run the Gradio demo app
CMD ["python", "scripts/demo_app.py"]