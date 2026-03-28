FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/yatinmodi750/meditriage-env"

WORKDIR /app

# Copy all project files
COPY setup.py .
COPY meditriage_env/ ./meditriage_env/
COPY graders/ ./graders/
COPY scripts/ ./scripts/
COPY openenv.yaml .
COPY README.md .

# Install all dependencies
RUN pip install --no-cache-dir numpy>=1.24 pydantic>=2.0 openai>=1.0 pyyaml gradio

# Install the package
RUN pip install --no-cache-dir .

# Expose Gradio port (HF Spaces default)
EXPOSE 7860

# Run the Gradio demo app
CMD ["python", "scripts/demo_app.py"]