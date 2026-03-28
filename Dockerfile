FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/yatinmodi750/meditriage-env"

WORKDIR /app

COPY pyproject.toml .
COPY setup.py .
COPY meditriage_env/ ./meditriage_env/
COPY graders/ ./graders/
COPY server/ ./server/
COPY server.py .
COPY scripts/ ./scripts/
COPY openenv.yaml .
COPY README.md .

RUN pip install --no-cache-dir \
    numpy>=1.24 \
    pydantic>=2.0 \
    openai>=1.0 \
    openenv-core>=0.2.0 \
    pyyaml \
    gradio \
    fastapi \
    uvicorn

RUN pip install --no-cache-dir .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]