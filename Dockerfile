# syntax=docker/dockerfile:1.4
# Production-ready Dockerfile for Coarch

ARG PYTHON_VERSION=3.10-slim

# Build stage
FROM python:${PYTHON_VERSION} as builder

ARG FAISS_CPU_VERSION=1.7.4
ARG PYTORCH_VERSION=2.0.0
ARG TRANSFORMERS_VERSION=4.30.0

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    cmake \
    && pip download --no-deps --dest /wheels faiss-cpu==${FAISS_CPU_VERSION} \
    && pip install --no-cache-dir --find-links /wheels faiss-cpu==${FAISS_CPU_VERSION}

# Final stage
FROM python:${PYTHON_VERSION}

LABEL maintainer="syedazeez337"
LABEL description="Coarch - Local-first semantic code search engine"
LABEL version="1.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV COARCH_INDEX_PATH=/data/index
ENV COARCH_DB_PATH=/data/coarch.db
ENV COARCH_SERVER_HOST=0.0.0.0
ENV COARCH_SERVER_PORT=8000
ENV COARCH_LOG_LEVEL=INFO
ENV COARCH_ALLOWED_ORIGINS=http://localhost:3000

VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER 1000:1000

CMD ["python", "-m", "backend.server"]
