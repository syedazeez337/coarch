# syntax=docker/dockerfile:1.4

ARG PYTHON_VERSION=3.10-slim

# Build stage
FROM python:${PYTHON_VERSION} as builder

ARG FAISS_CPU_VERSION=1.7.4

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

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV COARCH_INDEX_PATH=/data/index
ENV COARCH_DB_PATH=/data/coarch.db
ENV COARCH_SERVER_HOST=0.0.0.0
ENV COARCH_SERVER_PORT=8000

VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/index/status')" || exit 1

CMD ["python", "-m", "backend.server"]
