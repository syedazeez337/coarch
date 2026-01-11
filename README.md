# Coarch - Production-Ready Code Search Engine

A production-ready semantic code search engine with security, monitoring, and enterprise features.

## Security Features

- **Authentication**: API token-based authentication with configurable enable/disable
- **Rate Limiting**: Built-in rate limiter to prevent DoS attacks (60 req/min default)
- **Input Validation**: Comprehensive input sanitization for queries, paths, and parameters
- **Path Traversal Prevention**: Strict path validation and allowed directory enforcement
- **CORS Configuration**: Configurable CORS origins (not wildcarded by default)

## Production Features

- **Structured Logging**: JSON-formatted logs with configurable output
- **Health Checks**: `/health` endpoint for container orchestration
- **Metrics**: Prometheus-compatible `/metrics` endpoint
- **Request Tracing**: X-Request-ID and X-Process-Time headers
- **Error Handling**: Global exception handler with structured error responses

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize configuration
coarch init

# Index a repository
coarch index /path/to/your/repo

# Start the server
coarch serve --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COARCH_INDEX_PATH` | Path to FAISS index | `~/.coarch/index` |
| `COARCH_DB_PATH` | Path to SQLite database | `~/.coarch/coarch.db` |
| `COARCH_SERVER_HOST` | Server bind host | `0.0.0.0` |
| `COARCH_SERVER_PORT` | Server port | `8000` |
| `COARCH_LOG_LEVEL` | Logging level | `INFO` |
| `COARCH_LOG_JSON` | Use JSON logging | `false` |
| `COARCH_ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:3000` |
| `COARCH_RATE_LIMIT` | Requests per minute | `60` |
| `COARCH_ENABLE_AUTH` | Enable API authentication | `false` |
| `COARCH_API_KEY` | API key for authentication | `None` |

## API Endpoints

### Search

```bash
POST /search
{
  "query": "function that parses JSON",
  "language": "python",
  "limit": 10
}
```

### Health Check

```bash
GET /health
```

### Metrics

```bash
GET /metrics
```

## Docker

```bash
# Build
docker build -t coarch .

# Run
docker run -p 8000:8000 \
  -v coarch_data:/data \
  -e COARCH_ALLOWED_ORIGINS="http://localhost:3000" \
  coarch
```

## Configuration

Configuration is stored in `~/.coarch/config.json`:

```json
{
  "version": "1.0",
  "index_path": "~/.coarch/index",
  "db_path": "~/.coarch/coarch.db",
  "server_port": 8000,
  "rate_limit_per_minute": 60,
  "enable_auth": false,
  "cors_origins": ["http://localhost:3000"]
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Coarch Server (FastAPI)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Search    │  │   Index     │  │  Health/Metrics     │  │
│  │   API       │  │   Manager   │  │  /health /metrics   │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘  │
└─────────┼────────────────┼───────────────────────────────────┘
          │                │
    ┌─────▼─────┐    ┌─────▼─────┐
    │   FAISS   │    │  SQLite   │
    │  (HNSW)   │    │ (metadata)│
    └─────┬─────┘    └───────────┘
          │
    ┌─────▼─────┐
    │ CodeBERT  │
    │ Embeddings│
    └───────────┘
```

## Development

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black mypy pytest pytest-cov

# Run tests
pytest tests/ -v --cov=backend

# Format code
black .

# Type checking
mypy .
```

## Security Checklist for Production

- [ ] Set `COARCH_ENABLE_AUTH=true` and provide `COARCH_API_KEY`
- [ ] Configure `COARCH_ALLOWED_ORIGINS` with your frontend domain
- [ ] Set `COARCH_LOG_LEVEL=INFO` or higher
- [ ] Enable TLS/HTTPS on the server
- [ ] Configure rate limiting based on your needs
- [ ] Review and customize `IGNORE_DIRS` for your codebase

## License

MIT License
