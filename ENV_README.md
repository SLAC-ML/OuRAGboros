# Environment Configuration

This project uses two environment files for different deployment scenarios:

## Environment Files

### `.env` - Docker Compose Configuration
Used when running with `docker-compose up` or `./scripts/local-dev.sh`

- **Service URLs**: Uses Docker service names (e.g., `http://qdrant:6333`)
- **Purpose**: Multi-container Docker environment
- **Usage**: Automatically loaded by Docker Compose

### `.env.local` - Local Development Configuration  
Used when running applications directly with `uv run` on localhost

- **Service URLs**: Uses localhost (e.g., `http://localhost:6333`)  
- **Purpose**: Local development with external services
- **Usage**: Manually source or copy to `.env` when needed

## Quick Start

### Docker Compose (Recommended)
```bash
# Uses .env automatically
./scripts/local-dev.sh
```

### Local Development
```bash
# Copy localhost config and start services separately
cp .env.local .env
uv run streamlit run src/main.py
```

## Key Configuration

- `PREFER_QDRANT=true` - Use Qdrant for vector storage
- `PREFER_OPENSEARCH=false` - Disable OpenSearch  
- `QDRANT_BASE_URL` - Qdrant connection URL
- `STANFORD_API_KEY` - Add your Stanford AI API key for testing

## Service Endpoints (Docker Compose)

- **Streamlit UI**: http://localhost:8501
- **REST API**: http://localhost:8001
- **Qdrant**: http://localhost:6333  
- **OpenSearch**: http://localhost:9200
- **Ollama**: http://localhost:11434