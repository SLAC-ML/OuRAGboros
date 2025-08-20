# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OuRAGboros is a Stanford ICME research RAG (Retrieval-Augmented Generation) application built with Python, Streamlit, and LangChain. It provides both a web UI and REST API for document retrieval and question answering using multiple LLM providers (Ollama, OpenAI) and embedding models.

## Development Commands

### Primary Development
```bash
# Start the Streamlit web application
uv run streamlit run src/main.py

# Install dependencies 
uv sync

# Code linting (configured in pyproject.toml)
uv run ruff check
uv run ruff format
```

### Docker Development
```bash
# Start full stack with OpenSearch vector database (Intel Mac compatible)
docker compose up

# For GPU-enabled systems (NVIDIA only)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up

# Build and push Docker images (for releases)
docker compose build --push

# Rebuild containers after code changes (recommended after UI updates)
docker compose build --no-cache

# Stop all services
docker compose down
```

### Command Line Tools
```bash
# Extract and chunk PDF documents
uv run chunk_pdf <pdf_path> --outpath <output_dir>

# Fine-tune embedding models
uv run finetune_model <text_file> --outpath ./models --base-model <model_name> --tuned-model-name <name>
```

## Architecture

### Core Components

- **src/main.py**: Streamlit web application entry point
- **src/app_api.py**: FastAPI REST API server with CORS support
- **src/lib/rag_service.py**: Central RAG orchestration - handles document retrieval and LLM queries
- **src/lib/config.py**: Environment configuration management using `.default.env`

### LangChain Integration (`src/lib/langchain/`)
- **llm.py**: Multi-provider LLM interface (Ollama, OpenAI) with streaming support
- **embeddings.py**: HuggingFace and other embedding model management
- **models.py**: Model downloading and caching logic
- **opensearch.py**: OpenSearch vector store integration
- **util.py**: Model name parsing utilities

### Document Processing (`src/lib/pdf/`)
- **nougat_extractor.py**: Facebook Nougat model for academic PDF parsing
- **extractor.py**: General PDF text extraction interface

### Tools (`src/tools/`)
- **chunk_pdf.py**: CLI tool for offline PDF processing
- **finetune.py**: CLI tool for embedding model fine-tuning

## Configuration

### Environment Variables
Key configuration in `.env` (for Docker Compose) and `.default.env` (for uv development):
- `OLLAMA_BASE_URL`: Ollama server endpoint
- `OLLAMA_MODEL_DEFAULT`: Default Ollama model
- `OPENSEARCH_BASE_URL`: OpenSearch instance URL
- `HUGGINGFACE_EMBEDDING_MODEL_DEFAULT`: Default embedding model
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `NVIDIA_GPUS`: Set to `0` for Intel Mac compatibility, `1` or higher for GPU systems

### Intel Mac Compatibility
The Docker Compose setup has been configured for Intel Mac compatibility:
- GPU requirements removed from base configuration
- Use `docker-compose.gpu.yml` override for NVIDIA GPU systems
- OpenSearch configured with appropriate memory limits for local development

### Model Storage
- Local models cached in `models/` directory
- HuggingFace models use `SENTENCE_TRANSFORMERS_HOME` path
- Ollama models managed by Ollama service

## Vector Store Strategy

The application supports dual vector store modes:
1. **In-memory**: FAISS-based, data lost on restart
2. **OpenSearch**: Persistent storage with advanced search capabilities

Vector store selection controlled by `USE_OPENSEARCH` toggle in UI or `use_opensearch` parameter in API.

## Knowledge Base Management

**Multiple Knowledge Bases**: The application supports multiple isolated knowledge bases:
- Each knowledge base contains its own set of documents and embeddings
- Knowledge bases are completely isolated from each other
- Useful for organizing different document types or testing different embedding models
- OpenSearch index format: `{prefix}_{kb_name}_{vector_size}_{model_hash}`
- Backward compatible with existing indices (treated as "default" knowledge base)

**Creating Knowledge Bases**:
1. Use the "Create New Knowledge Base" section in the sidebar
2. Knowledge base names must contain only letters, numbers, and underscores
3. Available on both main chat page and document embedding page

**Switching Knowledge Bases**:
- Select from dropdown in sidebar on any page
- Documents uploaded to one knowledge base won't appear in queries to another
- Each knowledge base maintains separate document counts and embeddings

**Knowledge Base Deletion**:
- Uses modal dialog confirmation to prevent UI freezing issues
- Cannot delete the "default" knowledge base
- Deletion confirmations are handled outside the sidebar to avoid Streamlit state conflicts

## Deployment

### Kubernetes/Helm
- Helm charts in `docker-compose/` directory
- Generated from `docker-compose.yml` using Kompose
- Support for NVIDIA GPU acceleration
- S3DF cluster deployment configuration included

### Development Workflow
1. Modify `docker-compose.yml` for new releases
2. Run `kompose convert -c && helm template --values ./docker-compose/values.yaml ./docker-compose > k8s.yaml`
3. Deploy with `kubectl apply --namespace ouragboros -k .`

## API Endpoints

### REST API (`/ask`)
- Accepts: query, embedding model, LLM model, files, chat history
- Returns: answer text and document snippets
- Supports both RAG and non-RAG modes
- CORS enabled for cross-origin requests

## User Interface

### Recent UI Improvements (2025-01)
The Streamlit interface has been streamlined for better user experience:

**Simplified Layout**:
- Removed redundant "Vector Store" and "Models" section headers from document embedding page
- Knowledge Base management now has its own dedicated container with clear visual hierarchy
- Cleaner, more professional appearance without unnecessary section divisions

**Minimalist Design**:
- Replaced emoji icons with clean text labels for better accessibility and professional appearance
- Button labels are now clear and descriptive: "Create", "Delete", "Cancel", "Yes, delete"
- Consistent text-based UI elements throughout the application

**Improved User Experience**:
- Modal dialogs for knowledge base deletion prevent UI freezing issues
- Confirmation dialogs are properly isolated from sidebar to avoid Streamlit state conflicts
- Better handling of configuration changes that trigger app reruns
- Knowledge base ordering: "default" always appears first, others in creation order (not alphabetical)
- Enhanced knowledge base management with proper validation and error handling
- Consistent success message placement at page top for all operations
- Improved delete button width and cancel functionality
- Fixed delete confirmation dialog to show correct knowledge base names

**Docker Integration**:
- Hot reloading enabled with `--server.runOnSave=true` in Dockerfile for development
- Docker Compose watch mode automatically syncs code changes without rebuilds
- For dependency changes: `docker compose build --no-cache && docker compose up -d`
- All improvements are automatically included in the containerized application

**Development Workflow** (2025-08):
- Streamlit hot reloading works in Docker environment for rapid development
- Use `docker compose up --watch` for automatic file syncing during development
- Code changes in `src/` directory are immediately reflected without manual rebuilds
- Only dependency changes (uv.lock) require container rebuilds

## Dependencies

- **uv**: Python package management and virtual environments
- **LangChain**: LLM orchestration framework
- **Streamlit**: Web UI framework
- **FastAPI**: REST API framework
- **OpenSearch**: Vector database (optional)
- **Ollama**: Local LLM serving
- **Transformers/HuggingFace**: Embedding models