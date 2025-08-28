# Scripts Directory

This directory contains utility scripts for local development, deployment, and debugging OuRAGboros.

## 🚀 Local Development

### `local-dev.sh`

**New!** Streamlined script for local development with Docker Compose. Perfect for rapid development and testing.

**Quick Start:**
```bash
# Rebuild ouragboros and start all services  
./scripts/local-dev.sh

# Just rebuild the app container
./scripts/local-dev.sh build

# Rebuild without cache (if having issues)
./scripts/local-dev.sh build --no-cache

# Quick restart after code changes
./scripts/local-dev.sh restart

# View logs for debugging
./scripts/local-dev.sh logs

# Stop services  
./scripts/local-dev.sh down

# Full cleanup (stop + remove containers/images)
./scripts/local-dev.sh clean
```

**Features:**
- ✅ **Automatic prerequisites check** (Docker, compose files, .env)  
- ✅ **Smart build options** (app-only vs full rebuild, cache control)
- ✅ **Service management** (start, stop, restart, status)
- ✅ **Log monitoring** with real-time follow
- ✅ **Cleanup utilities** for housekeeping
- ✅ **Clear service URLs** displayed after startup

**Service URLs:**
- Streamlit UI: http://localhost:8501
- REST API: http://localhost:8001  
- OpenSearch: http://localhost:9200
- Ollama: http://localhost:11434

## 🐳 Docker Image Management

### `build-and-push.sh`

Builds and pushes Docker images with date-based tagging.

**Usage:**
```bash
# Build with today's date (e.g., slacml/ouragboros:25.08.26)
./scripts/build-and-push.sh

# Build with suffix for multiple builds in same day
./scripts/build-and-push.sh 2  # Creates slacml/ouragboros:25.08.26-2
```

**Features:**
- Automatic date-based tagging (`YY.MM.DD` format)
- Support for build suffixes (`-1`, `-2`, etc.)
- Provides deployment instructions after successful build

## 🔍 OpenSearch Database Browser

### `opensearch-browser.sh`

Comprehensive tool for exploring and debugging your OpenSearch knowledge bases with full multi-KB support.

**Core Commands:**
```bash
# List all available knowledge bases
./scripts/opensearch-browser.sh kbs

# View OpenSearch indices overview
./scripts/opensearch-browser.sh indices

# Get quick entry count for a knowledge base
./scripts/opensearch-browser.sh count                    # All KBs
./scripts/opensearch-browser.sh count default            # Default KB only
./scripts/opensearch-browser.sh count physics_papers     # Custom KB

# Show document sources and detailed counts
./scripts/opensearch-browser.sh docs                     # All KBs
./scripts/opensearch-browser.sh docs default             # Default KB only
./scripts/opensearch-browser.sh docs physics_papers      # Custom KB

# View sample document content
./scripts/opensearch-browser.sh sample                   # All KBs
./scripts/opensearch-browser.sh sample default           # Default KB only
./scripts/opensearch-browser.sh sample physics_papers    # Custom KB

# Search with full-text search and relevance scores
./scripts/opensearch-browser.sh search "neural network"                    # All KBs
./scripts/opensearch-browser.sh search "quantum mechanics" physics_papers  # Specific KB
./scripts/opensearch-browser.sh search "liquid argon" default              # Default KB

# Clean up port forwarding when done
./scripts/opensearch-browser.sh cleanup
```

**Knowledge Base Support:**
- **Multi-KB aware**: All commands support optional knowledge base parameter
- **Automatic detection**: Discovers knowledge bases from OpenSearch indices
- **Default KB**: Use `default` for the original knowledge base
- **Custom KBs**: Use names like `physics_papers`, `legal_docs`, etc.
- **Cross-KB search**: Omit KB parameter to search across all knowledge bases

**Quick Inspection Workflow:**
```bash
# 1. See what knowledge bases exist
./scripts/opensearch-browser.sh kbs

# 2. Check entry counts (great for quick verification)
./scripts/opensearch-browser.sh count physics_papers
./scripts/opensearch-browser.sh count default

# 3. View document sources and detailed breakdown  
./scripts/opensearch-browser.sh docs physics_papers

# 4. Test search functionality and see scores
./scripts/opensearch-browser.sh search "quantum" physics_papers
```

**Features:**
- ✅ **Automatic port forwarding** to OpenSearch service
- ✅ **Multi-knowledge base support** with automatic discovery
- ✅ **Quick entry counting** for rapid verification
- ✅ **Document source analysis** with per-file chunk counts
- ✅ **Full-text search** with relevance scores and ranking
- ✅ **Sample document preview** with metadata
- ✅ **Color-coded output** for easy reading
- ✅ **Cross-KB operations** when KB parameter is omitted

**Debugging RAG Issues:**
Use this tool to:
1. **Verify KB isolation**: Ensure documents are in the correct knowledge base
2. **Check entry counts**: Quick verification that uploads succeeded
3. **Test search queries**: See actual relevance scores for your queries
4. **Understand content**: Preview what documents are available for retrieval
5. **Adjust score thresholds**: Use search scores to set appropriate thresholds in Streamlit UI
6. **Validate KB creation/deletion**: Confirm knowledge base operations worked correctly

## 🧪 Testing Utilities

### `test-local.sh`, `test-embedding-cache.sh`, `test-concurrent.sh`

Various testing scripts for validating functionality:
- **test-local.sh**: Local development testing
- **test-embedding-cache.sh**: Embedding cache performance testing  
- **test-concurrent.sh**: Concurrent request testing

---

## 📝 Quick Reference

**Local Development Workflow:**
1. Make code changes
2. Rebuild and restart: `./scripts/local-dev.sh restart`
3. Test at: http://localhost:8501
4. Debug with logs: `./scripts/local-dev.sh logs`

**Production Deployment Workflow:**
1. Build new image: `./scripts/build-and-push.sh [build-number]`
2. Update `k8s/base/k8s.yaml` with new image tag
3. Deploy: `kubectl apply -n ouragboros -k k8s/base`
4. Inspect knowledge bases: `./scripts/opensearch-browser.sh kbs`
5. Check entry counts: `./scripts/opensearch-browser.sh count [kb_name]`
6. Test search: `./scripts/opensearch-browser.sh search "query" [kb_name]`

**Knowledge Base Management:**
- **Create/Delete**: Use Streamlit UI to create and delete knowledge bases
- **List KBs**: `./scripts/opensearch-browser.sh kbs`
- **Quick Count**: `./scripts/opensearch-browser.sh count physics_papers`
- **Detailed Info**: `./scripts/opensearch-browser.sh docs physics_papers`

**Troubleshooting RAG:**
- **No documents retrieved**: Check score threshold in Streamlit UI (try 0.5-0.7)
- **Wrong KB results**: Verify correct knowledge base is selected in UI
- **Search testing**: Use `./scripts/opensearch-browser.sh search "query" kb_name`
- **Typical scores**: > 1.0 for relevant matches, > 0.5 for marginal matches
- **KB isolation**: Ensure documents uploaded to intended knowledge base