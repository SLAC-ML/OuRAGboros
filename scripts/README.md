# Scripts Directory

This directory contains utility scripts for managing and debugging the OuRAGboros deployment.

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

Explore and debug your OpenSearch knowledge base.

**Usage:**
```bash
# Show all document sources and counts
./scripts/opensearch-browser.sh docs

# View sample document content
./scripts/opensearch-browser.sh sample  

# Search for specific terms
./scripts/opensearch-browser.sh search "neural network"
./scripts/opensearch-browser.sh search "MicroBooNE"
./scripts/opensearch-browser.sh search "liquid argon"

# View OpenSearch indices overview
./scripts/opensearch-browser.sh indices

# Stop port forwarding when done
./scripts/opensearch-browser.sh cleanup
```

**Features:**
- Automatic port forwarding to OpenSearch service
- Document source analysis with chunk counts
- Full-text search with relevance scores
- Sample document preview
- Color-coded output for easy reading

**Debugging RAG Issues:**
Use this tool to:
1. Verify documents are properly indexed
2. Test search queries and see relevance scores
3. Understand what content is available for retrieval
4. Adjust score thresholds in Streamlit UI based on actual scores

## 🚀 Kubernetes Deployment

### `deploy-to-s3df.sh`

**Legacy:** Automated deployment script for S3DF vCluster (from previous development session).

**Note:** This script references old file paths and may need updates for the current organized directory structure.

### `monitor-resources.sh` & `production-monitor.sh`

**Legacy:** Resource monitoring scripts for Kubernetes deployments.

### `test-api-service.sh`

**Legacy:** API testing script for validating deployed services.

---

## 📝 Quick Reference

**Common Workflow:**
1. Build new image: `./scripts/build-and-push.sh`
2. Update `k8s/base/k8s.yaml` with new image tag
3. Deploy: `kubectl apply -n ouragboros -k k8s/base`
4. Debug knowledge base: `./scripts/opensearch-browser.sh docs`
5. Test search: `./scripts/opensearch-browser.sh search "your query"`

**Troubleshooting RAG:**
- If no documents are retrieved, check score threshold in Streamlit UI
- Use `opensearch-browser.sh search` to see actual relevance scores
- Typical good scores are > 1.0 for relevant matches
- Set score threshold in UI to 0.5-0.7 for initial testing