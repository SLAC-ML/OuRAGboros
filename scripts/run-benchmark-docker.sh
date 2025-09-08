#!/bin/bash

# Run benchmark scripts using Docker to avoid local Python environment issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
BENCHMARK_SCRIPT="${1:-test-mock-validation.py}"
USE_NETWORK="${USE_NETWORK:-host}"

echo "🐳 Running benchmark script in Docker container..."
echo "Script: $BENCHMARK_SCRIPT"
echo "Network: $USE_NETWORK"

# Build a minimal Python image with required dependencies
cat > "$PROJECT_ROOT/Dockerfile.benchmark" << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    apache2-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for benchmarking
RUN pip install --no-cache-dir \
    requests \
    psutil \
    numpy \
    pandas

# Copy source code (needed for imports)
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Set Python path
ENV PYTHONPATH=/app

ENTRYPOINT ["python3"]
EOF

# Build the benchmark image
echo "📦 Building benchmark Docker image..."
docker build -f "$PROJECT_ROOT/Dockerfile.benchmark" -t ouragboros-benchmark:latest "$PROJECT_ROOT"

# Run the benchmark script
echo "🚀 Running benchmark script..."
docker run --rm \
    --network="$USE_NETWORK" \
    -e USE_MOCK_EMBEDDINGS="${USE_MOCK_EMBEDDINGS:-false}" \
    -e USE_MOCK_LLM="${USE_MOCK_LLM:-false}" \
    -e LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://host.docker.internal:8001}" \
    -e K8S_BASE_URL="${K8S_BASE_URL:-http://host.docker.internal:8501}" \
    -v "$PROJECT_ROOT/benchmark-results-systematic:/app/benchmark-results-systematic" \
    -v "$PROJECT_ROOT/benchmark-results-profiled:/app/benchmark-results-profiled" \
    ouragboros-benchmark:latest \
    "/app/scripts/$BENCHMARK_SCRIPT"

# Clean up
rm -f "$PROJECT_ROOT/Dockerfile.benchmark"

echo "✅ Benchmark complete!"