#!/bin/bash

# Run OuRAGboros with different mock configurations for benchmarking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
MODE="${1:-baseline}"

echo "🚀 Starting OuRAGboros with configuration: $MODE"

# Stop existing services
echo "📋 Stopping existing services..."
docker compose down

# Set environment variables based on mode
case "$MODE" in
    baseline)
        echo "📊 Running baseline (no mocks)"
        export USE_MOCK_EMBEDDINGS=false
        export USE_MOCK_LLM=false
        ;;
    mock-embeddings)
        echo "🔧 Running with mock embeddings"
        export USE_MOCK_EMBEDDINGS=true
        export USE_MOCK_LLM=false
        ;;
    mock-llm)
        echo "🤖 Running with mock LLM"
        export USE_MOCK_EMBEDDINGS=false
        export USE_MOCK_LLM=true
        ;;
    mock-both)
        echo "⚡ Running with both mocks"
        export USE_MOCK_EMBEDDINGS=true
        export USE_MOCK_LLM=true
        ;;
    *)
        echo "Usage: $0 [baseline|mock-embeddings|mock-llm|mock-both]"
        exit 1
        ;;
esac

# Start services with the configured environment
echo "🐳 Starting Docker services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker compose ps | grep -q "ouragboros.*running"; then
    echo "✅ Services are running with configuration: $MODE"
    echo ""
    echo "📝 Environment:"
    echo "  USE_MOCK_EMBEDDINGS=$USE_MOCK_EMBEDDINGS"
    echo "  USE_MOCK_LLM=$USE_MOCK_LLM"
    echo ""
    echo "🔍 You can now run benchmarks:"
    echo "  python3 scripts/test-api-benchmark.py"
    echo "  ./scripts/test-concurrency-systematic.sh"
else
    echo "❌ Failed to start services"
    exit 1
fi