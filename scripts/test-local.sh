#!/bin/bash

# Local development testing script for OuRAGboros
echo "🧪 OURAGBOROS LOCAL DEVELOPMENT TESTING"
echo "========================================"
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please create it first."
    echo "   See README or docker-compose.local.yml for required variables."
    exit 1
fi

echo "📋 STARTING LOCAL SERVICES"
echo "=========================="
echo ""

# Use the local Docker Compose file
echo "🚀 Starting services with docker-compose.local.yml..."
docker compose -f docker-compose.local.yml up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo ""
echo "📊 SERVICE STATUS:"
echo "=================="
docker compose -f docker-compose.local.yml ps

echo ""
echo "🔍 HEALTH CHECKS:"
echo "================="

# Check OpenSearch
echo -n "OpenSearch (port 9200): "
if curl -s http://localhost:9200 >/dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "❌ Not ready"
fi

# Check Ollama
echo -n "Ollama (port 11434): "
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "❌ Not ready"
fi

# Check OuRAGboros Streamlit UI
echo -n "Streamlit UI (port 8501): "
if curl -s http://localhost:8501 >/dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "❌ Not ready"
fi

# Check OuRAGboros FastAPI
echo -n "FastAPI REST (port 8001): "
if curl -s http://localhost:8001 >/dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "❌ Not ready"
fi

echo ""
echo "🧪 BASIC API TESTS:"
echo "=================="

# Test basic API endpoint
echo "Testing /ask endpoint (without RAG)..."
api_response=$(curl -s -X POST "http://localhost:8001/ask" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is 2+2?",
        "embedding_model": "huggingface:thellert/physbert_cased",
        "llm_model": "ollama:llama3.1",
        "knowledge_base": "default",
        "use_rag": false,
        "prompt": "Answer briefly and clearly."
    }' 2>/dev/null)

if [ $? -eq 0 ] && [ ! -z "$api_response" ]; then
    echo "✅ API test passed"
    echo "   Response preview: $(echo "$api_response" | jq -r '.answer' 2>/dev/null | head -c 50)..."
else
    echo "❌ API test failed"
    echo "   Check the logs: docker compose -f docker-compose.local.yml logs ouragboros"
fi

echo ""
echo "📋 ACCESS INFORMATION:"
echo "====================="
echo "• Streamlit UI:     http://localhost:8501"
echo "• FastAPI REST:     http://localhost:8001"
echo "• API Docs:         http://localhost:8001/docs"  
echo "• OpenSearch:       http://localhost:9200"
echo "• Ollama:           http://localhost:11434"
echo ""

echo "📋 USEFUL COMMANDS:"
echo "=================="
echo "• View logs:        docker compose -f docker-compose.local.yml logs -f ouragboros"
echo "• Stop services:    docker compose -f docker-compose.local.yml down"
echo "• Restart:          docker compose -f docker-compose.local.yml restart ouragboros"
echo "• Clean rebuild:    docker compose -f docker-compose.local.yml build --no-cache ouragboros"
echo ""

echo "🎯 TESTING CHECKLIST:"
echo "===================="
echo "1. ✅ Services started"
echo "2. [ ] Open Streamlit UI and test knowledge base creation"
echo "3. [ ] Upload a document and test RAG queries"
echo "4. [ ] Switch embedding models and verify caching works"
echo "5. [ ] Test concurrent API requests"
echo "6. [ ] Monitor logs for embedding instantiation warnings"
echo ""

echo "To run embedding instantiation test:"
echo "./scripts/test-embedding-cache.sh"
echo ""
echo "To run concurrency tests:"
echo "./scripts/test-concurrent.sh"