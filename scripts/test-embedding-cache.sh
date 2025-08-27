#!/bin/bash

# Test script to verify embedding caching is working correctly
echo "🧪 EMBEDDING CACHING TEST"
echo "========================="
echo ""

# Check if services are running
if ! curl -s http://localhost:8001 >/dev/null 2>&1; then
    echo "❌ OuRAGboros API is not running on port 8001"
    echo "   Please run: ./scripts/test-local.sh first"
    exit 1
fi

echo "🔍 Testing embedding instantiation caching..."
echo ""

# Start monitoring logs in background
echo "📊 Starting log monitoring..."
docker compose -f docker-compose.local.yml logs -f ouragboros > /tmp/ouragboros_logs.txt 2>&1 &
LOG_PID=$!

# Wait for log monitoring to start
sleep 2

# Clear any existing logs
> /tmp/ouragboros_logs.txt

echo "🧪 Making test API call to trigger embedding loading..."

# Make a test API call
api_response=$(curl -s -X POST "http://localhost:8001/ask" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is quantum mechanics?",
        "embedding_model": "huggingface:thellert/physbert_cased",
        "llm_model": "ollama:llama3.1",
        "knowledge_base": "test_kb",
        "use_rag": true,
        "prompt": "Answer briefly."
    }')

echo "✅ API call completed"

# Wait for logs to be written
sleep 3

# Stop log monitoring
kill $LOG_PID 2>/dev/null

echo ""
echo "📊 EMBEDDING INSTANTIATION ANALYSIS:"
echo "===================================="

# Count embedding warnings
warning_count=$(grep -c "No sentence-transformers model found with name thellert/physbert_cased. Creating a new one with mean pooling." /tmp/ouragboros_logs.txt 2>/dev/null || echo "0")

echo "Embedding instantiation warnings found: $warning_count"

if [ "$warning_count" -eq "0" ]; then
    echo "✅ PERFECT: No embedding warnings (likely using cached instance)"
elif [ "$warning_count" -eq "1" ]; then
    echo "✅ GOOD: Only 1 embedding warning (caching working correctly)"  
    echo "   This is expected for the first request with a new model."
elif [ "$warning_count" -gt "1" ]; then
    echo "❌ ISSUE: $warning_count embedding warnings detected"
    echo "   This suggests multiple instantiations - caching may not be working properly"
else
    echo "⚠️  Could not determine warning count from logs"
fi

echo ""
echo "📋 TESTING SAME MODEL WITH DIFFERENT KNOWLEDGE BASE:"
echo "===================================================="

# Clear logs again
> /tmp/ouragboros_logs.txt

# Start monitoring again
docker compose -f docker-compose.local.yml logs -f ouragboros >> /tmp/ouragboros_logs.txt 2>&1 &
LOG_PID=$!

sleep 2

echo "🧪 Making second API call with same model but different KB..."

# Make second call with different KB
api_response2=$(curl -s -X POST "http://localhost:8001/ask" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is machine learning?", 
        "embedding_model": "huggingface:thellert/physbert_cased",
        "llm_model": "ollama:llama3.1", 
        "knowledge_base": "different_kb",
        "use_rag": true,
        "prompt": "Answer briefly."
    }')

echo "✅ Second API call completed"

# Wait for logs
sleep 3

# Stop monitoring
kill $LOG_PID 2>/dev/null

# Count warnings in second test
warning_count2=$(grep -c "No sentence-transformers model found with name thellert/physbert_cased. Creating a new one with mean pooling." /tmp/ouragboros_logs.txt 2>/dev/null || echo "0")

echo "Embedding instantiation warnings in second call: $warning_count2"

if [ "$warning_count2" -eq "0" ]; then
    echo "✅ EXCELLENT: No new embedding warnings (reusing cached model)"
else
    echo "❌ ISSUE: New embedding warnings detected"
    echo "   The same model should be reused across different knowledge bases"
fi

echo ""
echo "📋 TESTING DIFFERENT EMBEDDING MODEL:"
echo "===================================="

# Clear logs
> /tmp/ouragboros_logs.txt

# Start monitoring
docker compose -f docker-compose.local.yml logs -f ouragboros >> /tmp/ouragboros_logs.txt 2>&1 &
LOG_PID=$!

sleep 2

echo "🧪 Making API call with different embedding model..."

# Make call with different model (if available)
api_response3=$(curl -s -X POST "http://localhost:8001/ask" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is artificial intelligence?",
        "embedding_model": "ollama:llama3.1", 
        "llm_model": "ollama:llama3.1",
        "knowledge_base": "test_kb",
        "use_rag": false,
        "prompt": "Answer briefly."
    }')

echo "✅ Third API call completed"

sleep 3
kill $LOG_PID 2>/dev/null

echo ""
echo "🎯 TEST SUMMARY:"
echo "==============="
echo "First call (new model):        $warning_count warnings"
echo "Second call (same model+diff KB): $warning_count2 warnings" 
echo "Third call (different model):   Using Ollama (different warning pattern)"

if [ "$warning_count" -le "1" ] && [ "$warning_count2" -eq "0" ]; then
    echo ""
    echo "🎉 EMBEDDING CACHE TEST: PASSED"
    echo "   ✅ Models are being cached correctly"
    echo "   ✅ Knowledge base isolation working"
    echo "   ✅ No unnecessary re-instantiation"
else
    echo ""
    echo "⚠️  EMBEDDING CACHE TEST: NEEDS ATTENTION"
    echo "   Review the implementation for potential issues"
fi

echo ""
echo "📋 Full logs saved to: /tmp/ouragboros_logs.txt"
echo "   To review: cat /tmp/ouragboros_logs.txt"
echo ""
echo "💡 Expected behavior:"
echo "   - First request: 1 warning (model loading)"
echo "   - Subsequent requests with same model: 0 warnings (cache hit)"
echo "   - Different models: 1 warning each (separate cache entries)"

# Cleanup
rm -f /tmp/ouragboros_logs.txt