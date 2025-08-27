#!/bin/bash

# Concurrency test script for OuRAGboros API
echo "🧪 CONCURRENCY & THREAD-SAFETY TEST"
echo "==================================="
echo ""

# Check if services are running
if ! curl -s http://localhost:8001 >/dev/null 2>&1; then
    echo "❌ OuRAGboros API is not running on port 8001"
    echo "   Please run: ./scripts/test-local.sh first"
    exit 1
fi

echo "🔍 Testing concurrent requests with different embedding models and knowledge bases..."
echo ""

# Start log monitoring
echo "📊 Starting log monitoring..."
docker compose -f docker-compose.local.yml logs -f ouragboros > /tmp/concurrent_logs.txt 2>&1 &
LOG_PID=$!

# Wait for monitoring to start
sleep 2

# Clear existing logs
> /tmp/concurrent_logs.txt

echo "🧪 Launching 5 concurrent API requests..."

# Launch 5 concurrent requests with different combinations
(
    curl -s -X POST "http://localhost:8001/ask" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "What is quantum physics?",
            "embedding_model": "huggingface:thellert/physbert_cased",
            "llm_model": "ollama:llama3.1",
            "knowledge_base": "physics_kb",
            "use_rag": false,
            "prompt": "Answer briefly."
        }' > /tmp/response1.json
    echo "Request 1 (physbert + physics_kb) completed"
) &

(
    curl -s -X POST "http://localhost:8001/ask" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "What is machine learning?", 
            "embedding_model": "huggingface:thellert/physbert_cased",
            "llm_model": "ollama:llama3.1",
            "knowledge_base": "ml_kb", 
            "use_rag": false,
            "prompt": "Answer briefly."
        }' > /tmp/response2.json
    echo "Request 2 (physbert + ml_kb) completed"
) &

(
    curl -s -X POST "http://localhost:8001/ask" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "What is chemistry?",
            "embedding_model": "ollama:llama3.1",
            "llm_model": "ollama:llama3.1", 
            "knowledge_base": "chem_kb",
            "use_rag": false,
            "prompt": "Answer briefly."
        }' > /tmp/response3.json
    echo "Request 3 (ollama + chem_kb) completed"
) &

(
    curl -s -X POST "http://localhost:8001/ask" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "What is biology?",
            "embedding_model": "huggingface:thellert/physbert_cased",
            "llm_model": "ollama:llama3.1",
            "knowledge_base": "bio_kb",
            "use_rag": false, 
            "prompt": "Answer briefly."
        }' > /tmp/response4.json
    echo "Request 4 (physbert + bio_kb) completed"
) &

(
    curl -s -X POST "http://localhost:8001/ask" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "What is mathematics?",
            "embedding_model": "ollama:llama3.1", 
            "llm_model": "ollama:llama3.1",
            "knowledge_base": "math_kb",
            "use_rag": false,
            "prompt": "Answer briefly."
        }' > /tmp/response5.json
    echo "Request 5 (ollama + math_kb) completed"
) &

# Wait for all background jobs to complete
echo "⏳ Waiting for all requests to complete..."
wait

echo "✅ All concurrent requests completed!"

# Wait for logs to be written
sleep 5

# Stop log monitoring
kill $LOG_PID 2>/dev/null

echo ""
echo "📊 CONCURRENCY ANALYSIS:"
echo "========================"

# Analyze embedding instantiation patterns
physbert_warnings=$(grep -c "No sentence-transformers model found with name thellert/physbert_cased. Creating a new one with mean pooling." /tmp/concurrent_logs.txt 2>/dev/null || echo "0")

echo "HuggingFace physbert embedding warnings: $physbert_warnings"

# Check for any threading errors or exceptions
error_count=$(grep -i -E "error|exception|failed" /tmp/concurrent_logs.txt | grep -v "INFO" | wc -l)
echo "Error/Exception count in logs: $error_count"

# Check for race conditions (multiple simultaneous instantiations)
if [ "$physbert_warnings" -eq "0" ]; then
    echo "✅ EXCELLENT: No embedding warnings (using pre-cached models)"
elif [ "$physbert_warnings" -eq "1" ]; then
    echo "✅ GOOD: Only 1 embedding warning (proper caching)"
    echo "   This indicates the first request loaded the model, others reused it."
elif [ "$physbert_warnings" -gt "3" ]; then
    echo "❌ POTENTIAL RACE CONDITION: $physbert_warnings warnings"
    echo "   Multiple threads may be instantiating the same model simultaneously"
else
    echo "⚠️  MODERATE: $physbert_warnings warnings detected"
    echo "   May indicate some inefficiency but not necessarily a race condition"
fi

echo ""
echo "📋 RESPONSE ANALYSIS:"
echo "===================="

# Check if all responses were successful
successful_responses=0
for i in {1..5}; do
    if [ -f "/tmp/response$i.json" ] && jq -e '.answer' /tmp/response$i.json >/dev/null 2>&1; then
        successful_responses=$((successful_responses + 1))
        echo "Response $i: ✅ Success"
    else
        echo "Response $i: ❌ Failed or malformed"
    fi
done

echo ""
echo "🎯 CONCURRENCY TEST SUMMARY:"
echo "============================"
echo "Total requests sent: 5"
echo "Successful responses: $successful_responses/5"
echo "HuggingFace embedding warnings: $physbert_warnings"
echo "Errors in logs: $error_count"

if [ "$successful_responses" -eq "5" ] && [ "$error_count" -eq "0" ] && [ "$physbert_warnings" -le "1" ]; then
    echo ""
    echo "🎉 CONCURRENCY TEST: PASSED"
    echo "   ✅ All requests completed successfully"
    echo "   ✅ No race conditions detected"  
    echo "   ✅ Proper thread-safe caching"
elif [ "$successful_responses" -ge "4" ] && [ "$error_count" -eq "0" ]; then
    echo ""
    echo "✅ CONCURRENCY TEST: MOSTLY PASSED"
    echo "   Minor issues detected but overall functioning correctly"
else
    echo ""
    echo "⚠️  CONCURRENCY TEST: NEEDS ATTENTION"
    echo "   Review logs for threading or caching issues"
fi

echo ""
echo "📋 DETAILED ANALYSIS:"
echo "===================="
echo "• Full logs: /tmp/concurrent_logs.txt"
echo "• Response files: /tmp/response[1-5].json"
echo ""
echo "View logs: cat /tmp/concurrent_logs.txt"
echo "View specific response: jq . /tmp/response1.json"

echo ""
echo "💡 Expected behavior for thread-safe caching:"
echo "   - Same embedding model reused across requests (max 1 instantiation per model)"
echo "   - Different knowledge bases properly isolated"
echo "   - No race conditions or threading errors"
echo "   - All requests complete successfully"

# Cleanup response files
rm -f /tmp/response*.json
# Keep logs for manual inspection