#!/bin/bash

# Simplified systematic benchmark that works without Python dependencies

set -e

# Configuration
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:8001}"
K8S_BASE_URL="${K8S_BASE_URL:-http://localhost:8501}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-50}"

# Output directory
RESULTS_DIR="benchmark-results-systematic"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/results_$TIMESTAMP.txt"

echo "🚀 OuRAGboros Systematic Benchmark" | tee "$RESULTS_FILE"
echo "====================================" | tee -a "$RESULTS_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$RESULTS_FILE"
echo "Local URL: $LOCAL_BASE_URL" | tee -a "$RESULTS_FILE"
echo "K8s URL: $K8S_BASE_URL" | tee -a "$RESULTS_FILE"
echo "Max Concurrent: $MAX_CONCURRENT" | tee -a "$RESULTS_FILE"
echo "Total Requests: $TOTAL_REQUESTS" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Function to run Apache Bench test
run_ab_test() {
    local name="$1"
    local url="$2"
    local concurrent="$3"
    local requests="$4"
    
    echo "Testing $name with concurrency=$concurrent..." | tee -a "$RESULTS_FILE"
    
    # Create request payload
    cat > /tmp/ab_payload.json <<EOF
{
    "query": "What is quantum mechanics and how does it relate to classical physics?",
    "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "stanford:gpt-4o",
    "prompt": "You are a helpful physics teacher.",
    "use_rag": false
}
EOF
    
    # Run Apache Bench
    ab -n "$requests" -c "$concurrent" \
       -T "application/json" \
       -p /tmp/ab_payload.json \
       "$url/ask" 2>&1 | grep -E "Requests per second:|Time per request:|Failed requests:|Total:" | tee -a "$RESULTS_FILE"
    
    echo "" | tee -a "$RESULTS_FILE"
}

# Function to test streaming endpoint
test_streaming() {
    local name="$1"
    local url="$2"
    
    echo "Testing streaming for $name..." | tee -a "$RESULTS_FILE"
    
    # Measure time to first token
    start_time=$(date +%s%N)
    
    curl -X POST "$url/ask/stream" \
         -H "Content-Type: application/json" \
         -d '{
            "query": "Explain quantum entanglement",
            "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "stanford:gpt-4o",
            "prompt": "You are a helpful physics teacher.",
            "use_rag": false
         }' \
         --silent \
         --no-buffer \
         2>&1 | head -n 1 > /dev/null
    
    end_time=$(date +%s%N)
    ttft=$(( (end_time - start_time) / 1000000 ))
    
    echo "  Time to first token: ${ttft}ms" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
}

# Test configurations
CONFIGS=("baseline" "mock-embeddings" "mock-llm" "mock-both")
CONCURRENCY_LEVELS=(1 5 10 20)

# Main benchmark loop
for config in "${CONFIGS[@]}"; do
    echo "================================================" | tee -a "$RESULTS_FILE"
    echo "Configuration: $config" | tee -a "$RESULTS_FILE"
    echo "================================================" | tee -a "$RESULTS_FILE"
    
    # Start services with the right configuration
    echo "Starting services with $config configuration..." | tee -a "$RESULTS_FILE"
    ./scripts/run-with-mocks.sh "$config"
    
    # Wait for services to stabilize
    sleep 10
    
    # Test LOCAL deployment
    echo "" | tee -a "$RESULTS_FILE"
    echo "--- LOCAL Deployment ---" | tee -a "$RESULTS_FILE"
    
    # Check if local service is running
    if curl -s "$LOCAL_BASE_URL/docs" > /dev/null 2>&1; then
        # Test streaming
        test_streaming "LOCAL" "$LOCAL_BASE_URL"
        
        # Test different concurrency levels
        for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
            run_ab_test "LOCAL" "$LOCAL_BASE_URL" "$concurrency" "$TOTAL_REQUESTS"
        done
    else
        echo "❌ Local service not available" | tee -a "$RESULTS_FILE"
    fi
    
    # Test K8s deployment (if available)
    echo "" | tee -a "$RESULTS_FILE"
    echo "--- K8s Deployment ---" | tee -a "$RESULTS_FILE"
    
    if curl -s "$K8S_BASE_URL/docs" > /dev/null 2>&1; then
        # Test streaming
        test_streaming "K8s" "$K8S_BASE_URL"
        
        # Test different concurrency levels
        for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
            run_ab_test "K8s" "$K8S_BASE_URL" "$concurrency" "$TOTAL_REQUESTS"
        done
    else
        echo "⚠️  K8s service not available (port-forward may not be running)" | tee -a "$RESULTS_FILE"
    fi
    
    echo "" | tee -a "$RESULTS_FILE"
done

# Summary
echo "================================================" | tee -a "$RESULTS_FILE"
echo "📊 BENCHMARK COMPLETE" | tee -a "$RESULTS_FILE"
echo "================================================" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "💡 Next steps:" | tee -a "$RESULTS_FILE"
echo "  1. Review results in $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "  2. Compare Local vs K8s performance" | tee -a "$RESULTS_FILE"
echo "  3. Identify bottlenecks from mock configurations" | tee -a "$RESULTS_FILE"