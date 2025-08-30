#!/bin/bash

# Simplified Streaming API Benchmark
# Focuses on Apache Bench load testing with TTFT sampling

set -e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8001}"
ENDPOINT="${BASE_URL}/ask/stream"
OUTPUT_DIR="benchmark-results-streaming"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${OUTPUT_DIR}/streaming_simple_${TIMESTAMP}.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "${OUTPUT_DIR}"

echo -e "${GREEN}=== Streaming API Benchmark ===${NC}"
echo "Endpoint: ${ENDPOINT}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Create request body
cat > "/tmp/benchmark_request_${TIMESTAMP}.json" <<EOF
{
    "query": "What are the key principles of quantum mechanics?",
    "embedding_model": "huggingface:thellert/physbert_cased",
    "llm_model": "stanford:gpt-4o-mini",
    "prompt": "You are a helpful physics assistant.",
    "use_rag": true,
    "use_qdrant": true,
    "use_opensearch": false,
    "knowledge_base": "default",
    "max_documents": 3,
    "score_threshold": 0.0
}
EOF

echo -e "${YELLOW}Testing connectivity...${NC}"
test_response=$(curl -s -X POST -H "Content-Type: application/json" -p "/tmp/benchmark_request_${TIMESTAMP}.json" "${ENDPOINT}" | head -3)
if [[ -n "$test_response" ]]; then
    echo -e "${GREEN}✓ Endpoint responding${NC}"
else
    echo -e "${RED}✗ Endpoint not responding${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting benchmarks...${NC}"

# Test different concurrency levels
results="["
first=true

for concurrency in 1 5 10 20; do
    echo -e "${YELLOW}=== Concurrency: ${concurrency} ===${NC}"
    
    requests=20
    if [[ $concurrency -gt 10 ]]; then
        requests=50
    fi
    
    echo "Running ab: $requests requests, $concurrency concurrent"
    
    # Run ab benchmark
    ab_output="/tmp/ab_${concurrency}_${TIMESTAMP}.txt"
    ab -n $requests \
       -c $concurrency \
       -T "application/json" \
       -p "/tmp/benchmark_request_${TIMESTAMP}.json" \
       "${ENDPOINT}" > "$ab_output" 2>&1
    
    # Extract key metrics
    total_time=$(grep "Time taken for tests:" "$ab_output" | awk '{print $5}' || echo "0")
    complete=$(grep "Complete requests:" "$ab_output" | awk '{print $3}' || echo "0")
    failed=$(grep "Failed requests:" "$ab_output" | awk '{print $3}' || echo "0")
    rps=$(grep "Requests per second:" "$ab_output" | awk '{print $4}' || echo "0")
    mean_time=$(grep "Time per request:" "$ab_output" | grep "(mean)" | awk '{print $4}' || echo "0")
    
    echo "  Complete: $complete/$requests, Failed: $failed, RPS: $rps, Mean: ${mean_time}ms"
    
    # Sample TTFT (Time To First Token)
    echo "  Sampling TTFT..."
    ttft_start=$(date +%s.%N)
    ttft_response=$(curl -s -N -X POST \
        -H "Content-Type: application/json" \
        -p "/tmp/benchmark_request_${TIMESTAMP}.json" \
        "${ENDPOINT}" | head -10 | grep 'data:.*token' | head -1)
    
    if [[ -n "$ttft_response" ]]; then
        ttft_time=$(date +%s.%N)
        ttft=$(echo "$ttft_time - $ttft_start" | bc -l)
        echo "  TTFT: ${ttft}s"
    else
        ttft="null"
        echo "  TTFT: No tokens received"
    fi
    
    # Build result JSON
    result="{\"concurrency\": $concurrency, \"requests\": $requests, \"complete\": $complete, \"failed\": $failed, \"rps\": $rps, \"mean_ms\": $mean_time, \"ttft\": $ttft, \"total_time\": $total_time}"
    
    if [[ "$first" == false ]]; then
        results="$results,"
    fi
    results="$results$result"
    first=false
    
    echo ""
    sleep 2
done

results="$results]"

# Save results
echo "$results" > "$RESULT_FILE"

# Summary
echo -e "${GREEN}=== Summary ===${NC}"
echo "Results saved to: $RESULT_FILE"
echo ""
echo "Concurrency | Complete | Failed | RPS  | Mean(ms) | TTFT(s)"
echo "------------|----------|--------|------|----------|--------"

# Parse and display results
echo "$results" | jq -r '.[] | "\(.concurrency)\t\(.complete)\t\(.failed)\t\(.rps)\t\(.mean_ms)\t\(.ttft)"' | column -t

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"

# Clean up
rm -f "/tmp/benchmark_request_${TIMESTAMP}.json"
rm -f /tmp/ab_*_${TIMESTAMP}.txt