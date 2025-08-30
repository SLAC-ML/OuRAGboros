#!/bin/bash

# Streaming API Concurrency Benchmark using Apache Bench (ab)
# Measures Time-To-First-Token (TTFT) for streaming responses
# Uses ab for consistent load testing like our previous benchmarks

set -e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8001}"
ENDPOINT="${BASE_URL}/ask/stream"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-100}"
OUTPUT_DIR="benchmark-results-streaming"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${OUTPUT_DIR}/streaming_benchmark_${TIMESTAMP}.json"
LOG_FILE="${OUTPUT_DIR}/streaming_benchmark_${TIMESTAMP}.log"
RAW_DIR="${OUTPUT_DIR}/raw_${TIMESTAMP}"
VECTOR_STORAGE="${VECTOR_STORAGE:-qdrant}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-default}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create output directories
mkdir -p "${OUTPUT_DIR}" "${RAW_DIR}"

# Check if ab is installed
if ! command -v ab &> /dev/null; then
    echo -e "${RED}Apache Bench (ab) is not installed. Installing...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install httpd || brew install apache2
    else
        sudo apt-get update && sudo apt-get install -y apache2-utils
    fi
fi

# Test queries for variety
QUERIES=(
    "What are the key principles of quantum mechanics?"
    "Explain the concept of entanglement in physics"
    "What is the standard model of particle physics?"
    "How does general relativity differ from special relativity?"
    "What are the applications of superconductivity?"
)

# Function to create request body file for ab
create_request_body() {
    local query="${1:-What are the key principles of quantum mechanics?}"
    local temp_file="/tmp/ab_request_${TIMESTAMP}.json"
    
    cat > "$temp_file" <<EOF
{
    "query": "${query}",
    "embedding_model": "huggingface:thellert/physbert_cased",
    "llm_model": "stanford:gpt-4o-mini",
    "prompt": "You are a helpful physics assistant.",
    "use_rag": true,
    "use_qdrant": $([ "$VECTOR_STORAGE" = "qdrant" ] && echo "true" || echo "false"),
    "use_opensearch": $([ "$VECTOR_STORAGE" = "opensearch" ] && echo "true" || echo "false"),
    "knowledge_base": "${KNOWLEDGE_BASE}",
    "max_documents": 3,
    "score_threshold": 0.0
}
EOF
    echo "$temp_file"
}

# Function to measure TTFT with curl alongside ab
measure_ttft_sample() {
    local query="$1"
    local sample_id="$2"
    local start_time=$(date +%s.%N)
    
    # Create request body
    local request_body=$(cat <<EOF
{
    "query": "${query}",
    "embedding_model": "huggingface:thellert/physbert_cased",
    "llm_model": "stanford:gpt-4o-mini",
    "prompt": "You are a helpful physics assistant.",
    "use_rag": true,
    "use_qdrant": $([ "$VECTOR_STORAGE" = "qdrant" ] && echo "true" || echo "false"),
    "use_opensearch": $([ "$VECTOR_STORAGE" = "opensearch" ] && echo "true" || echo "false"),
    "knowledge_base": "${KNOWLEDGE_BASE}",
    "max_documents": 3,
    "score_threshold": 0.0
}
EOF
)
    
    # Make streaming request and capture TTFT
    local ttft=""
    local first_token_received=false
    
    {
        curl -s -N -X POST \
            -H "Content-Type: application/json" \
            -d "${request_body}" \
            "${ENDPOINT}" 2>/dev/null | while IFS= read -r line; do
            
            if [[ "$line" == data:* && "$first_token_received" == false ]]; then
                json_data="${line#data: }"
                if [[ -n "$json_data" && "$json_data" != " " ]]; then
                    event_type=$(echo "$json_data" | jq -r '.type' 2>/dev/null || echo "")
                    if [[ "$event_type" == "token" ]]; then
                        ttft=$(echo "$(date +%s.%N) - $start_time" | bc)
                        echo "TTFT:${ttft}"
                        first_token_received=true
                        break
                    fi
                fi
            fi
        done
    } 2>/dev/null
}

# Function to run ab benchmark and collect TTFT samples
run_ab_benchmark() {
    local concurrent="$1"
    local requests="$2"
    
    echo -e "${YELLOW}Running ab with concurrency ${concurrent}, ${requests} requests...${NC}"
    
    # Create request body file
    local request_file=$(create_request_body "${QUERIES[0]}")
    
    # Run ab benchmark
    local ab_output="${RAW_DIR}/ab_c${concurrent}_n${requests}.txt"
    
    ab -n "$requests" \
       -c "$concurrent" \
       -T "application/json" \
       -p "$request_file" \
       -g "${RAW_DIR}/ab_c${concurrent}_n${requests}.tsv" \
       -e "${RAW_DIR}/ab_c${concurrent}_n${requests}.csv" \
       "${ENDPOINT}" > "$ab_output" 2>&1
    
    # Extract ab metrics
    local total_time=$(grep "Time taken for tests:" "$ab_output" | awk '{print $5}')
    local rps=$(grep "Requests per second:" "$ab_output" | awk '{print $4}')
    local mean_time=$(grep "Time per request:" "$ab_output" | grep "(mean)" | awk '{print $4}')
    local failed=$(grep "Failed requests:" "$ab_output" | awk '{print $3}')
    local complete=$((requests - failed))
    
    # Sample TTFT measurements (run a few streaming requests to measure TTFT)
    echo -e "  Sampling TTFT measurements..."
    local ttft_samples=()
    local sample_count=5  # Take 5 samples
    
    for ((i=0; i<sample_count; i++)); do
        query="${QUERIES[$((i % ${#QUERIES[@]}))]}"
        ttft_result=$(measure_ttft_sample "$query" "${concurrent}_${i}")
        if [[ "$ttft_result" == TTFT:* ]]; then
            ttft_value="${ttft_result#TTFT:}"
            ttft_samples+=("$ttft_value")
        fi
    done
    
    # Calculate TTFT statistics
    local avg_ttft=0
    local min_ttft=999999
    local max_ttft=0
    local ttft_count=${#ttft_samples[@]}
    
    if [[ $ttft_count -gt 0 ]]; then
        local total_ttft=0
        for ttft in "${ttft_samples[@]}"; do
            total_ttft=$(echo "$total_ttft + $ttft" | bc)
            if (( $(echo "$ttft < $min_ttft" | bc -l) )); then
                min_ttft=$ttft
            fi
            if (( $(echo "$ttft > $max_ttft" | bc -l) )); then
                max_ttft=$ttft
            fi
        done
        avg_ttft=$(echo "scale=3; $total_ttft / $ttft_count" | bc)
    fi
    
    # Clean up request file
    rm -f "$request_file"
    
    # Return results as JSON
    echo "{
        \"concurrency\": ${concurrent},
        \"total_requests\": ${requests},
        \"successful\": ${complete},
        \"failed\": ${failed},
        \"requests_per_second\": ${rps},
        \"mean_response_time_ms\": ${mean_time},
        \"total_test_time_s\": ${total_time},
        \"ttft_samples\": ${ttft_count},
        \"avg_ttft_s\": ${avg_ttft},
        \"min_ttft_s\": ${min_ttft},
        \"max_ttft_s\": ${max_ttft}
    }"
}

# Main benchmark execution
echo -e "${GREEN}=== OuRAGboros Streaming API Benchmark (Apache Bench) ===${NC}"
echo -e "Endpoint: ${ENDPOINT}"
echo -e "Vector Storage: ${VECTOR_STORAGE}"
echo -e "Knowledge Base: ${KNOWLEDGE_BASE}"
echo -e "Max Concurrency: ${MAX_CONCURRENT}"
echo -e "Total Requests: ${TOTAL_REQUESTS}"
echo -e "Output: ${RESULT_FILE}"
echo ""

# Test connectivity first
echo -e "${YELLOW}Testing endpoint connectivity...${NC}"
test_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2", "llm_model": "stanford:gpt-4o-mini", "prompt": "You are a helpful assistant."}' \
    "${ENDPOINT}" 2>&1 | head -5)

if [[ -z "$test_response" ]]; then
    echo -e "${RED}Error: Cannot connect to ${ENDPOINT}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Endpoint is reachable${NC}"
echo ""

# Warm-up request
echo -e "${YELLOW}Sending warm-up request...${NC}"
curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "warm up", "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2", "llm_model": "stanford:gpt-4o-mini", "prompt": "You are a helpful assistant."}' \
    "${ENDPOINT}" > /dev/null 2>&1
echo -e "${GREEN}✓ Warm-up complete${NC}"
echo ""

# Run benchmarks at different concurrency levels
echo -e "${GREEN}Starting benchmark tests...${NC}"
echo ""

benchmark_results="["
first_result=true

# Test different concurrency levels
for concurrent in 1 5 10 20 30 40 50; do
    if [[ $concurrent -gt $MAX_CONCURRENT ]]; then
        break
    fi
    
    echo -e "${GREEN}=== Concurrency Level: ${concurrent} ===${NC}"
    
    # Calculate requests for this level
    requests_for_level=$TOTAL_REQUESTS
    if [[ $concurrent -eq 1 ]]; then
        requests_for_level=20  # Fewer requests for single-threaded
    fi
    
    # Run the benchmark
    result=$(run_ab_benchmark "$concurrent" "$requests_for_level")
    
    # Add to results array
    if [[ "$first_result" == false ]]; then
        benchmark_results="${benchmark_results},"
    fi
    benchmark_results="${benchmark_results}${result}"
    first_result=false
    
    # Display results
    echo "$result" | jq '.'
    echo ""
    
    # Small delay between tests
    sleep 2
done

benchmark_results="${benchmark_results}]"

# Save final results
echo "$benchmark_results" | jq '.' > "$RESULT_FILE"

# Generate summary report
echo -e "${GREEN}=== Benchmark Summary ===${NC}"
echo ""
echo "Concurrency | Requests | Success | Failed | RPS    | Mean RT (ms) | Avg TTFT (s)"
echo "------------|----------|---------|--------|--------|--------------|-------------"
echo "$benchmark_results" | jq -r '.[] | "\(.concurrency | tostring | .[0:11]) | \(.total_requests | tostring | .[0:8]) | \(.successful | tostring | .[0:7]) | \(.failed | tostring | .[0:6]) | \(.requests_per_second | tostring | .[0:6]) | \(.mean_response_time_ms | tostring | .[0:12]) | \(.avg_ttft_s)"'
echo ""

# Calculate overall statistics
total_success=$(echo "$benchmark_results" | jq '[.[].successful] | add')
total_requests=$(echo "$benchmark_results" | jq '[.[].total_requests] | add')
overall_avg_ttft=$(echo "$benchmark_results" | jq '[.[].avg_ttft_s] | add / length')
max_rps=$(echo "$benchmark_results" | jq '[.[].requests_per_second] | max')

echo -e "${GREEN}=== Overall Results ===${NC}"
echo "Total Requests: ${total_requests}"
echo "Total Successful: ${total_success}"
echo "Success Rate: $(echo "scale=2; $total_success * 100 / $total_requests" | bc)%"
echo "Average TTFT: ${overall_avg_ttft}s"
echo "Peak RPS: ${max_rps}"
echo ""
echo "Results saved to: ${RESULT_FILE}"
echo "Raw ab outputs in: ${RAW_DIR}/"

# Performance comparison
echo ""
echo -e "${GREEN}=== Performance Analysis ===${NC}"
echo "Time-To-First-Token (TTFT) shows when users start seeing response"
echo "This is critical for perceived performance in streaming UIs"
echo ""

# Find optimal concurrency level
optimal=$(echo "$benchmark_results" | jq 'max_by(.requests_per_second) | "Optimal concurrency: \(.concurrency) with \(.requests_per_second) RPS"')
echo "$optimal"

# Compare with non-streaming if data exists
if ls benchmark-results/benchmark_*.json 2>/dev/null | head -1 > /dev/null; then
    echo ""
    echo -e "${GREEN}=== Streaming vs Non-Streaming Comparison ===${NC}"
    latest_nonstreaming=$(ls -t benchmark-results/benchmark_*.json 2>/dev/null | head -1)
    if [[ -f "$latest_nonstreaming" ]]; then
        echo "Comparing with: $(basename $latest_nonstreaming)"
        old_avg=$(jq '[.[].avg_response_time] | add / length' "$latest_nonstreaming" 2>/dev/null || echo "N/A")
        if [[ "$old_avg" != "N/A" ]]; then
            echo "Non-streaming avg response time: ${old_avg}s (full response)"
            echo "Streaming avg TTFT: ${overall_avg_ttft}s (first token)"
            improvement=$(echo "scale=1; ($old_avg - $overall_avg_ttft) / $old_avg * 100" | bc 2>/dev/null || echo "N/A")
            echo -e "${GREEN}Perceived performance improvement: ${improvement}%${NC}"
            echo ""
            echo "Users see first response ~${improvement}% faster with streaming!"
        fi
    fi
fi