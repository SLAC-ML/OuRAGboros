#!/bin/bash

# Systematic Concurrency Benchmark: Local vs K8s with Bottleneck Isolation
# This script runs identical benchmarks against local and K8s deployments
# with different mock configurations to isolate specific bottlenecks

set -e

# Configuration
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:8001}"
K8S_BASE_URL="${K8S_BASE_URL:-http://localhost:8501}"  # Port-forwarded K8s service
ENDPOINT_PATH="/ask/stream"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-50}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"

# Output configuration
OUTPUT_DIR="benchmark-results-systematic"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${OUTPUT_DIR}/systematic_benchmark_${TIMESTAMP}.log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "🚀 Systematic Concurrency Benchmark - Local vs K8s" | tee "$MAIN_LOG"
echo "=================================================" | tee -a "$MAIN_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$MAIN_LOG"
echo "Local URL: $LOCAL_BASE_URL" | tee -a "$MAIN_LOG"
echo "K8s URL: $K8S_BASE_URL" | tee -a "$MAIN_LOG"
echo "Max Concurrency: $MAX_CONCURRENT" | tee -a "$MAIN_LOG"
echo "Total Requests: $TOTAL_REQUESTS" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Test configurations to isolate different bottlenecks
declare -a TEST_CONFIGS=(
    "baseline:false:false:No mocking - full pipeline"
    "mock-embeddings:true:false:Mock embeddings only"
    "mock-llm:false:true:Mock LLM only" 
    "mock-both:true:true:Mock embeddings + LLM"
)

# Concurrency levels to test
CONCURRENCY_LEVELS=(1 2 5 10 15 20)

# Function to make a test request to check connectivity
test_connectivity() {
    local base_url="$1"
    local deployment_name="$2"
    
    echo -e "${YELLOW}Testing connectivity to ${deployment_name}...${NC}" | tee -a "$MAIN_LOG"
    
    local test_payload='{"query": "connectivity test", "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2", "llm_model": "stanford:gpt-4o-mini", "prompt": "You are a helpful assistant.", "use_rag": false}'
    
    local response=$(curl -s -m 10 -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "${base_url}${ENDPOINT_PATH}" 2>&1 | head -5)
    
    if [[ -z "$response" || "$response" == *"Connection refused"* ]]; then
        echo -e "${RED}✗ ${deployment_name} is not reachable at ${base_url}${NC}" | tee -a "$MAIN_LOG"
        return 1
    else
        echo -e "${GREEN}✓ ${deployment_name} is reachable${NC}" | tee -a "$MAIN_LOG"
        return 0
    fi
}

# Function to measure TTFT for a single request
measure_single_ttft() {
    local base_url="$1"
    local use_mock_embeddings="$2"
    local use_mock_llm="$3"
    local query="$4"
    local sample_id="$5"
    
    local start_time=$(date +%s.%N)
    
    # Enhanced request body with mock configurations
    local request_body=$(cat <<EOF
{
    "query": "${query}",
    "embedding_model": "huggingface:thellert/physbert_cased",
    "llm_model": "stanford:gpt-4o-mini",
    "prompt": "You are a helpful physics assistant.",
    "use_rag": true,
    "use_qdrant": true,
    "knowledge_base": "default",
    "max_documents": 3,
    "score_threshold": 0.0
}
EOF
)
    
    # Set environment variables for the request (if targeting local)
    local env_vars=""
    if [[ "$base_url" == *"localhost:8001"* ]]; then
        env_vars="USE_MOCK_EMBEDDINGS=$use_mock_embeddings USE_MOCK_LLM=$use_mock_llm"
    fi
    
    # Make streaming request and capture TTFT
    local ttft=""
    local first_token_received=false
    local total_time=""
    local response_complete=false
    
    {
        if [[ -n "$env_vars" ]]; then
            env $env_vars curl -s -N -m 30 -X POST \
                -H "Content-Type: application/json" \
                -d "${request_body}" \
                "${base_url}${ENDPOINT_PATH}" 2>/dev/null
        else
            curl -s -N -m 30 -X POST \
                -H "Content-Type: application/json" \
                -d "${request_body}" \
                "${base_url}${ENDPOINT_PATH}" 2>/dev/null
        fi
    } | while IFS= read -r line; do
        if [[ "$line" == data:* ]]; then
            json_data="${line#data: }"
            if [[ -n "$json_data" && "$json_data" != " " ]]; then
                event_type=$(echo "$json_data" | jq -r '.type' 2>/dev/null || echo "")
                
                if [[ "$event_type" == "token" && "$first_token_received" == false ]]; then
                    ttft=$(echo "$(date +%s.%N) - $start_time" | bc)
                    echo "TTFT:${ttft}"
                    first_token_received=true
                elif [[ "$event_type" == "complete" || "$event_type" == "end" ]]; then
                    total_time=$(echo "$(date +%s.%N) - $start_time" | bc)
                    echo "TOTAL:${total_time}"
                    break
                fi
            fi
        fi
    done 2>/dev/null
}

# Function to run concurrent benchmark using Apache Bench
run_concurrent_benchmark() {
    local base_url="$1"
    local deployment_name="$2"
    local config_name="$3"
    local use_mock_embeddings="$4"
    local use_mock_llm="$5"
    local concurrent="$6"
    local requests="$7"
    
    echo -e "${BLUE}  Running concurrency ${concurrent} (${requests} requests)...${NC}" | tee -a "$MAIN_LOG"
    
    # Create temporary request body file
    local temp_request="/tmp/bench_request_${TIMESTAMP}_${concurrent}.json"
    cat > "$temp_request" <<EOF
{
    "query": "What are the key principles of quantum mechanics?",
    "embedding_model": "huggingface:thellert/physbert_cased",
    "llm_model": "stanford:gpt-4o-mini",
    "prompt": "You are a helpful physics assistant.",
    "use_rag": true,
    "use_qdrant": true,
    "knowledge_base": "default",
    "max_documents": 3,
    "score_threshold": 0.0
}
EOF
    
    # Run Apache Bench
    local ab_output="/tmp/ab_${deployment_name}_${config_name}_c${concurrent}.txt"
    local ab_success=false
    
    if command -v ab &> /dev/null; then
        timeout 300 ab -n "$requests" \
           -c "$concurrent" \
           -T "application/json" \
           -p "$temp_request" \
           "${base_url}${ENDPOINT_PATH}" > "$ab_output" 2>&1 && ab_success=true
    fi
    
    local total_time="0"
    local rps="0"
    local mean_time="0"
    local failed="0"
    local successful="0"
    
    if [[ "$ab_success" == true ]]; then
        total_time=$(grep "Time taken for tests:" "$ab_output" | awk '{print $5}' | head -1)
        rps=$(grep "Requests per second:" "$ab_output" | awk '{print $4}' | head -1)
        mean_time=$(grep "Time per request:" "$ab_output" | grep "(mean)" | awk '{print $4}' | head -1)
        failed=$(grep "Failed requests:" "$ab_output" | awk '{print $3}' | head -1)
        successful=$((requests - ${failed:-0}))
    else
        echo "    ⚠️  Apache Bench failed, using manual measurement" | tee -a "$MAIN_LOG"
        failed=$requests
        successful=0
    fi
    
    # Sample TTFT measurements
    local ttft_samples=()
    local sample_count=3  # Take fewer samples to speed up benchmarks
    
    for ((i=0; i<sample_count; i++)); do
        query="What are quantum mechanics principles? (Sample $i)"
        ttft_result=$(measure_single_ttft "$base_url" "$use_mock_embeddings" "$use_mock_llm" "$query" "${concurrent}_${i}" 2>/dev/null)
        if [[ "$ttft_result" == *"TTFT:"* ]]; then
            ttft_value=$(echo "$ttft_result" | grep "TTFT:" | cut -d: -f2)
            if [[ -n "$ttft_value" && "$ttft_value" != "0" ]]; then
                ttft_samples+=("$ttft_value")
            fi
        fi
    done
    
    # Calculate TTFT statistics
    local avg_ttft="0"
    local min_ttft="999999"
    local max_ttft="0"
    local ttft_count=${#ttft_samples[@]}
    
    if [[ $ttft_count -gt 0 ]]; then
        local total_ttft=0
        for ttft in "${ttft_samples[@]}"; do
            total_ttft=$(echo "$total_ttft + $ttft" | bc 2>/dev/null || echo "$total_ttft")
            if (( $(echo "$ttft < $min_ttft" | bc -l 2>/dev/null) )); then
                min_ttft=$ttft
            fi
            if (( $(echo "$ttft > $max_ttft" | bc -l 2>/dev/null) )); then
                max_ttft=$ttft
            fi
        done
        avg_ttft=$(echo "scale=3; $total_ttft / $ttft_count" | bc 2>/dev/null || echo "0")
    fi
    
    # Clean up
    rm -f "$temp_request" "$ab_output" 2>/dev/null
    
    # Return results as JSON
    echo "{
        \"deployment\": \"${deployment_name}\",
        \"config\": \"${config_name}\",
        \"mock_embeddings\": ${use_mock_embeddings},
        \"mock_llm\": ${use_mock_llm},
        \"concurrency\": ${concurrent},
        \"total_requests\": ${requests},
        \"successful\": ${successful},
        \"failed\": ${failed},
        \"requests_per_second\": ${rps:-0},
        \"mean_response_time_ms\": ${mean_time:-0},
        \"total_test_time_s\": ${total_time:-0},
        \"ttft_samples\": ${ttft_count},
        \"avg_ttft_s\": ${avg_ttft},
        \"min_ttft_s\": ${min_ttft},
        \"max_ttft_s\": ${max_ttft}
    }"
}

# Main execution starts here

echo -e "${GREEN}Step 1: Testing connectivity${NC}" | tee -a "$MAIN_LOG"
local_available=false
k8s_available=false

if test_connectivity "$LOCAL_BASE_URL" "Local"; then
    local_available=true
fi

if test_connectivity "$K8S_BASE_URL" "K8s"; then
    k8s_available=true
fi

if [[ "$local_available" == false && "$k8s_available" == false ]]; then
    echo -e "${RED}Error: Neither local nor K8s deployment is available!${NC}" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "" | tee -a "$MAIN_LOG"

# Run benchmarks for each configuration and deployment
all_results="[]"

for config_line in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r config_name use_mock_embeddings use_mock_llm description <<< "$config_line"
    
    echo -e "${GREEN}=== Configuration: ${config_name} (${description}) ===${NC}" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    
    # Test on local deployment
    if [[ "$local_available" == true ]]; then
        echo -e "${YELLOW}Testing Local deployment...${NC}" | tee -a "$MAIN_LOG"
        
        for concurrent in "${CONCURRENCY_LEVELS[@]}"; do
            if [[ $concurrent -gt $MAX_CONCURRENT ]]; then
                break
            fi
            
            # Calculate requests for this level  
            requests_for_level=$TOTAL_REQUESTS
            if [[ $concurrent -eq 1 ]]; then
                requests_for_level=10  # Fewer requests for single-threaded
            fi
            
            result=$(run_concurrent_benchmark "$LOCAL_BASE_URL" "local" "$config_name" "$use_mock_embeddings" "$use_mock_llm" "$concurrent" "$requests_for_level")
            all_results=$(echo "$all_results" | jq ". + [$result]")
            
            # Show quick result summary
            rps=$(echo "$result" | jq -r '.requests_per_second')
            ttft=$(echo "$result" | jq -r '.avg_ttft_s')
            echo "    Local C$concurrent: $rps RPS, ${ttft}s TTFT" | tee -a "$MAIN_LOG"
        done
        echo "" | tee -a "$MAIN_LOG"
    fi
    
    # Test on K8s deployment  
    if [[ "$k8s_available" == true ]]; then
        echo -e "${YELLOW}Testing K8s deployment...${NC}" | tee -a "$MAIN_LOG"
        
        for concurrent in "${CONCURRENCY_LEVELS[@]}"; do
            if [[ $concurrent -gt $MAX_CONCURRENT ]]; then
                break
            fi
            
            requests_for_level=$TOTAL_REQUESTS
            if [[ $concurrent -eq 1 ]]; then
                requests_for_level=10
            fi
            
            result=$(run_concurrent_benchmark "$K8S_BASE_URL" "k8s" "$config_name" "$use_mock_embeddings" "$use_mock_llm" "$concurrent" "$requests_for_level")
            all_results=$(echo "$all_results" | jq ". + [$result]")
            
            # Show quick result summary
            rps=$(echo "$result" | jq -r '.requests_per_second')
            ttft=$(echo "$result" | jq -r '.avg_ttft_s')
            echo "    K8s C$concurrent: $rps RPS, ${ttft}s TTFT" | tee -a "$MAIN_LOG"
        done
        echo "" | tee -a "$MAIN_LOG"
    fi
    
    echo "" | tee -a "$MAIN_LOG"
done

# Save results
RESULTS_FILE="${OUTPUT_DIR}/systematic_results_${TIMESTAMP}.json"
echo "$all_results" | jq '.' > "$RESULTS_FILE"

echo -e "${GREEN}=== Final Analysis ===${NC}" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Generate comparison tables for each configuration
for config_line in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r config_name use_mock_embeddings use_mock_llm description <<< "$config_line"
    
    echo -e "${GREEN}Configuration: ${config_name} - ${description}${NC}" | tee -a "$MAIN_LOG"
    echo "Conc | Local RPS | K8s RPS | Local TTFT | K8s TTFT | Ratio (L/K)" | tee -a "$MAIN_LOG"
    echo "-----|-----------|---------|------------|----------|-------------" | tee -a "$MAIN_LOG"
    
    for concurrent in "${CONCURRENCY_LEVELS[@]}"; do
        if [[ $concurrent -gt $MAX_CONCURRENT ]]; then
            break
        fi
        
        local_rps=$(echo "$all_results" | jq -r ".[] | select(.deployment==\"local\" and .config==\"$config_name\" and .concurrency==$concurrent) | .requests_per_second")
        k8s_rps=$(echo "$all_results" | jq -r ".[] | select(.deployment==\"k8s\" and .config==\"$config_name\" and .concurrency==$concurrent) | .requests_per_second")
        local_ttft=$(echo "$all_results" | jq -r ".[] | select(.deployment==\"local\" and .config==\"$config_name\" and .concurrency==$concurrent) | .avg_ttft_s")
        k8s_ttft=$(echo "$all_results" | jq -r ".[] | select(.deployment==\"k8s\" and .config==\"$config_name\" and .concurrency==$concurrent) | .avg_ttft_s")
        
        if [[ "$local_rps" != "null" && "$k8s_rps" != "null" && "$local_rps" != "0" && "$k8s_rps" != "0" ]]; then
            ratio=$(echo "scale=2; $local_rps / $k8s_rps" | bc 2>/dev/null || echo "N/A")
            printf "%4s | %9s | %7s | %10s | %8s | %11s\n" "$concurrent" "$local_rps" "$k8s_rps" "$local_ttft" "$k8s_ttft" "$ratio" | tee -a "$MAIN_LOG"
        fi
    done
    echo "" | tee -a "$MAIN_LOG"
done

echo "" | tee -a "$MAIN_LOG"
echo "Detailed results saved to: $RESULTS_FILE" | tee -a "$MAIN_LOG"
echo "Full log saved to: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo -e "${GREEN}=== Key Findings ===${NC}" | tee -a "$MAIN_LOG"
echo "1. Compare 'baseline' results to see overall Local vs K8s performance difference" | tee -a "$MAIN_LOG"
echo "2. Compare 'mock-embeddings' to isolate embedding computation bottlenecks" | tee -a "$MAIN_LOG" 
echo "3. Compare 'mock-llm' to isolate LLM API communication bottlenecks" | tee -a "$MAIN_LOG"
echo "4. Compare 'mock-both' to isolate pure FastAPI + vector store bottlenecks" | tee -a "$MAIN_LOG"
echo "5. Look for patterns in the Ratio column - consistently high ratios indicate K8s bottlenecks" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "Benchmark completed successfully! 🎉" | tee -a "$MAIN_LOG"