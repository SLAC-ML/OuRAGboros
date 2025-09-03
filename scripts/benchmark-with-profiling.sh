#!/bin/bash

# Enhanced Benchmark Script with Resource Profiling
# Combines systematic benchmarking with detailed resource monitoring

set -e

# Configuration
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:8001}"
K8S_BASE_URL="${K8S_BASE_URL:-}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-50}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'  
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Output setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark-results-profiled"
mkdir -p "$OUTPUT_DIR"

BENCHMARK_LOG="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.log"
PROFILE_LOG="${OUTPUT_DIR}/profiling_${TIMESTAMP}.log"

echo "🚀 Enhanced Benchmark with Resource Profiling" | tee "$BENCHMARK_LOG"
echo "=============================================" | tee -a "$BENCHMARK_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$BENCHMARK_LOG"
echo "Local URL: $LOCAL_BASE_URL" | tee -a "$BENCHMARK_LOG"
echo "K8s URL: ${K8S_BASE_URL:-Not configured}" | tee -a "$BENCHMARK_LOG"
echo "" | tee -a "$BENCHMARK_LOG"

# Function to check if services are running locally
check_local_services() {
    echo -e "${YELLOW}Checking local services...${NC}" | tee -a "$BENCHMARK_LOG"
    
    # Check if FastAPI is running
    local fastapi_running=false
    if curl -s -f "$LOCAL_BASE_URL/docs" >/dev/null 2>&1; then
        fastapi_running=true
        echo -e "${GREEN}✓ FastAPI service is running at $LOCAL_BASE_URL${NC}" | tee -a "$BENCHMARK_LOG"
    else
        echo -e "${RED}✗ FastAPI service not found at $LOCAL_BASE_URL${NC}" | tee -a "$BENCHMARK_LOG"
        echo "  To start: cd $(dirname $0)/.. && uv run uvicorn src.app_api:app --reload --port 8001" | tee -a "$BENCHMARK_LOG"
    fi
    
    # Check what OuRAGboros processes are running
    echo "Currently running OuRAGboros processes:" | tee -a "$BENCHMARK_LOG"
    pgrep -fl "uvicorn.*app_api\|streamlit.*main.py\|python.*rag" | tee -a "$BENCHMARK_LOG" || echo "  None found" | tee -a "$BENCHMARK_LOG"
    
    return $([[ "$fastapi_running" == true ]] && echo 0 || echo 1)
}

# Function to run a single benchmark test with profiling
run_profiled_benchmark() {
    local deployment_name="$1"
    local base_url="$2"
    local config_name="$3"
    local use_mock_embeddings="$4"
    local use_mock_llm="$5"
    local concurrency="$6"
    
    echo -e "${BLUE}Running ${deployment_name} benchmark: ${config_name} (C=${concurrency})${NC}" | tee -a "$BENCHMARK_LOG"
    
    # Set up environment for local testing
    local env_vars=""
    if [[ "$deployment_name" == "local" ]]; then
        env_vars="USE_MOCK_EMBEDDINGS=$use_mock_embeddings USE_MOCK_LLM=$use_mock_llm"
    fi
    
    # Prepare profiler output files
    local profile_output="${OUTPUT_DIR}/profile_${deployment_name}_${config_name}_c${concurrency}_${TIMESTAMP}.json"
    
    # Create request payload
    local request_payload='{
        "query": "What are the fundamental principles of quantum mechanics and how do they relate to wave-particle duality?",
        "embedding_model": "huggingface:thellert/physbert_cased",
        "llm_model": "stanford:gpt-4o-mini", 
        "prompt": "You are a knowledgeable physics researcher.",
        "use_rag": true,
        "use_qdrant": true,
        "knowledge_base": "default",
        "max_documents": 5,
        "score_threshold": 0.0
    }'
    
    # Start resource profiler in background
    echo "  📊 Starting resource profiler..." | tee -a "$BENCHMARK_LOG"
    if [[ "$deployment_name" == "local" && -n "$env_vars" ]]; then
        env $env_vars python3 scripts/profile-resources.py --interval 0.5 --output "$profile_output" &
    else
        python3 scripts/profile-resources.py --interval 0.5 --output "$profile_output" &
    fi
    local profiler_pid=$!
    
    # Give profiler time to start
    sleep 2
    
    # Measure baseline performance (single request TTFT)
    echo "  ⚡ Measuring baseline TTFT..." | tee -a "$BENCHMARK_LOG"
    local baseline_start=$(date +%s.%N)
    local baseline_ttft=""
    
    {
        if [[ "$deployment_name" == "local" && -n "$env_vars" ]]; then
            env $env_vars curl -s -N -m 30 -X POST \
                -H "Content-Type: application/json" \
                -d "$request_payload" \
                "${base_url}/ask/stream" 2>/dev/null
        else
            curl -s -N -m 30 -X POST \
                -H "Content-Type: application/json" \
                -d "$request_payload" \
                "${base_url}/ask/stream" 2>/dev/null
        fi
    } | while IFS= read -r line; do
        if [[ "$line" == data:* ]]; then
            json_data="${line#data: }"
            if [[ -n "$json_data" ]]; then
                event_type=$(echo "$json_data" | jq -r '.type' 2>/dev/null || echo "")
                if [[ "$event_type" == "token" ]]; then
                    baseline_ttft=$(echo "$(date +%s.%N) - $baseline_start" | bc)
                    echo "BASELINE_TTFT:$baseline_ttft"
                    break
                fi
            fi
        fi
    done 2>/dev/null &
    
    local curl_pid=$!
    wait $curl_pid 2>/dev/null
    
    # Extract baseline TTFT if captured
    if baseline_result=$(wait $curl_pid 2>/dev/null | grep "BASELINE_TTFT:" | cut -d: -f2); then
        echo "  📏 Baseline TTFT: ${baseline_result}s" | tee -a "$BENCHMARK_LOG"
    else
        baseline_result="unknown"
        echo "  ⚠️  Could not measure baseline TTFT" | tee -a "$BENCHMARK_LOG"
    fi
    
    # Run concurrent load test
    echo "  🔥 Running concurrent load test (C=$concurrency)..." | tee -a "$BENCHMARK_LOG"
    
    # Create temporary request file
    local temp_request="/tmp/bench_request_${TIMESTAMP}_${concurrency}.json"
    echo "$request_payload" > "$temp_request"
    
    # Run Apache Bench if available
    local ab_result=""
    if command -v ab >/dev/null 2>&1; then
        local ab_output="/tmp/ab_${deployment_name}_${config_name}_c${concurrency}.txt"
        
        # Set environment and run ab
        if [[ "$deployment_name" == "local" && -n "$env_vars" ]]; then
            timeout 180 env $env_vars ab -n $((concurrency * 5)) -c "$concurrency" \
                -T "application/json" -p "$temp_request" \
                "${base_url}/ask/stream" > "$ab_output" 2>&1 || true
        else
            timeout 180 ab -n $((concurrency * 5)) -c "$concurrency" \
                -T "application/json" -p "$temp_request" \
                "${base_url}/ask/stream" > "$ab_output" 2>&1 || true
        fi
        
        # Parse ab results
        if [[ -f "$ab_output" ]]; then
            local rps=$(grep "Requests per second:" "$ab_output" | awk '{print $4}' | head -1)
            local mean_time=$(grep "Time per request:" "$ab_output" | grep "(mean)" | awk '{print $4}' | head -1)
            local failed=$(grep "Failed requests:" "$ab_output" | awk '{print $3}' | head -1)
            
            ab_result="RPS: ${rps:-0}, Mean: ${mean_time:-0}ms, Failed: ${failed:-0}"
            echo "  📈 Apache Bench: $ab_result" | tee -a "$BENCHMARK_LOG"
        fi
    else
        echo "  ⚠️  Apache Bench not available, skipping load test" | tee -a "$BENCHMARK_LOG"
    fi
    
    # Let the test run for a bit longer to gather profiling data
    sleep 10
    
    # Stop profiler
    echo "  🛑 Stopping profiler..." | tee -a "$BENCHMARK_LOG"
    kill $profiler_pid 2>/dev/null || true
    wait $profiler_pid 2>/dev/null || true
    
    # Clean up
    rm -f "$temp_request" "/tmp/ab_${deployment_name}_${config_name}_c${concurrency}.txt" 2>/dev/null || true
    
    # Generate quick summary if profile exists
    if [[ -f "$profile_output" ]]; then
        local sample_count=$(jq -r '.metadata.total_samples' "$profile_output" 2>/dev/null || echo "0")
        local avg_cpu=$(jq -r '[.data_points[].system.cpu_percent] | add / length' "$profile_output" 2>/dev/null || echo "0")
        local max_memory=$(jq -r '[.data_points[].system.memory_used_gb] | max' "$profile_output" 2>/dev/null || echo "0")
        local avg_threads=$(jq -r '[.data_points[].processes.total_threads] | add / length' "$profile_output" 2>/dev/null || echo "0")
        
        echo "  📊 Resource Summary: CPU avg ${avg_cpu}%, RAM max ${max_memory}GB, Threads avg ${avg_threads}, Samples ${sample_count}" | tee -a "$BENCHMARK_LOG"
    fi
    
    echo "  ✅ Complete: $profile_output" | tee -a "$BENCHMARK_LOG"
    echo "" | tee -a "$BENCHMARK_LOG"
}

# Function to run systematic tests
run_systematic_tests() {
    local deployment_name="$1" 
    local base_url="$2"
    
    echo -e "${GREEN}=== Testing ${deployment_name} deployment ===${NC}" | tee -a "$BENCHMARK_LOG"
    echo "" | tee -a "$BENCHMARK_LOG"
    
    # Test configurations: name:mock_embeddings:mock_llm:description
    local configs=(
        "baseline:false:false:Full pipeline (no mocking)"
        "mock-embeddings:true:false:Mock embeddings only" 
        "mock-llm:false:true:Mock LLM only"
        "mock-both:true:true:Mock embeddings + LLM"
    )
    
    # Concurrency levels to test
    local concurrency_levels=(1 5 10 15 20)
    
    for config_line in "${configs[@]}"; do
        IFS=':' read -r config_name use_mock_embeddings use_mock_llm description <<< "$config_line"
        
        echo -e "${YELLOW}Configuration: ${config_name} - ${description}${NC}" | tee -a "$BENCHMARK_LOG"
        
        for concurrency in "${concurrency_levels[@]}"; do
            if [[ $concurrency -gt $MAX_CONCURRENT ]]; then
                break
            fi
            
            run_profiled_benchmark "$deployment_name" "$base_url" "$config_name" \
                "$use_mock_embeddings" "$use_mock_llm" "$concurrency"
        done
        
        echo "" | tee -a "$BENCHMARK_LOG"
    done
}

# Main execution
echo -e "${GREEN}Step 1: Service discovery${NC}" | tee -a "$BENCHMARK_LOG"
local_available=false

if check_local_services; then
    local_available=true
    echo -e "${GREEN}✓ Local services ready for benchmarking${NC}" | tee -a "$BENCHMARK_LOG"
else
    echo -e "${RED}✗ Local services not available${NC}" | tee -a "$BENCHMARK_LOG"
    echo "Please start the FastAPI service first:" | tee -a "$BENCHMARK_LOG"
    echo "  cd $(dirname $0)/.." | tee -a "$BENCHMARK_LOG"
    echo "  uv run uvicorn src.app_api:app --reload --port 8001" | tee -a "$BENCHMARK_LOG"
fi

echo "" | tee -a "$BENCHMARK_LOG"

# Check Python dependencies for profiler
echo -e "${YELLOW}Checking profiler dependencies...${NC}" | tee -a "$BENCHMARK_LOG"
if python3 -c "import psutil, json" 2>/dev/null; then
    echo -e "${GREEN}✓ Profiler dependencies available${NC}" | tee -a "$BENCHMARK_LOG"
else
    echo -e "${RED}✗ Missing psutil library${NC}" | tee -a "$BENCHMARK_LOG"
    echo "  Install with: pip3 install psutil" | tee -a "$BENCHMARK_LOG"
    exit 1
fi

echo "" | tee -a "$BENCHMARK_LOG"

# Run benchmarks
if [[ "$local_available" == true ]]; then
    echo -e "${GREEN}Step 2: Running systematic benchmarks${NC}" | tee -a "$BENCHMARK_LOG"
    run_systematic_tests "local" "$LOCAL_BASE_URL"
else
    echo -e "${RED}Cannot run benchmarks without local services${NC}" | tee -a "$BENCHMARK_LOG"
    exit 1
fi

# Generate final summary
echo -e "${GREEN}=== Benchmark Complete ===${NC}" | tee -a "$BENCHMARK_LOG"
echo "" | tee -a "$BENCHMARK_LOG"
echo "Results directory: $OUTPUT_DIR" | tee -a "$BENCHMARK_LOG"
echo "Benchmark log: $BENCHMARK_LOG" | tee -a "$BENCHMARK_LOG"
echo "" | tee -a "$BENCHMARK_LOG"

# List generated files
echo "Generated files:" | tee -a "$BENCHMARK_LOG"
ls -la "$OUTPUT_DIR"/*${TIMESTAMP}* | tee -a "$BENCHMARK_LOG"

echo "" | tee -a "$BENCHMARK_LOG"
echo -e "${GREEN}🎉 Profiled benchmark complete!${NC}" | tee -a "$BENCHMARK_LOG"
echo "" | tee -a "$BENCHMARK_LOG"

echo "Next steps:" | tee -a "$BENCHMARK_LOG"
echo "1. Analyze the JSON profile files for resource usage patterns" | tee -a "$BENCHMARK_LOG"
echo "2. Compare baseline vs mock configurations to isolate bottlenecks" | tee -a "$BENCHMARK_LOG"
echo "3. Look for CPU, memory, or thread count spikes during concurrent load" | tee -a "$BENCHMARK_LOG"
echo "4. Correlate resource usage with performance metrics (RPS, TTFT)" | tee -a "$BENCHMARK_LOG"