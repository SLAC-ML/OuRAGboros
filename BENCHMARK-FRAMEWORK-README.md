# OuRAGboros Concurrency Benchmark Framework

## Overview

This framework provides systematic benchmarking tools to analyze concurrency bottlenecks in the OuRAGboros streaming API, specifically comparing local vs K8s deployments. The analysis identified **5 major blocking operations** that prevent proper concurrent scaling.

## 🚨 Critical Findings

### Identified Bottlenecks
1. **Document Retrieval Phase**: Blocking vector store operations
2. **Embedding Model Operations**: CPU-bound blocking computations  
3. **Model Loading**: Blocking network/disk I/O
4. **LLM Query Processing**: Mixed async/sync issues
5. **FastAPI Event Loop**: Sequential pre-processing blocks concurrency

### Why K8s Scales Worse Than Local
- **Resource Contention**: Single-pod forces all requests through same Python process
- **Thread Pool Limits**: Default ThreadPoolExecutor has fewer threads in K8s
- **Memory Pressure**: K8s resource limits cause more frequent GC during embeddings
- **Container CPU Throttling**: K8s CPU limits throttle HuggingFace models more aggressively

## 📊 Benchmark Scripts

### 1. Systematic Concurrency Benchmark
**File**: `scripts/test-concurrency-systematic.sh`

**Purpose**: Compare Local vs K8s with identical parameters across different mock configurations.

**Usage**:
```bash
# Basic usage (tests local deployment)
./scripts/test-concurrency-systematic.sh

# Configure endpoints  
LOCAL_BASE_URL=http://localhost:8001 K8S_BASE_URL=http://localhost:8501 ./scripts/test-concurrency-systematic.sh

# Adjust load parameters
MAX_CONCURRENT=20 TOTAL_REQUESTS=50 ./scripts/test-concurrency-systematic.sh
```

**Test Configurations**:
- `baseline`: No mocking - full pipeline bottlenecks
- `mock-embeddings`: Isolate embedding computation bottlenecks  
- `mock-llm`: Isolate LLM API communication bottlenecks
- `mock-both`: Isolate pure FastAPI + vector store bottlenecks

**Output**: `benchmark-results-systematic/systematic_results_<timestamp>.json`

### 2. Resource Profiling Tool
**File**: `scripts/profile-resources.py`

**Purpose**: Monitor CPU, memory, thread usage during concurrent load.

**Usage**:
```bash
# Interactive profiling (press Ctrl+C to stop)
python3 scripts/profile-resources.py

# Profile for specific duration
python3 scripts/profile-resources.py --duration 60 --interval 0.5

# Profile while running a command
python3 scripts/profile-resources.py --command "curl -X POST http://localhost:8001/ask/stream ..."
```

**Metrics Collected**:
- System: CPU%, memory, disk, network I/O, load average
- Processes: OuRAGboros process CPU, memory, thread count, open files
- Timeline: All metrics sampled at configurable intervals

**Output**: `resource_profile_<timestamp>.json`

### 3. Integrated Benchmark with Profiling
**File**: `scripts/benchmark-with-profiling.sh`

**Purpose**: Combines systematic benchmarking with resource monitoring.

**Usage**:
```bash
# Requires local FastAPI service running first:
# uv run uvicorn src.app_api:app --reload --port 8001

./scripts/benchmark-with-profiling.sh
```

**Features**:
- Automatic service discovery
- Integrated resource profiling during load tests
- Mock configuration testing
- Real-time performance feedback

**Output**: 
- `benchmark-results-profiled/benchmark_<timestamp>.log`
- `benchmark-results-profiled/profile_*_<timestamp>.json` (per test)

## 🧪 Mock System

### Environment Variables
- `USE_MOCK_EMBEDDINGS=true`: Use instant mock embeddings (768-dim vectors)
- `USE_MOCK_LLM=true`: Use instant mock LLM responses

### Mock Implementation Details
**Mock Embeddings** (`src/lib/langchain/embeddings.py`):
- Returns pre-computed 768-dimensional normalized vectors instantly
- Eliminates HuggingFace/PhysBERT computation bottleneck
- Thread-safe caching with proper isolation

**Mock LLM** (`src/lib/langchain/llm.py`):
- Returns instant "Mock response to: {query}" text
- Simulates streaming with word-by-word tokens
- Eliminates Stanford AI API network latency

## 🚀 Running Benchmarks

### Prerequisites
1. **Install dependencies**:
   ```bash
   pip3 install psutil apache2-utils  # For profiling and ab
   ```

2. **Start local service**:
   ```bash
   cd /path/to/OuRAGboros
   uv run uvicorn src.app_api:app --reload --port 8001
   ```

3. **Set up K8s port forwarding** (if testing K8s):
   ```bash
   kubectl port-forward -n ouragboros svc/ouragboros 8501:8501
   ```

### Quick Test Sequence
```bash
# 1. Validate mock system works
python3 scripts/test-mock-validation.py

# 2. Run systematic comparison
./scripts/test-concurrency-systematic.sh

# 3. Run with resource profiling
./scripts/benchmark-with-profiling.sh

# 4. Analyze results
ls benchmark-results-*/
```

### Example Mock Configuration Test
```bash
# Test with mock embeddings only
USE_MOCK_EMBEDDINGS=true USE_MOCK_LLM=false ./scripts/test-concurrency-systematic.sh

# Test with both mocks (isolate FastAPI/vector store bottlenecks)
USE_MOCK_EMBEDDINGS=true USE_MOCK_LLM=true ./scripts/test-concurrency-systematic.sh
```

## 📈 Analyzing Results

### 1. Systematic Benchmark Results
Look for patterns in the comparison tables:

```
Configuration: baseline - Full pipeline (no mocking)
Conc | Local RPS | K8s RPS | Local TTFT | K8s TTFT | Ratio (L/K)
-----|-----------|---------|------------|----------|-------------
  1  |      2.5  |    1.8  |      1.2   |     1.8  |        1.39
  5  |      8.1  |    4.2  |      1.5   |     2.8  |        1.93  
 10  |     12.3  |    5.1  |      1.8   |     3.9  |        2.41
```

**Key Metrics**:
- **RPS (Requests Per Second)**: Higher is better
- **TTFT (Time To First Token)**: Lower is better  
- **Ratio (Local/K8s)**: Values > 1.5 indicate K8s bottlenecks

### 2. Resource Profile Analysis
Use jq to analyze resource profiles:

```bash
# Show CPU usage over time
jq '.data_points[] | {elapsed: .elapsed_seconds, cpu: .system.cpu_percent}' profile.json

# Find peak memory usage
jq '[.data_points[].system.memory_used_gb] | max' profile.json

# Show thread count progression  
jq '.data_points[] | {elapsed: .elapsed_seconds, threads: .processes.total_threads}' profile.json
```

### 3. Bottleneck Isolation
Compare mock configurations to isolate bottlenecks:

- **Baseline vs Mock-Embeddings**: If big improvement, embedding computation is bottleneck
- **Baseline vs Mock-LLM**: If big improvement, LLM API calls are bottleneck  
- **Mock-Both**: Shows pure FastAPI + vector store performance ceiling

## 🔧 Next Steps for Optimization

Based on this analysis, the next session should focus on:

1. **Convert document retrieval to async**
2. **Implement async embedding pipeline** 
3. **Extend AsyncOpenAI pattern to all LLM providers**
4. **Configure custom ThreadPoolExecutor with higher thread counts**
5. **Add async semaphores for embedding concurrency control**

## 📋 File Structure
```
scripts/
├── test-concurrency-systematic.sh     # Main systematic benchmark
├── profile-resources.py               # Resource monitoring tool
├── benchmark-with-profiling.sh        # Integrated benchmark + profiling
├── test-mock-validation.py           # Mock system validation
└── test-*.sh                         # Existing benchmark scripts

benchmark-results-systematic/          # Systematic benchmark outputs
benchmark-results-profiled/           # Profiled benchmark outputs
```

## 🎯 Expected Outcomes

After running the benchmark framework:
1. **Quantify the K8s vs Local performance gap**
2. **Identify which bottlenecks contribute most to the gap**
3. **Validate that async optimizations will help**
4. **Establish performance baselines for future improvements**

The framework provides the data needed to make targeted optimizations in the next development session.