#!/bin/bash

# TTFT (Time To First Token) Benchmark Script for OuRAGboros
# This script provides a simple interface to run TTFT measurements with
# automatic environment detection and reasonable defaults.

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
DEFAULT_SAMPLES=10
DEFAULT_CONCURRENCY=5
DEFAULT_TIMEOUT=30
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTFT_SCRIPT="${SCRIPT_DIR}/ttft_test.py"

# Function to check if required dependencies are installed
check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is required but not installed${NC}"
        exit 1
    fi
    
    # Check if httpx and httpx-sse are installed
    if ! python3 -c "import httpx, httpx_sse" 2>/dev/null; then
        echo -e "${YELLOW}⚠️ Required Python packages missing. Installing...${NC}"
        echo -e "Installing httpx and httpx-sse..."
        if command -v uv &> /dev/null; then
            uv add httpx httpx-sse
        elif command -v pip3 &> /dev/null; then
            pip3 install httpx httpx-sse
        else
            echo -e "${RED}❌ Neither uv nor pip3 found. Please install httpx and httpx-sse manually.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}✅ Dependencies OK${NC}"
}

# Function to detect if we're running in a k8s environment
detect_environment() {
    if kubectl get svc ouragboros -n ouragboros &>/dev/null; then
        echo "k8s"
    elif curl -s http://localhost:8001/docs &>/dev/null; then
        echo "local"
    else
        echo "unknown"
    fi
}

# Function to setup port forwarding for k8s
setup_k8s_port_forward() {
    local namespace="${1:-ouragboros}"
    local service="${2:-ouragboros}"
    
    echo -e "${BLUE}🚢 Setting up Kubernetes port-forward...${NC}"
    
    # Check if port-forward is already running
    if lsof -i :8001 &>/dev/null; then
        echo -e "${YELLOW}⚠️ Port 8001 is already in use${NC}"
        echo -e "If it's not a port-forward to OuRAGboros, please stop it first"
        return 0
    fi
    
    # Start port-forward in background
    echo -e "Starting: kubectl port-forward -n ${namespace} svc/${service} 8001:8001"
    kubectl port-forward -n "${namespace}" "svc/${service}" 8001:8001 >/dev/null 2>&1 &
    local pf_pid=$!
    
    # Wait for port-forward to be ready
    local attempts=0
    while [[ $attempts -lt 10 ]]; do
        if curl -s http://localhost:8001/docs &>/dev/null; then
            echo -e "${GREEN}✅ Port-forward ready (PID: ${pf_pid})${NC}"
            echo "${pf_pid}" > /tmp/ouragboros-ttft-portforward.pid
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    
    echo -e "${RED}❌ Port-forward failed to start${NC}"
    kill $pf_pid 2>/dev/null || true
    return 1
}

# Function to cleanup port-forward
cleanup_port_forward() {
    if [[ -f /tmp/ouragboros-ttft-portforward.pid ]]; then
        local pid=$(cat /tmp/ouragboros-ttft-portforward.pid)
        if kill "$pid" 2>/dev/null; then
            echo -e "${BLUE}🧹 Cleaned up port-forward (PID: ${pid})${NC}"
        fi
        rm -f /tmp/ouragboros-ttft-portforward.pid
    fi
}

# Trap to cleanup on exit
trap cleanup_port_forward EXIT

# Function to show usage
show_usage() {
    echo "TTFT (Time To First Token) Benchmark for OuRAGboros"
    echo ""
    echo "Usage:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --samples N       Number of measurements (default: ${DEFAULT_SAMPLES})"
    echo "  -c, --concurrency N   Concurrent requests (default: ${DEFAULT_CONCURRENCY})"
    echo "  -t, --timeout N       Timeout in seconds (default: ${DEFAULT_TIMEOUT})"
    echo "  -e, --endpoint URL    API endpoint (default: auto-detect)"
    echo "  -o, --output FILE     Output JSON file (default: auto-generated)"
    echo "  -k, --k8s             Force Kubernetes mode with port-forward"
    echo "  -l, --local           Force local mode (localhost:8001)"
    echo "  -n, --namespace NS    K8s namespace (default: ouragboros)"
    echo "  --service NAME        K8s service name (default: ouragboros)"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Auto-detect environment, 10 samples"
    echo "  $0 -s 20 -c 10                      # 20 samples with 10 concurrent"
    echo "  $0 -k -s 50                         # Force k8s mode, 50 samples"
    echo "  $0 -e http://my-api.com:8001 -s 30  # Custom endpoint"
    echo ""
}

# Parse command line arguments
SAMPLES=$DEFAULT_SAMPLES
CONCURRENCY=$DEFAULT_CONCURRENCY
TIMEOUT=$DEFAULT_TIMEOUT
ENDPOINT=""
OUTPUT=""
FORCE_MODE=""
NAMESPACE="ouragboros"
SERVICE="ouragboros"

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--samples)
            SAMPLES="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -e|--endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -k|--k8s)
            FORCE_MODE="k8s"
            shift
            ;;
        -l|--local)
            FORCE_MODE="local"
            shift
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --service)
            SERVICE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${GREEN}=== OuRAGboros TTFT Benchmark ===${NC}"
echo ""

# Check dependencies
check_dependencies

# Determine endpoint if not provided
if [[ -z "$ENDPOINT" ]]; then
    if [[ "$FORCE_MODE" == "local" ]]; then
        ENDPOINT="http://localhost:8001"
        env_type="local (forced)"
    elif [[ "$FORCE_MODE" == "k8s" ]]; then
        setup_k8s_port_forward "$NAMESPACE" "$SERVICE"
        ENDPOINT="http://localhost:8001"
        env_type="k8s (forced)"
    else
        env_type=$(detect_environment)
        case $env_type in
            "k8s")
                setup_k8s_port_forward "$NAMESPACE" "$SERVICE"
                ENDPOINT="http://localhost:8001"
                ;;
            "local")
                ENDPOINT="http://localhost:8001"
                ;;
            *)
                echo -e "${RED}❌ Cannot detect OuRAGboros API. Please specify --endpoint or check your setup${NC}"
                echo -e "   Try: $0 --local    (if running locally)"
                echo -e "   Try: $0 --k8s      (if running on Kubernetes)"
                exit 1
                ;;
        esac
    fi
else
    env_type="manual"
fi

echo -e "${BLUE}📡 Environment: ${env_type}${NC}"
echo -e "${BLUE}🎯 Endpoint: ${ENDPOINT}${NC}"
echo -e "${BLUE}📊 Samples: ${SAMPLES} (concurrency: ${CONCURRENCY})${NC}"
echo ""

# Build python command arguments
PYTHON_ARGS=(
    "--endpoint" "$ENDPOINT"
    "--samples" "$SAMPLES"
    "--concurrency" "$CONCURRENCY"
    "--timeout" "$TIMEOUT"
)

if [[ -n "$OUTPUT" ]]; then
    PYTHON_ARGS+=("--output" "$OUTPUT")
fi

# Run the TTFT benchmark
echo -e "${GREEN}🚀 Starting TTFT measurements...${NC}"
echo ""

if python3 "$TTFT_SCRIPT" "${PYTHON_ARGS[@]}"; then
    echo ""
    echo -e "${GREEN}✅ TTFT benchmark completed successfully!${NC}"
else
    exit_code=$?
    echo ""
    echo -e "${RED}❌ TTFT benchmark failed (exit code: ${exit_code})${NC}"
    exit $exit_code
fi