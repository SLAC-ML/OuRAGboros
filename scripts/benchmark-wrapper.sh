#!/bin/bash

# Wrapper to run benchmark scripts using the existing OuRAGboros container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if services are running
echo "🔍 Checking Docker services..."
if ! docker compose ps | grep -q "ouragboros.*running"; then
    echo "❌ OuRAGboros service is not running!"
    echo "💡 Start it with: docker compose up -d"
    exit 1
fi

echo "✅ Services are running"

# Get the container name
CONTAINER=$(docker compose ps -q ouragboros)
if [ -z "$CONTAINER" ]; then
    echo "❌ Could not find OuRAGboros container"
    exit 1
fi

echo "📦 Container ID: $CONTAINER"

# Function to run Python scripts in the container
run_in_container() {
    local script="$1"
    shift
    echo "🚀 Running: $script $@"
    
    # Copy the script into the container if needed
    if [[ "$script" == *.py ]]; then
        docker exec -i "$CONTAINER" python3 "/app/$script" "$@"
    else
        docker exec -i "$CONTAINER" bash -c "$script $@"
    fi
}

# Main execution
case "${1:-test}" in
    test)
        echo "🧪 Running mock validation test..."
        run_in_container "scripts/test-mock-validation.py"
        ;;
    
    systematic)
        echo "📊 Running systematic benchmark..."
        run_in_container "scripts/test-concurrency-systematic.sh"
        ;;
    
    profile)
        echo "📈 Running benchmark with profiling..."
        run_in_container "scripts/benchmark-with-profiling.sh"
        ;;
    
    custom)
        shift
        echo "🔧 Running custom command: $@"
        run_in_container "$@"
        ;;
    
    *)
        echo "Usage: $0 [test|systematic|profile|custom <command>]"
        echo "  test       - Run mock validation tests"
        echo "  systematic - Run systematic concurrency benchmark"
        echo "  profile    - Run benchmark with resource profiling"
        echo "  custom     - Run custom command in container"
        exit 1
        ;;
esac