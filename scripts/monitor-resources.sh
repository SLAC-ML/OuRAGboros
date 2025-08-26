#!/bin/bash

# Resource monitoring script for OuRAGboros deployment

echo "🔍 OuRAGboros Resource Monitor"
echo "=============================="
echo ""

while true; do
    clear
    echo "🔍 OuRAGboros Resource Monitor - $(date)"
    echo "=============================="
    echo ""
    
    echo "📊 Pod Status:"
    kubectl get pods -n ouragboros
    echo ""
    
    echo "💾 Resource Usage:"
    kubectl top pod -n ouragboros --containers 2>/dev/null || echo "Metrics server not available"
    echo ""
    
    echo "⚠️  Resource Limits:"
    echo "OuRAGboros: 6Gi memory, 2 CPU cores"
    echo "Ollama:     8Gi memory, 4 CPU cores"
    echo "OpenSearch: 2Gi memory, 1 CPU core"
    echo ""
    
    echo "🔄 Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done