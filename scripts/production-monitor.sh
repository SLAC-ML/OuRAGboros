#!/bin/bash

# Production deployment monitoring script
echo "🚀 OuRAGboros Production Monitoring"
echo "===================================="

while true; do
    clear
    echo "🚀 OuRAGboros Production Status - $(date)"
    echo "===================================="
    echo ""
    
    echo "📊 Deployment Status:"
    kubectl get deployments -n ouragboros
    echo ""
    
    echo "📱 Pod Status & Distribution:"
    kubectl get pods -n ouragboros -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName,AGE:.metadata.creationTimestamp --sort-by=.spec.nodeName
    echo ""
    
    echo "💾 Resource Usage:"
    kubectl top pods -n ouragboros --containers 2>/dev/null || echo "Resource metrics updating..."
    echo ""
    
    # Count ready vs total
    READY_PODS=$(kubectl get pods -n ouragboros -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)
    TOTAL_PODS=$(kubectl get pods -n ouragboros -o jsonpath='{.items[*].metadata.name}' | wc -w)
    
    echo "🎯 Progress: $READY_PODS/$TOTAL_PODS pods ready"
    
    if [ "$READY_PODS" -eq "$TOTAL_PODS" ]; then
        echo ""
        echo "🎉 ALL PODS READY! Production deployment complete!"
        echo ""
        echo "🌐 Access your application:"
        echo "   kubectl port-forward -n ouragboros svc/ouragboros 8501:8501"
        echo ""
        echo "📈 Your cluster now runs:"
        echo "   • 2 Ollama replicas (LLM inference with load balancing)"
        echo "   • 3 OuRAGboros replicas (Web UI with load balancing)"
        echo "   • 1 OpenSearch replica (Enhanced vector database)"
        echo ""
        echo "Press Ctrl+C to exit monitoring"
    fi
    
    echo ""
    echo "🔄 Refreshing in 10 seconds... (Ctrl+C to stop)"
    sleep 10
done