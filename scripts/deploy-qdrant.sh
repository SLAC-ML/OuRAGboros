#!/bin/bash
set -e

echo "🚀 Deploying OuRAGboros with Qdrant to Kubernetes..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if we can connect to cluster
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

# Create namespace if it doesn't exist
echo "📝 Creating namespace..."
kubectl create namespace ouragboros --dry-run=client -o yaml | kubectl apply -f -

# Apply Stanford AI secret if it exists
if [ -f "k8s/base/stanford-ai-secret.yaml" ]; then
    echo "🔐 Applying Stanford AI secret..."
    kubectl apply -f k8s/base/stanford-ai-secret.yaml -n ouragboros
fi

# Deploy with Qdrant overlay
echo "🔧 Deploying Qdrant configuration..."
kubectl apply -k k8s/overlays/qdrant

echo "⏳ Waiting for deployments to be ready..."

# Wait for Qdrant to be ready first
echo "🔍 Waiting for Qdrant..."
kubectl wait --for=condition=available --timeout=300s deployment/qdrant -n ouragboros

# Wait for OpenSearch (if still deployed)
if kubectl get deployment opensearch -n ouragboros > /dev/null 2>&1; then
    echo "🔍 Waiting for OpenSearch..."
    kubectl wait --for=condition=available --timeout=300s deployment/opensearch -n ouragboros
fi

# Wait for Ollama
echo "🔍 Waiting for Ollama..."
kubectl wait --for=condition=available --timeout=300s deployment/ollama -n ouragboros

# Wait for OuRAGboros app
echo "🔍 Waiting for OuRAGboros..."
kubectl wait --for=condition=available --timeout=300s deployment/ouragboros -n ouragboros

echo "✅ All deployments are ready!"

# Show status
echo ""
echo "📊 Deployment Status:"
kubectl get pods -n ouragboros
echo ""
kubectl get svc -n ouragboros

echo ""
echo "🎉 OuRAGboros with Qdrant deployed successfully!"
echo ""
echo "📡 Access the application:"
echo "  Streamlit UI: kubectl port-forward -n ouragboros svc/ouragboros 8501:8501"
echo "  FastAPI:      kubectl port-forward -n ouragboros svc/ouragboros 8001:8001"
echo "  Qdrant UI:    kubectl port-forward -n ouragboros svc/qdrant 6333:6333"
echo ""
echo "🧪 Test Qdrant performance:"
echo "  kubectl exec -it -n ouragboros deployment/ouragboros -- python -c \"
from qdrant_client import QdrantClient
client = QdrantClient(url='http://qdrant:6333')
collections = client.get_collections()
print(f'Qdrant connected! Collections: {len(collections.collections)}')
\""
echo ""
echo "🔍 Monitor logs:"
echo "  kubectl logs -f -n ouragboros deployment/ouragboros"
echo "  kubectl logs -f -n ouragboros deployment/qdrant"