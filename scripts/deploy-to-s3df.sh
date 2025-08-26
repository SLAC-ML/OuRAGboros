#!/bin/bash

# OuRAGboros Deployment Script for S3DF vCluster
# This script helps deploy the application to your S3DF Kubernetes cluster

set -e

echo "================================================"
echo "    OuRAGboros S3DF Deployment Helper"
echo "================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}✗ kubectl not found${NC}"
        echo "  Install kubectl: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    else
        echo -e "${GREEN}✓ kubectl found${NC}"
    fi
    
    # Check current context
    CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "none")
    if [ "$CURRENT_CONTEXT" == "none" ]; then
        echo -e "${YELLOW}⚠ No kubectl context set${NC}"
        echo ""
        echo "Please configure S3DF access first:"
        echo "1. Go to https://k8s.slac.stanford.edu/<your-cluster-name>"
        echo "2. Log in with SLAC credentials"
        echo "3. Copy and run the provided kubectl config commands"
        echo ""
        read -p "Have you configured S3DF kubectl access? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Please configure S3DF access and run this script again."
            exit 1
        fi
    else
        echo -e "${GREEN}✓ kubectl context: $CURRENT_CONTEXT${NC}"
    fi
    
    # Test cluster connectivity
    echo "Testing cluster connection..."
    if kubectl get nodes &> /dev/null; then
        echo -e "${GREEN}✓ Connected to cluster${NC}"
    else
        echo -e "${RED}✗ Cannot connect to cluster${NC}"
        echo "Please check your kubectl configuration"
        exit 1
    fi
}

# Function to check GPU availability
check_gpu_resources() {
    echo ""
    echo "Checking GPU resources in cluster..."
    
    # Check if GPU nodes exist
    GPU_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.allocatable."nvidia.com/gpu" != null) | .metadata.name' 2>/dev/null || echo "")
    
    if [ -z "$GPU_NODES" ]; then
        echo -e "${YELLOW}⚠ No GPU nodes found in cluster${NC}"
        echo "The application requires GPUs. Proceeding anyway..."
    else
        echo -e "${GREEN}✓ GPU nodes available:${NC}"
        echo "$GPU_NODES"
    fi
}

# Main deployment function
deploy_application() {
    echo ""
    echo "Starting deployment..."
    
    # GPU configuration
    echo ""
    echo "GPU Configuration Options:"
    echo "  0 - No GPU (for testing only, models won't work properly)"
    echo "  1 - 1 GPU for Ollama, 1 GPU for OuRAGboros (recommended)"
    echo "  2 - 2 GPUs for Ollama, 1 GPU for OuRAGboros"
    echo ""
    read -p "Select GPU configuration (0/1/2, default: 1): " GPU_CONFIG
    GPU_CONFIG=${GPU_CONFIG:-1}
    
    case $GPU_CONFIG in
        0)
            echo "Using no GPU configuration (testing mode)"
            KUSTOMIZATION_FILE="k8s/overlays/development/kustomization-nogpu.yaml"
            ;;
        1)
            echo "Using 1 GPU for each service"
            KUSTOMIZATION_FILE="k8s/overlays/gpu-configs/kustomization-1gpu.yaml"
            ;;
        2)
            echo "Using 2 GPUs for Ollama, 1 for OuRAGboros"
            KUSTOMIZATION_FILE="k8s/overlays/gpu-configs/kustomization-2gpu.yaml"
            ;;
        *)
            echo "Invalid selection, using 1 GPU configuration"
            KUSTOMIZATION_FILE="k8s/overlays/gpu-configs/kustomization-1gpu.yaml"
            ;;
    esac
    
    # Copy the selected kustomization file
    echo ""
    echo "Applying GPU configuration..."
    cp $KUSTOMIZATION_FILE k8s/base/kustomization.yaml
    echo -e "${GREEN}✓ GPU configuration set${NC}"
    
    # Create namespace
    echo ""
    echo "Creating namespace 'ouragboros'..."
    kubectl create namespace ouragboros --dry-run=client -o yaml | kubectl apply -f -
    
    # Check if custom docker registry is needed
    echo ""
    read -p "Do you need to use a custom Docker registry? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your registry URL (e.g., registry.slac.stanford.edu/your-project): " REGISTRY
        echo ""
        echo -e "${YELLOW}Note: You'll need to update the image references in k8s.yaml:${NC}"
        echo "  Change: schrodingersket/ouragboros:0.0.1"
        echo "      To: $REGISTRY/ouragboros:0.0.1"
        echo ""
        echo "Build and push your image:"
        echo "  docker build -t $REGISTRY/ouragboros:0.0.1 ."
        echo "  docker push $REGISTRY/ouragboros:0.0.1"
        echo ""
        read -p "Have you updated k8s.yaml and pushed the image? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Please update the image references and run this script again."
            exit 1
        fi
    fi
    
    # Apply the configuration
    echo ""
    echo "Applying Kubernetes configuration..."
    kubectl apply --namespace ouragboros -k k8s/base
    
    echo ""
    echo -e "${GREEN}✓ Deployment initiated!${NC}"
}

# Function to check deployment status
check_deployment_status() {
    echo ""
    echo "Checking deployment status..."
    echo ""
    
    # Wait a bit for pods to start
    sleep 5
    
    # Check pods
    echo "Pods in ouragboros namespace:"
    kubectl get pods -n ouragboros
    
    echo ""
    echo "Services:"
    kubectl get svc -n ouragboros
    
    echo ""
    echo "Persistent Volume Claims:"
    kubectl get pvc -n ouragboros
    
    # Check for issues
    echo ""
    NOT_RUNNING=$(kubectl get pods -n ouragboros --field-selector=status.phase!=Running -o name 2>/dev/null | wc -l)
    if [ "$NOT_RUNNING" -gt 0 ]; then
        echo -e "${YELLOW}⚠ Some pods are not running yet${NC}"
        echo "This is normal during initial deployment. Pods may take a few minutes to start."
        echo ""
        echo "To monitor pod startup:"
        echo "  kubectl get pods -n ouragboros -w"
        echo ""
        echo "To check pod logs if there are issues:"
        echo "  kubectl logs -n ouragboros deployment/ouragboros"
        echo "  kubectl logs -n ouragboros deployment/ollama"
        echo "  kubectl logs -n ouragboros deployment/opensearch"
    else
        echo -e "${GREEN}✓ All pods are running!${NC}"
    fi
}

# Function to set up access
setup_access() {
    echo ""
    echo "================================================"
    echo "    How to Access Your Application"
    echo "================================================"
    echo ""
    echo "Option 1: Port Forwarding (Easiest for testing)"
    echo "------------------------------------------------"
    echo "Run this command to access the application:"
    echo ""
    echo -e "${GREEN}kubectl port-forward -n ouragboros svc/ouragboros 8501:8501${NC}"
    echo ""
    echo "Then open http://localhost:8501 in your browser"
    echo ""
    echo "Option 2: Ingress (For production use)"
    echo "---------------------------------------"
    echo "If your cluster has an ingress controller, check:"
    echo "  kubectl get ingress -n ouragboros"
    echo ""
    echo "Option 3: LoadBalancer Service"
    echo "-------------------------------"
    echo "Check if LoadBalancer IP is assigned:"
    echo "  kubectl get svc ouragboros -n ouragboros"
}

# Function to show post-deployment steps
post_deployment_steps() {
    echo ""
    echo "================================================"
    echo "    Post-Deployment Steps"
    echo "================================================"
    echo ""
    echo "1. Pull Ollama models (after pods are running):"
    echo "   kubectl exec -n ouragboros deployment/ollama -- ollama pull llama3.1"
    echo ""
    echo "2. Monitor logs:"
    echo "   kubectl logs -n ouragboros -f deployment/ouragboros"
    echo ""
    echo "3. Check GPU allocation:"
    echo "   kubectl describe pod -n ouragboros -l io.kompose.service=ollama | grep nvidia"
    echo ""
    echo "4. Scale deployments if needed:"
    echo "   kubectl scale deployment/ouragboros -n ouragboros --replicas=2"
}

# Main execution
main() {
    check_prerequisites
    check_gpu_resources
    
    echo ""
    read -p "Ready to deploy? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_application
        check_deployment_status
        setup_access
        post_deployment_steps
        
        echo ""
        echo "================================================"
        echo -e "${GREEN}    Deployment Complete!${NC}"
        echo "================================================"
        echo ""
        echo "Next steps:"
        echo "1. Wait for all pods to be running"
        echo "2. Set up port forwarding"
        echo "3. Access the application"
        echo ""
        echo "Need help? Check pod status and logs:"
        echo "  kubectl get pods -n ouragboros"
        echo "  kubectl describe pod -n ouragboros <pod-name>"
    else
        echo "Deployment cancelled."
    fi
}

# Run main function
main