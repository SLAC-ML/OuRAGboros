#!/bin/bash

# Quick fix for deployment without GPU requirements initially
# This allows you to deploy and then add GPU resources later

echo "Quick Deployment Fix for OuRAGboros"
echo "===================================="
echo ""
echo "This script deploys without GPU patches first to avoid the error."
echo ""

# Check kubectl connection
echo "Checking cluster connection..."
if kubectl get nodes &> /dev/null; then
    echo "✓ Connected to cluster"
else
    echo "✗ Cannot connect to cluster"
    echo "Please configure kubectl for S3DF first"
    exit 1
fi

# Create namespace
echo ""
echo "Creating namespace..."
kubectl create namespace ouragboros --dry-run=client -o yaml | kubectl apply -f -

# Deploy without GPU patches (using k8s.yaml directly)
echo ""
echo "Deploying application (without GPU requirements initially)..."
kubectl apply -n ouragboros -f k8s.yaml

echo ""
echo "Deployment started!"
echo ""
echo "Check pod status:"
echo "  kubectl get pods -n ouragboros"
echo ""
echo "Note: Pods may be pending due to GPU requirements."
echo "To add GPU resources later, use kubectl edit deployment."
echo ""
echo "Access the app (after pods are running):"
echo "  kubectl port-forward -n ouragboros svc/ouragboros 8501:8501"