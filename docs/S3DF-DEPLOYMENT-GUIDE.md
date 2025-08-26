# S3DF Kubernetes Deployment Guide for OuRAGboros

## Quick Start

1. **Run the deployment script:**
   ```bash
   ./deploy-to-s3df.sh
   ```
   This script will guide you through the entire deployment process.

2. **Access the application:**
   ```bash
   kubectl port-forward -n ouragboros svc/ouragboros 8501:8501
   ```
   Then open http://localhost:8501 in your browser.

## Detailed Explanation for Beginners

### What is Kubernetes (K8s)?
Kubernetes is like a smart manager for your applications. Instead of manually starting programs on servers, you tell Kubernetes what you want to run, and it handles:
- Starting your applications (in "containers")
- Keeping them running (restarting if they crash)
- Managing storage and networking
- Scaling up/down based on demand

### Key Concepts

1. **Pod**: The smallest unit in K8s, containing one or more containers
2. **Deployment**: Manages pods, ensuring desired number are running
3. **Service**: Provides networking, allowing pods to communicate
4. **Namespace**: Logical separation of resources (like folders)
5. **PersistentVolumeClaim (PVC)**: Request for storage space

### Your OuRAGboros Architecture

```
Your Laptop                     S3DF vCluster
     |                               |
     |  kubectl commands             |
     |------------------------------>|
     |                               |
     |                          [Namespace: ouragboros]
     |                               |
     |                          3 Deployments (Pods):
     |                          - ollama (LLM engine)
     |                          - opensearch (Vector DB)
     |                          - ouragboros (Web UI)
     |                               |
     |  Port-forward                 |
     |<------------------------------|
     |                               |
  Browser                            |
  http://localhost:8501              |
```

## Step-by-Step Manual Deployment

### Step 1: Configure kubectl

First, get your S3DF cluster credentials:

1. Open your browser and go to: `https://k8s.slac.stanford.edu/<your-cluster-name>`
2. Log in with your SLAC credentials
3. Copy the kubectl commands shown (they look like this):
   ```bash
   kubectl config set-cluster "cluster-name" --server=https://...
   kubectl config set-credentials "user@slac.stanford.edu@cluster" ...
   kubectl config set-context "cluster-name" ...
   kubectl config use-context "cluster-name"
   ```
4. Run these commands on your laptop

Verify connection:
```bash
kubectl get nodes
```

### Step 2: Create Namespace

A namespace isolates your application from others:

```bash
kubectl create namespace ouragboros
```

### Step 3: Deploy the Application

Set GPU allocation (adjust based on availability):
```bash
export NVIDIA_GPUS=1
```

Apply all configurations:
```bash
kubectl apply --namespace ouragboros -k .
```

This command:
- Reads `kustomization.yaml` (which references `k8s.yaml`)
- Applies GPU patches
- Creates all resources in the cluster

### Step 4: Monitor Deployment

Watch pods starting up:
```bash
kubectl get pods -n ouragboros -w
```

Expected output (after a few minutes):
```
NAME                          READY   STATUS    RESTARTS   AGE
ollama-xxxxx-xxxxx           1/1     Running   0          2m
opensearch-xxxxx-xxxxx       1/1     Running   0          2m
ouragboros-xxxxx-xxxxx       1/1     Running   0          2m
```

### Step 5: Access the Application

For testing (easiest method):
```bash
kubectl port-forward -n ouragboros svc/ouragboros 8501:8501
```

This creates a tunnel from your laptop's port 8501 to the application in the cluster.

## Common Issues and Solutions

### Issue 1: Pods stuck in "Pending" state
**Cause**: Usually insufficient resources (CPU, Memory, GPU)

**Check**:
```bash
kubectl describe pod -n ouragboros <pod-name>
```

Look for "Events" section showing scheduling issues.

**Solution**: 
- Reduce GPU request in kustomization.yaml
- Check if cluster has GPU nodes: `kubectl get nodes -L nvidia.com/gpu`

### Issue 2: Pods in "CrashLoopBackOff"
**Cause**: Application failing to start

**Check logs**:
```bash
kubectl logs -n ouragboros <pod-name>
```

**Common fixes**:
- Image pull issues: Check image name/tag in k8s.yaml
- Missing environment variables: Check ConfigMap
- Storage issues: Verify PVCs are bound

### Issue 3: Cannot connect after port-forward
**Check**:
- Port-forward is still running
- No other process using port 8501: `lsof -i :8501`
- Pod is actually running: `kubectl get pod -n ouragboros`

### Issue 4: GPU not allocated
**Check GPU assignment**:
```bash
kubectl describe pod -n ouragboros -l io.kompose.service=ollama | grep -A 5 "Requests"
```

**Fix**: Ensure `NVIDIA_GPUS` was set before applying kustomization

## Useful Commands

### Viewing Resources
```bash
# All resources in namespace
kubectl get all -n ouragboros

# Detailed pod information
kubectl describe pod -n ouragboros <pod-name>

# Pod logs
kubectl logs -n ouragboros <pod-name>

# Follow logs in real-time
kubectl logs -n ouragboros <pod-name> -f
```

### Debugging
```bash
# Execute commands in pod
kubectl exec -it -n ouragboros <pod-name> -- /bin/bash

# Check resource usage
kubectl top pods -n ouragboros

# Get events
kubectl get events -n ouragboros --sort-by='.lastTimestamp'
```

### Management
```bash
# Restart a deployment
kubectl rollout restart deployment/ouragboros -n ouragboros

# Scale deployment
kubectl scale deployment/ouragboros -n ouragboros --replicas=2

# Delete and recreate everything
kubectl delete namespace ouragboros
# Then redeploy from Step 2
```

## Loading Ollama Models

After Ollama pod is running:
```bash
# Pull the default model
kubectl exec -n ouragboros deployment/ollama -- ollama pull llama3.1

# Pull additional models
kubectl exec -n ouragboros deployment/ollama -- ollama pull codellama
kubectl exec -n ouragboros deployment/ollama -- ollama pull mistral
```

## Production Considerations

### Using a Custom Docker Registry

S3DF might require using their internal registry:

1. Build your image:
   ```bash
   docker build -t registry.slac.stanford.edu/your-project/ouragboros:0.0.1 .
   ```

2. Push to registry:
   ```bash
   docker push registry.slac.stanford.edu/your-project/ouragboros:0.0.1
   ```

3. Update `k8s.yaml`:
   ```yaml
   image: registry.slac.stanford.edu/your-project/ouragboros:0.0.1
   ```

### Setting up Ingress

For permanent URL access, configure ingress:

1. Check if ingress controller exists:
   ```bash
   kubectl get ingressclass
   ```

2. Update ingress in k8s.yaml with your domain:
   ```yaml
   spec:
     rules:
     - host: ouragboros.your-cluster.slac.stanford.edu
       http:
         paths:
         - path: /
           pathType: Prefix
           backend:
             service:
               name: ouragboros
               port:
                 number: 8501
   ```

### Persistent Storage

The application uses PVCs for storage. If pods restart, data persists. Check storage:
```bash
kubectl get pvc -n ouragboros
```

All PVCs should show "Bound" status.

## Getting Help

1. **Check pod logs first**: Most issues are visible in logs
2. **S3DF Documentation**: https://s3df.slac.stanford.edu/
3. **Kubernetes Documentation**: https://kubernetes.io/docs/
4. **Ask for help**: Include output of:
   - `kubectl get pods -n ouragboros`
   - `kubectl describe pod -n ouragboros <failing-pod>`
   - `kubectl logs -n ouragboros <failing-pod>`

## Clean Up

To remove everything:
```bash
kubectl delete namespace ouragboros
```

This deletes all pods, services, and storage in the namespace.