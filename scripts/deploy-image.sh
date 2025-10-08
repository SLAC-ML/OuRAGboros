#!/bin/bash

# OuRAGboros Kubernetes Deployment Automation Script
# Automatically updates k8s deployment with a new image tag and deploys to cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
K8S_FILE="k8s/base/k8s.yaml"
NAMESPACE="ouragboros"
DEPLOYMENT_NAME="ouragboros"
IMAGE_REPO="slacml/ouragboros"

# Help function
show_help() {
    echo -e "${BLUE}OuRAGboros Kubernetes Deployment Script${NC}"
    echo ""
    echo "Usage: $0 <image-tag> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  <image-tag>     The Docker image tag to deploy (e.g., 25.10.07-2)"
    echo ""
    echo "Options:"
    echo "  --skip-verify   Skip image existence verification on Docker Hub"
    echo "  --no-restart    Update config but don't restart deployment"
    echo "  --dry-run       Show what would be changed without applying"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 25.10.07-2                  # Deploy image with tag 25.10.07-2"
    echo "  $0 latest --skip-verify        # Deploy latest without checking Hub"
    echo "  $0 25.10.07-2 --dry-run        # Preview changes without applying"
    echo ""
    echo "What this script does:"
    echo "  1. Verifies image exists on Docker Hub (unless --skip-verify)"
    echo "  2. Updates ${K8S_FILE} with new image tag"
    echo "  3. Applies changes to namespace ${NAMESPACE}"
    echo "  4. Restarts deployment to pull new image"
    echo "  5. Waits for rollout to complete"
    echo "  6. Verifies new image is running"
}

# Parse arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Image tag required${NC}"
    echo ""
    show_help
    exit 1
fi

IMAGE_TAG="$1"
shift

SKIP_VERIFY=false
NO_RESTART=false
DRY_RUN=false

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --no-restart)
            NO_RESTART=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

FULL_IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  OuRAGboros Kubernetes Deployment Automation          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Image:${NC}      ${FULL_IMAGE}"
echo -e "${BLUE}Namespace:${NC}  ${NAMESPACE}"
echo -e "${BLUE}K8s File:${NC}   ${K8S_FILE}"
echo ""

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl installed${NC}"

if [ ! -f "$K8S_FILE" ]; then
    echo -e "${RED}✗ K8s file not found: $K8S_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ K8s file found${NC}"

# Check kubectl context
CURRENT_CONTEXT=$(kubectl config current-context)
echo -e "${GREEN}✓ kubectl context: ${CURRENT_CONTEXT}${NC}"

# Verify namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${RED}✗ Namespace '${NAMESPACE}' not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Namespace '${NAMESPACE}' exists${NC}"
echo ""

# Verify image exists on Docker Hub (unless skipped)
if [ "$SKIP_VERIFY" = false ]; then
    echo -e "${BLUE}🔍 Verifying image exists on Docker Hub...${NC}"

    # Use docker manifest inspect to check if image exists
    if docker manifest inspect "${FULL_IMAGE}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Image ${FULL_IMAGE} found on Docker Hub${NC}"
    else
        echo -e "${YELLOW}⚠️  Warning: Could not verify image on Docker Hub${NC}"
        echo -e "${YELLOW}   This could mean:${NC}"
        echo -e "${YELLOW}   - Image hasn't been pushed yet${NC}"
        echo -e "${YELLOW}   - You don't have permission to access it${NC}"
        echo -e "${YELLOW}   - Network issues${NC}"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Deployment cancelled${NC}"
            exit 1
        fi
    fi
    echo ""
fi

# Get current image tag
CURRENT_IMAGE=$(grep -A 1 "image: ${IMAGE_REPO}:" "$K8S_FILE" | grep "image:" | awk '{print $2}')
echo -e "${BLUE}📝 Current image:${NC} ${CURRENT_IMAGE}"
echo -e "${BLUE}📝 New image:${NC}     ${FULL_IMAGE}"
echo ""

if [ "$CURRENT_IMAGE" = "$FULL_IMAGE" ]; then
    echo -e "${YELLOW}⚠️  Image tag is already set to ${IMAGE_TAG}${NC}"
    read -p "Continue with deployment anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deployment cancelled${NC}"
        exit 0
    fi
fi

# Dry run mode - show what would change
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 DRY RUN MODE - No changes will be applied${NC}"
    echo ""
    echo -e "${BLUE}Would update:${NC}"
    echo "  ${CURRENT_IMAGE} → ${FULL_IMAGE}"
    echo ""
    exit 0
fi

# Update k8s file
echo -e "${BLUE}📝 Updating ${K8S_FILE}...${NC}"

# Create backup
cp "$K8S_FILE" "${K8S_FILE}.bak"
echo -e "${GREEN}✓ Created backup: ${K8S_FILE}.bak${NC}"

# Update image tag
sed -i.tmp "s|image: ${IMAGE_REPO}:.*|image: ${FULL_IMAGE}|g" "$K8S_FILE"
rm -f "${K8S_FILE}.tmp"
echo -e "${GREEN}✓ Updated image tag in ${K8S_FILE}${NC}"
echo ""

# Apply changes to cluster
echo -e "${BLUE}🚀 Applying changes to cluster...${NC}"
kubectl apply -k k8s/base -n "$NAMESPACE"
echo -e "${GREEN}✓ Configuration applied${NC}"
echo ""

# Restart deployment (unless --no-restart)
if [ "$NO_RESTART" = false ]; then
    echo -e "${BLUE}🔄 Restarting deployment to pull new image...${NC}"
    kubectl rollout restart deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE"
    echo -e "${GREEN}✓ Deployment restart triggered${NC}"
    echo ""

    echo -e "${BLUE}⏳ Waiting for rollout to complete (timeout: 5 minutes)...${NC}"
    if kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=5m; then
        echo -e "${GREEN}✓ Rollout completed successfully${NC}"
    else
        echo -e "${RED}✗ Rollout failed or timed out${NC}"
        echo -e "${YELLOW}Check pod status with: kubectl get pods -n ${NAMESPACE}${NC}"
        exit 1
    fi
    echo ""

    # Verify new image is running
    echo -e "${BLUE}🔍 Verifying new image is running...${NC}"
    RUNNING_IMAGE=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')

    if [ "$RUNNING_IMAGE" = "$FULL_IMAGE" ]; then
        echo -e "${GREEN}✓ Deployment is running image: ${FULL_IMAGE}${NC}"
    else
        echo -e "${YELLOW}⚠️  Warning: Deployment image shows as: ${RUNNING_IMAGE}${NC}"
    fi

    # Show pod status
    echo ""
    echo -e "${BLUE}📊 Pod Status:${NC}"
    kubectl get pods -n "$NAMESPACE" -l io.kompose.service=ouragboros
else
    echo -e "${YELLOW}⚠️  Skipping deployment restart (--no-restart flag)${NC}"
    echo -e "${YELLOW}   Run this to restart manually:${NC}"
    echo -e "${YELLOW}   kubectl rollout restart deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Deployment Complete!                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo -e "  • Image deployed: ${GREEN}${FULL_IMAGE}${NC}"
echo -e "  • Namespace:      ${GREEN}${NAMESPACE}${NC}"
echo -e "  • Backup saved:   ${GREEN}${K8S_FILE}.bak${NC}"
echo ""
