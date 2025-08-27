#!/bin/bash
# Build and push Docker image with date-based tagging
# Usage: ./scripts/build-and-push.sh [suffix]
#
# Examples:
#   ./scripts/build-and-push.sh       # Creates slacml/ouragboros:25.08.26
#   ./scripts/build-and-push.sh 2     # Creates slacml/ouragboros:25.08.26-2

set -e

# Get today's date in YY.MM.DD format
DATE_TAG=$(date +%y.%m.%d)

# Add suffix if provided
if [ -n "$1" ]; then
    IMAGE_TAG="${DATE_TAG}-$1"
else
    IMAGE_TAG="${DATE_TAG}"
fi

IMAGE_NAME="slacml/ouragboros:${IMAGE_TAG}"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "Pushing Docker image: ${IMAGE_NAME}"
docker push "${IMAGE_NAME}"

echo ""
echo "✅ Successfully built and pushed: ${IMAGE_NAME}"
echo ""
echo "To deploy to Kubernetes, update k8s/base/k8s.yaml with:"
echo "  image: ${IMAGE_NAME}"
echo ""
echo "Then apply with:"
echo "  kubectl apply -n ouragboros -k k8s/base"