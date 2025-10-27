#!/bin/bash
set -e

# Configuration
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-iyassou}"
IMAGE_NAME="ssregpros"

# Get current git commit SHA
GIT_SHA=$(git rev-parse HEAD 2> /dev/null || echo "")

# Build image.
echo "Building Docker image..."
echo "Git commit: $GIT_SHA"
docker build --build-arg SSREGPROS_GIT_COMMIT_SHA=$GIT_SHA -t $IMAGE_NAME:latest .

# Tag for Docker Hub
docker tag $IMAGE_NAME:latest $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# Push to Docker Hub.
docker login
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# Clean up.
echo ""
echo "Cleaning up dangling images..."
docker image prune -a -f

# Done!
echo ""
echo "✓ Docker image '$IMAGE_NAME:latest' built successfully!"
echo ""
echo "To pull on the HCL cluster:"
echo "  module load apptainer"
echo "  apptainer pull ssregpros.sif docker://$DOCKERHUB_USERNAME/$IMAGE_NAME:latest"