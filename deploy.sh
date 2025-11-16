#!/bin/bash
# Deployment script for toy_backend on server

set -e  # Exit on error

echo "=== Toy Backend Deployment Script ==="
echo ""

# Step 1: Clean up Docker to free space
echo "Step 1: Cleaning up Docker to free disk space..."
sudo docker system prune -a --volumes -f
echo "✓ Docker cleanup complete"
echo ""

# Step 2: Check disk space
echo "Step 2: Checking disk space..."
df -h / | tail -1
echo ""

# Step 3: Build Docker image
echo "Step 3: Building Docker image (this may take several minutes)..."
cd "$(dirname "$0")"
sudo docker build -t toy-backend:latest .
echo "✓ Docker image built successfully"
echo ""

# Step 4: Stop and remove existing container
echo "Step 4: Stopping existing container (if any)..."
sudo docker stop toy-backend 2>/dev/null || true
sudo docker rm toy-backend 2>/dev/null || true
echo "✓ Old container removed"
echo ""

# Step 5: Run new container
echo "Step 5: Starting new container..."
sudo docker run -d \
  --name toy-backend \
  -p 5050:5050 \
  --restart unless-stopped \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/agents.json:/app/agents.json:ro \
  -v $(pwd)/logs:/app/logs \
  toy-backend:latest
echo "✓ Container started"
echo ""

# Step 6: Wait for startup
echo "Step 6: Waiting for server to start..."
sleep 8

# Step 7: Check status
echo "Step 7: Checking container status..."
sudo docker ps | grep toy-backend || echo "⚠ Container not found in running list"
echo ""

# Step 8: Test endpoints
echo "Step 8: Testing endpoints..."
if curl -s http://localhost:5050/health > /dev/null; then
    echo "✓ Health endpoint: OK"
    curl -s http://localhost:5050/health
else
    echo "⚠ Health endpoint: Not responding yet"
    echo "View logs with: sudo docker logs -f toy-backend"
fi
echo ""

echo "=== Deployment Complete ==="
echo ""
echo "Useful commands:"
echo "  View logs:    sudo docker logs -f toy-backend"
echo "  Restart:      sudo docker restart toy-backend"
echo "  Stop:         sudo docker stop toy-backend"
echo "  Status:       sudo docker ps | grep toy-backend"
