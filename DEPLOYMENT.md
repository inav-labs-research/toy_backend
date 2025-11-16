# Deployment Guide for toy_backend

## Server Setup Steps

### Step 1: Free Up Disk Space

Before building, clean up Docker to free space:

```bash
# Check disk space
df -h

# Clean up Docker (removes unused images, containers, volumes, and build cache)
sudo docker system prune -a --volumes -f

# Check Docker disk usage
sudo docker system df

# If still low on space, remove specific large images
sudo docker images
sudo docker rmi <image-id>
```

### Step 2: Build the Docker Image

```bash
cd ~/toy_backend

# Build the image (this will take several minutes)
sudo docker build -t toy-backend:latest .
```

**Note:** The build uses CPU-only PyTorch to save ~2GB of disk space. This is sufficient for VAD (Voice Activity Detection).

### Step 3: Run the Container

```bash
# Stop and remove existing container if it exists
sudo docker stop toy-backend 2>/dev/null
sudo docker rm toy-backend 2>/dev/null

# Run the container
sudo docker run -d \
  --name toy-backend \
  -p 5050:5050 \
  --restart unless-stopped \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/agents.json:/app/agents.json:ro \
  -v $(pwd)/logs:/app/logs \
  toy-backend:latest
```

### Step 4: Verify Deployment

```bash
# Check container status
sudo docker ps | grep toy-backend

# View logs
sudo docker logs -f toy-backend

# Test health endpoint
curl http://localhost:5050/health

# Test root endpoint
curl http://localhost:5050/
```

## Management Commands

### View Logs
```bash
sudo docker logs -f toy-backend
```

### Restart Container
```bash
sudo docker restart toy-backend
```

### Stop Container
```bash
sudo docker stop toy-backend
```

### Start Container
```bash
sudo docker start toy-backend
```

### Update and Redeploy
```bash
cd ~/toy_backend

# Pull latest code (if using git)
git pull

# Rebuild
sudo docker stop toy-backend
sudo docker rm toy-backend
sudo docker build -t toy-backend:latest .
sudo docker run -d --name toy-backend -p 5050:5050 --restart unless-stopped \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/agents.json:/app/agents.json:ro \
  -v $(pwd)/logs:/app/logs \
  toy-backend:latest
```

## Troubleshooting

### Out of Disk Space

If you get "No space left on device" error:

```bash
# 1. Check disk usage
df -h

# 2. Clean Docker
sudo docker system prune -a --volumes -f

# 3. Remove old images
sudo docker images | grep -v "toy-backend" | awk 'NR>1 {print $3}' | xargs sudo docker rmi

# 4. Clean build cache
sudo docker builder prune -a -f

# 5. Check what's using space
sudo du -sh /var/lib/docker/*
```

### Container Won't Start

```bash
# Check logs
sudo docker logs toy-backend

# Check if port is in use
sudo lsof -i :5050
```

### Permission Issues

```bash
# Add user to docker group (optional, to avoid sudo)
sudo usermod -aG docker $USER
# Then logout and login again
```

## Firewall Configuration

If you need to expose the service externally:

```bash
# For Amazon Linux / CentOS
sudo firewall-cmd --permanent --add-port=5050/tcp
sudo firewall-cmd --reload

# OR for EC2 Security Groups, add inbound rule:
# Type: Custom TCP, Port: 5050, Source: 0.0.0.0/0 (or your IP)
```

