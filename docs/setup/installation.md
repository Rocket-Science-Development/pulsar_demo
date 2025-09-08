# Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Disk Space**: At least 2GB free space
- **Network**: Internet connection for downloading Docker images

### Required Software

#### Docker & Docker Compose
**Minimum Versions**:
- Docker: 20.10.0 or higher
- Docker Compose: 2.0.0 or higher

**Installation Instructions**:

**Linux (Ubuntu/Debian)**:
```bash
# Update package index
sudo apt update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for group changes to take effect
```

**macOS**:
```bash
# Install Docker Desktop for Mac from https://docs.docker.com/desktop/mac/install/
# Docker Compose is included with Docker Desktop
```

**Windows**:
```bash
# Install Docker Desktop for Windows from https://docs.docker.com/desktop/windows/install/
# Enable WSL2 backend
# Docker Compose is included with Docker Desktop
```

#### Python (Optional - for local development)
- **Version**: Python 3.9 or higher
- **Package Manager**: pip or Poetry

```bash
# Check Python version
python3 --version

# Install Poetry (optional)
curl -sSL https://install.python-poetry.org | python3 -
```

## Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rocket-Science-Development/pulsar_demo.git
cd pulsar_demo
```

### 2. Environment Configuration

#### Option A: Using Environment File (Recommended)
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your preferred settings
nano .env  # or use your preferred editor
```

#### Option B: Export Environment Variables
```bash
export GRAFANA_USERNAME=admin
export GRAFANA_PASSWORD=pass123
export DB_USER=admin
export DB_PASSWORD=pass123
export DB_HOST=influxdb
export DB_PORT=8086
```

### 3. Build and Start Services
```bash
# Build and start all services
docker-compose up --build

# Or run in background (detached mode)
docker-compose up --build -d
```

### 4. Verification Steps

#### Check Service Status
```bash
# Check all containers are running
docker-compose ps

# Expected output should show all services as "Up"
# - influxdb
# - grafana
# - chronograf
# - fake-data-generator
# - compute-metrics
```

#### Verify Service Accessibility
```bash
# Test InfluxDB connection
curl -i http://localhost:8086/ping

# Test Grafana accessibility
curl -i http://localhost:3000/api/health

# Test Chronograf accessibility
curl -i http://localhost:8889
```

#### Check Logs for Errors
```bash
# Check all service logs
docker-compose logs

# Check specific service logs
docker-compose logs grafana
docker-compose logs influxdb
docker-compose logs fake-data-generator
docker-compose logs compute-metrics
```

### 5. Access the Dashboard

1. **Open Grafana**: Navigate to http://localhost:3000
2. **Login**: Use credentials from your `.env` file or exported variables
   - Default: Username: `admin`, Password: `pass123`
3. **Verify Data**: Check that dashboards are loading with data

### 6. Additional Service Access

- **Chronograf (InfluxDB UI)**: http://localhost:8889
- **InfluxDB API**: http://localhost:8086

## Post-Installation Configuration

### Database Initialization
The system automatically creates necessary databases and indexes:
- `prediction` - Stores model predictions and metadata
- `metrics` - Stores computed metrics

### Data Flow Verification
1. Check that fake data is being generated every 2 minutes
2. Verify metrics are computed every 4 minutes
3. Confirm data appears in Grafana dashboards

## Local Development Setup (Optional)

### Install Python Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Using Poetry
poetry install
poetry shell
```

### Run Scripts Locally
```bash
# Generate fake data locally
python src/fake_data_script.py

# Compute metrics locally (requires InfluxDB connection)
python src/compute_metrics_script.py

# Train model
python src/app.py
```

## Uninstallation

### Stop and Remove Services
```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: This deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

### Clean Up Docker Resources
```bash
# Remove unused Docker resources
docker system prune -a --volumes
```

## Next Steps

After successful installation:
1. Review [Architecture Documentation](../architecture/overview.md)
2. Check [Security Guidelines](../security/guidelines.md)
3. Explore [Development Guide](../development/getting-started.md)
4. If you encounter issues, see [Troubleshooting Guide](../troubleshooting/common-issues.md)