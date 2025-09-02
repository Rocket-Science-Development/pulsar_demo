# Troubleshooting Guide

## Common Issues and Solutions

### Docker and Container Issues

#### Issue: Containers fail to start
**Symptoms**:
- `docker-compose up` fails with container exit codes
- Services show as "Exited" in `docker-compose ps`

**Solutions**:
```bash
# Check detailed error logs
docker-compose logs [service-name]

# Common fixes:
# 1. Ensure Docker is running
sudo systemctl status docker  # Linux
# Docker Desktop should be running on macOS/Windows

# 2. Check port conflicts
sudo netstat -tlnp | grep :3000  # Check if Grafana port is in use
sudo netstat -tlnp | grep :8086  # Check if InfluxDB port is in use
sudo netstat -tlnp | grep :8889  # Check if Chronograf port is in use

# 3. Clean up previous containers
docker-compose down -v
docker system prune -f
```

#### Issue: Permission denied errors
**Symptoms**:
- "Permission denied" errors in logs
- Volume mount failures

**Solutions**:
```bash
# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
# Logout and login again

# Fix file permissions
sudo chown -R $USER:$USER .
chmod -R 755 .

# SELinux issues (RHEL/CentOS)
sudo setsebool -P container_manage_cgroup on
```

### Environment Variable Issues

#### Issue: Services can't connect to each other
**Symptoms**:
- "Connection refused" errors
- Database connection failures
- Authentication errors

**Solutions**:
```bash
# 1. Verify environment variables
echo $GRAFANA_USERNAME
echo $DB_USER

# 2. Check .env file exists and is properly formatted
cat .env
# Ensure no spaces around = signs
# Example: DB_USER=admin (not DB_USER = admin)

# 3. Restart with clean environment
docker-compose down
docker-compose up --build
```

#### Issue: Wrong credentials in services
**Symptoms**:
- Login failures in Grafana
- Database authentication errors

**Solutions**:
```bash
# Reset to default credentials
export GRAFANA_USERNAME=admin
export GRAFANA_PASSWORD=pass123
export DB_USER=admin
export DB_PASSWORD=pass123

# Or edit .env file
nano .env
```

### Database Connection Issues

#### Issue: InfluxDB connection failures
**Symptoms**:
- Metrics computation script fails
- "Connection refused" or "Connection timeout" errors

**Diagnostic Commands**:
```bash
# Test InfluxDB connectivity
curl -i http://localhost:8086/ping

# Check InfluxDB logs
docker-compose logs influxdb

# Check if database is ready
docker exec -it $(docker-compose ps -q influxdb) influx -execute "SHOW DATABASES"
```

**Solutions**:
```bash
# 1. Wait for InfluxDB to fully initialize (can take 30-60 seconds)
# 2. Check InfluxDB container health
docker-compose ps influxdb

# 3. Restart InfluxDB service
docker-compose restart influxdb
```

#### Issue: No data in Grafana dashboards
**Symptoms**:
- Grafana loads but shows empty dashboards
- "No data points" messages

**Diagnostic Steps**:
```bash
# 1. Check if fake data generator is running
docker-compose logs fake-data-generator

# 2. Check metrics computation service
docker-compose logs compute-metrics

# 3. Verify data in InfluxDB
docker exec -it $(docker-compose ps -q influxdb) influx -execute "USE prediction; SHOW MEASUREMENTS"
docker exec -it $(docker-compose ps -q influxdb) influx -execute "USE metrics; SHOW MEASUREMENTS"
```

**Solutions**:
```bash
# 1. Restart data generation services
docker-compose restart fake-data-generator compute-metrics

# 2. Check cron job timing (services run every 2-4 minutes)
# Wait 5 minutes and check again

# 3. Verify database indexes exist
docker exec -it $(docker-compose ps -q influxdb) influx -execute "SHOW DATABASES"
```

### Network and Port Issues

#### Issue: Cannot access services via browser
**Symptoms**:
- Browser shows "Connection refused"
- Timeout errors when accessing localhost URLs

**Solutions**:
```bash
# 1. Check if ports are bound correctly
docker-compose ps
# Verify port mappings show 0.0.0.0:3000->3000/tcp

# 2. Test local connectivity
telnet localhost 3000  # Should connect to Grafana
telnet localhost 8086  # Should connect to InfluxDB

# 3. Check firewall settings
# Linux
sudo ufw status
sudo ufw allow 3000
sudo ufw allow 8086
sudo ufw allow 8889

# 4. Try alternative access methods
# Use Docker container IP directly
docker inspect $(docker-compose ps -q grafana) | grep IPAddress
```

### Performance Issues

#### Issue: Slow dashboard loading
**Symptoms**:
- Grafana dashboards take long time to load
- Timeouts in metric queries

**Solutions**:
```bash
# 1. Check system resources
docker stats  # Monitor CPU and memory usage

# 2. Increase Docker resource limits
# Docker Desktop: Settings > Resources > Advanced
# Recommended: 4GB RAM, 2 CPUs minimum

# 3. Check InfluxDB performance
docker exec -it $(docker-compose ps -q influxdb) influx -execute "SHOW QUERIES"
```

### Log File Locations and Interpretation

#### Container Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs grafana
docker-compose logs influxdb
docker-compose logs fake-data-generator
docker-compose logs compute-metrics

# Follow logs in real-time
docker-compose logs -f

# View last N lines
docker-compose logs --tail=100 grafana
```

#### Log Interpretation Guidelines

**InfluxDB Logs**:
- `[I] InfluxDB starting` - Normal startup
- `[E] unable to open database` - Database corruption or permission issue
- `[W] retention policy` - Configuration warnings (usually safe to ignore)

**Grafana Logs**:
- `HTTP Server Listen` - Normal startup
- `Database migration` - Initial setup (normal on first run)
- `Failed to connect to database` - Database connection issue

**Application Logs**:
- `Traceback` in Python services - Check environment variables and dependencies
- `Connection refused` - Network connectivity issue
- `Authentication failed` - Credential configuration problem

### Service Status Checking

#### Health Check Commands
```bash
# Overall system status
docker-compose ps

# Detailed container information
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Network connectivity between containers
docker network ls
docker network inspect pulsar_demo_default
```

#### Service-Specific Health Checks
```bash
# InfluxDB health
curl -f http://localhost:8086/ping || echo "InfluxDB not responding"

# Grafana health
curl -f http://localhost:3000/api/health || echo "Grafana not responding"

# Check if data is flowing
docker exec $(docker-compose ps -q influxdb) influx -execute "USE prediction; SELECT COUNT(*) FROM pokemon_predictions" 2>/dev/null || echo "No prediction data"
```

### Recovery Procedures

#### Complete System Reset
```bash
# Stop all services
docker-compose down -v

# Remove all related Docker resources
docker system prune -a --volumes --filter label=com.docker.compose.project=pulsar_demo

# Clean start
docker-compose up --build
```

#### Partial Service Recovery
```bash
# Restart specific service
docker-compose restart [service-name]

# Rebuild specific service
docker-compose up --build [service-name]

# Reset service data only
docker-compose down
docker volume rm pulsar_demo_influxdb_data  # Removes InfluxDB data only
docker-compose up
```

## Getting Additional Help

### Diagnostic Information to Collect
When reporting issues, include:

```bash
# System information
docker --version
docker-compose --version
uname -a

# Service status
docker-compose ps
docker-compose logs --tail=50

# Environment variables (sanitized)
env | grep -E "(GRAFANA|DB_)" | sed 's/PASSWORD=.*/PASSWORD=***/'

# Network information
docker network inspect pulsar_demo_default
```

### Support Channels
- **GitHub Issues**: [Pulsar Demo Repository](https://github.com/Rocket-Science-Development/pulsar_demo/issues)
- **Slack Community**: [PulsarML Slack](https://pulsarml.slack.com)
- **Documentation**: Check other files in `docs/` folder

### Known Limitations
- Initial startup can take 2-3 minutes for all services to be ready
- First data generation occurs after 2 minutes
- Dashboard data appears after first metrics computation (4 minutes)
- Windows Docker Desktop may require WSL2 backend for optimal performance