# Security Guidelines

## Overview

This document outlines security best practices for deploying and managing the Pulsar.ML demo environment. While this is a demonstration system, following these guidelines ensures secure development practices and prepares for production deployments.

## Credential Management

### Environment Variables

**Best Practices**:
- Use `.env` file for local development
- Never commit credentials to version control
- Use strong, unique passwords for all services
- Rotate credentials regularly

**Required Environment Variables**:
```bash
# Database credentials
DB_USER=admin
DB_PASSWORD=your_secure_password_here

# Grafana credentials  
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=your_secure_grafana_password

# Optional: Custom database settings
DB_HOST=influxdb
DB_PORT=8086
```

**Password Requirements**:
- Minimum 12 characters
- Include uppercase, lowercase, numbers, and special characters
- Avoid dictionary words or common patterns
- Use different passwords for each service

**Example Strong Passwords**:
```bash
# Good examples (don't use these exactly)
DB_PASSWORD=MyS3cur3DB#2023!
GRAFANA_PASSWORD=Gr4f4n4@Admin$2023
```

### Production Credential Management

**For Production Deployments**:
```bash
# Use external secret management
# Examples:
# - HashiCorp Vault
# - AWS Secrets Manager
# - Azure Key Vault
# - Kubernetes Secrets

# Docker Swarm secrets example
echo "my_secure_password" | docker secret create db_password -
```

**Environment Variable Injection**:
```yaml
# docker-compose.prod.yml example
services:
  influxdb:
    environment:
      - INFLUXDB_ADMIN_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password

secrets:
  db_password:
    external: true
```

## Access Control Configuration

### InfluxDB Security

**Authentication Settings**:
```bash
# Enable authentication (production)
INFLUXDB_HTTP_AUTH_ENABLED=true

# Create dedicated users
INFLUXDB_USER=pulsar_app
INFLUXDB_PASSWORD=secure_app_password

# Admin user (limited use)
INFLUXDB_ADMIN_USER=admin
INFLUXDB_ADMIN_PASSWORD=secure_admin_password
```

**Database Access Control**:
```sql
-- Create application-specific user
CREATE USER "pulsar_app" WITH PASSWORD 'secure_app_password'

-- Grant specific database access
GRANT WRITE ON "prediction" TO "pulsar_app"
GRANT WRITE ON "metrics" TO "pulsar_app"
GRANT READ ON "prediction" TO "grafana_user"
GRANT READ ON "metrics" TO "grafana_user"
```

### Grafana Security

**User Management**:
```bash
# Admin user configuration
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=secure_admin_password

# Disable user signup
GF_USERS_ALLOW_SIGN_UP=false

# Session security
GF_SESSION_COOKIE_SECURE=true
GF_SESSION_COOKIE_SAMESITE=strict
GF_SESSION_SESSION_LIFE_TIME=86400
```

**Authentication Options**:
```bash
# LDAP integration
GF_AUTH_LDAP_ENABLED=true
GF_AUTH_LDAP_CONFIG_FILE=/etc/grafana/ldap.toml

# OAuth integration
GF_AUTH_GENERIC_OAUTH_ENABLED=true
GF_AUTH_GENERIC_OAUTH_CLIENT_ID=your_client_id
GF_AUTH_GENERIC_OAUTH_CLIENT_SECRET=your_client_secret
```

### Network Security

**Docker Network Isolation**:
```yaml
# Custom network configuration
networks:
  pulsar_internal:
    driver: bridge
    internal: true  # No external access
  pulsar_external:
    driver: bridge

services:
  influxdb:
    networks:
      - pulsar_internal  # Internal communication only
  
  grafana:
    networks:
      - pulsar_internal  # Database access
      - pulsar_external  # External web access
    ports:
      - "3000:3000"
```

**Firewall Configuration**:
```bash
# UFW example (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific services
sudo ufw allow 3000/tcp comment 'Grafana'
sudo ufw allow from 172.18.0.0/16 to any port 8086 comment 'InfluxDB internal'

# Docker and UFW
sudo ufw reload
```

## Container Security

### Image Security

**Base Image Selection**:
```dockerfile
# Use official, minimal base images
FROM python:3.9-slim-bullseye

# Keep base images updated
FROM grafana/grafana:latest
FROM influxdb:1.8-alpine
```

**Vulnerability Scanning**:
```bash
# Scan images for vulnerabilities
docker scan pulsar-demo:latest

# Use security scanning tools
trivy image pulsar-demo:latest
```

### Container Hardening

**Resource Limits**:
```yaml
services:
  compute-metrics:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          memory: 512M
    
  influxdb:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
```

**Security Options**:
```yaml
services:
  app:
    security_opt:
      - no-new-privileges:true
      - apparmor=docker-default
    cap_drop:
      - ALL
    cap_add:
      - SETGID
      - SETUID
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
```

**User Configuration**:
```dockerfile
# Create non-root user
RUN groupadd -r pulsar && useradd -r -g pulsar pulsar
RUN chown -R pulsar:pulsar /app
USER pulsar

# Or use existing user
USER 1001:1001
```

## Data Security

### Data at Rest

**Volume Encryption**:
```bash
# Linux LUKS encryption
sudo cryptsetup luksFormat /dev/sdb
sudo cryptsetup luksOpen /dev/sdb encrypted_volume
sudo mkfs.ext4 /dev/mapper/encrypted_volume
```

**Backup Encryption**:
```bash
# Encrypted backups
tar -czf - /var/lib/docker/volumes/ | \
    gpg --cipher-algo AES256 --compress-algo 2 --symmetric \
    --output backup_$(date +%Y%m%d).tar.gz.gpg
```

### Data in Transit

**TLS Configuration**:
```yaml
# HTTPS for Grafana
services:
  grafana:
    environment:
      - GF_SERVER_PROTOCOL=https
      - GF_SERVER_CERT_FILE=/etc/ssl/certs/grafana.crt
      - GF_SERVER_CERT_KEY=/etc/ssl/private/grafana.key
    volumes:
      - ./certs:/etc/ssl/certs:ro
      - ./private:/etc/ssl/private:ro
```

**Certificate Management**:
```bash
# Generate self-signed certificates (development only)
openssl req -x509 -newkey rsa:4096 -keyout grafana.key \
    -out grafana.crt -days 365 -nodes

# Use Let's Encrypt for production
certbot certonly --standalone -d yourdomain.com
```

## Monitoring and Auditing

### Security Monitoring

**Log Configuration**:
```yaml
services:
  influxdb:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=influxdb"
```

**Security Events to Monitor**:
- Failed authentication attempts
- Unusual data access patterns
- Configuration changes
- Container restart events
- Network connection anomalies

**Example Log Monitoring**:
```bash
# Monitor authentication failures
docker logs grafana 2>&1 | grep "authentication failed"

# Monitor database connections
docker logs influxdb 2>&1 | grep "authentication"

# Monitor container events
docker events --filter container=pulsar-demo --since 24h
```

### Audit Trail

**Database Audit**:
```sql
-- Enable InfluxDB logging
-- Add to influxdb.conf
[http]
  log-enabled = true
  write-tracing = true
  pprof-enabled = false

[data]
  query-log-enabled = true
```

**Application Audit**:
```python
# Add audit logging to Python services
import logging
import json

# Configure audit logger
audit_logger = logging.getLogger('audit')
handler = logging.FileHandler('/var/log/audit.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
audit_logger.addHandler(handler)

# Log security events
def log_security_event(event_type, user_id, details):
    audit_log = {
        'event_type': event_type,
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'details': details
    }
    audit_logger.info(json.dumps(audit_log))
```

## Vulnerability Management

### Regular Updates

**Update Schedule**:
```bash
# Monthly security updates
#!/bin/bash
# update-containers.sh

# Pull latest images
docker-compose pull

# Rebuild custom images
docker-compose build --no-cache

# Restart services
docker-compose up -d
```

**Security Patches**:
```dockerfile
# Always include security updates
FROM python:3.9-slim
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --only-upgrade \
    base-files libc-bin && \
    rm -rf /var/lib/apt/lists/*
```

### Dependency Management

**Python Dependencies**:
```bash
# Check for known vulnerabilities
pip install safety
safety check

# Update dependencies
pip-review --auto

# Pin versions for reproducible builds
pip freeze > requirements.txt
```

**Container Scanning**:
```bash
# Regular vulnerability scans
docker scan --severity high pulsar-demo:latest

# Continuous monitoring
docker scan --json pulsar-demo:latest | \
    jq '.vulnerabilities[] | select(.severity == "high")'
```

## Incident Response

### Security Incident Procedures

**Detection**:
1. Monitor security logs and alerts
2. Check for unusual system behavior
3. Verify data integrity

**Response**:
1. Isolate affected containers
2. Collect forensic data
3. Apply security patches
4. Reset compromised credentials

**Recovery**:
```bash
# Emergency procedures

# 1. Stop compromised services
docker-compose stop

# 2. Backup current state
docker run --rm -v pulsar_demo_influxdb_data:/data \
    -v $(pwd):/backup ubuntu \
    tar czf /backup/emergency_backup.tar.gz /data

# 3. Reset credentials
./scripts/reset-credentials.sh

# 4. Restart with new credentials
docker-compose up -d
```

### Contact Information

**Security Contacts**:
- **Internal Security Team**: security@yourcompany.com
- **Pulsar.ML Security**: Report to GitHub Issues
- **Docker Security**: https://docs.docker.com/security/
- **CVE Database**: https://cve.mitre.org/

## Compliance Considerations

### Data Protection Regulations

**GDPR Compliance**:
- Implement data anonymization
- Provide data deletion capabilities
- Maintain audit trails for data processing

**Example Data Anonymization**:
```python
from faker import Faker
import pandas as pd

def anonymize_data(df, columns_to_anonymize):
    fake = Faker()
    anonymized_df = df.copy()
    
    for column in columns_to_anonymize:
        if column in df.columns:
            anonymized_df[column] = [fake.name() for _ in range(len(df))]
    
    return anonymized_df
```

### Industry Standards

**Security Frameworks**:
- NIST Cybersecurity Framework
- ISO 27001/27002
- CIS Controls
- OWASP Security Guidelines

**Implementation Checklist**:
- [ ] Strong authentication implemented
- [ ] Network segmentation configured
- [ ] Data encryption enabled
- [ ] Regular security updates applied
- [ ] Monitoring and logging configured
- [ ] Incident response plan documented
- [ ] Security training completed

## Security Testing

### Penetration Testing

**Automated Security Testing**:
```bash
# Network security testing
nmap -sV -sC localhost

# Web application testing
nikto -h http://localhost:3000

# Container security testing
docker-bench-security
```

**Security Test Script**:
```bash
#!/bin/bash
# security-test.sh

echo "Running security tests..."

# Test default passwords
echo "Checking for default passwords..."
curl -u admin:admin http://localhost:3000/api/health 2>/dev/null && \
    echo "WARNING: Default Grafana password detected!"

# Test network isolation
echo "Testing network isolation..."
docker exec -it pulsar-demo_compute-metrics_1 \
    nc -zv google.com 80 2>/dev/null && \
    echo "WARNING: External network access detected!"

# Test file permissions
echo "Checking file permissions..."
find . -name "*.env" -perm /044 -exec echo "WARNING: {} is world-readable!" \;

echo "Security test complete."
```

This comprehensive security guide provides the foundation for secure deployment and operation of the Pulsar.ML demo environment. Regularly review and update these practices as the security landscape evolves.