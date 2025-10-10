#!/bin/bash
# TradPal EC2 User Data Script
# This script runs on EC2 instance startup to configure the environment

set -e

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
yum install -y git

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U amazon-cloudwatch-agent.rpm

# Create CloudWatch agent configuration
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "metrics": {
        "metrics_collected": {
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 300
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 300,
                "resources": [
                    "/"
                ]
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# Create application directory
mkdir -p /opt/tradpal
cd /opt/tradpal

# Clone repository (replace with your actual repo)
git clone https://github.com/wurstgulasch/tradpal_indicator.git .

# Create environment file
cat > .env << EOF
# TradPal Configuration
TRADPAL_API_KEY=""
TRADPAL_API_SECRET=""
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""

# AWS Configuration
SECRETS_BACKEND=aws-secretsmanager
AWS_REGION=us-east-1
DEPLOYMENT_ENV=aws

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8000

# Application Settings
TEST_ENVIRONMENT=false
PYTHONPATH=/app
PYTHONUNBUFFERED=1
EOF

# Create docker-compose override for AWS
cat > docker-compose.aws.yml << EOF
version: '3.8'

services:
  tradpal-indicator:
    environment:
      - SECRETS_BACKEND=aws-secretsmanager
      - DEPLOYMENT_ENV=aws
      - PROMETHEUS_ENABLED=true
    volumes:
      - /opt/tradpal/output:/app/output:rw
      - /opt/tradpal/config:/app/config:ro
      - /opt/tradpal/logs:/app/logs:rw
      - /opt/tradpal/cache:/app/cache:rw
    restart: always
    logging:
      driver: awslogs
      options:
        awslogs-group: tradpal-indicator
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs

  prometheus:
    profiles: ["monitoring"]
    volumes:
      - /opt/tradpal/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    profiles: ["monitoring"]
EOF

# Create systemd service for TradPal
cat > /etc/systemd/system/tradpal.service << EOF
[Unit]
Description=TradPal Trading System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/tradpal
ExecStart=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.aws.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.aws.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable tradpal.service
systemctl start tradpal.service

# Setup logrotate for application logs
cat > /etc/logrotate.d/tradpal << EOF
/opt/tradpal/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 ec2-user ec2-user
    postrotate
        systemctl reload tradpal.service
    endscript
}
EOF

# Create backup script
cat > /opt/tradpal/backup.sh << 'EOF'
#!/bin/bash
# Daily backup script for TradPal data

BACKUP_DIR="/opt/tradpal/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup output and config
tar -czf $BACKUP_DIR/tradpal_backup_$DATE.tar.gz \
    -C /opt/tradpal output config

# Upload to S3 (if configured)
if command -v aws &> /dev/null; then
    aws s3 cp $BACKUP_DIR/tradpal_backup_$DATE.tar.gz s3://tradpal-backups/
fi

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "tradpal_backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/tradpal_backup_$DATE.tar.gz"
EOF

chmod +x /opt/tradpal/backup.sh

# Add backup to cron
echo "0 2 * * * /opt/tradpal/backup.sh" | crontab -

# Install monitoring tools
yum install -y htop iotop sysstat

# Final setup message
cat > /home/ec2-user/README.md << 'EOF'
# TradPal - AWS Deployment

Your TradPal instance has been successfully deployed!

## Access Information
- Application: Running on port 8000
- Prometheus: http://<instance-ip>:9090 (if enabled)
- Grafana: http://<instance-ip>:3000 (if enabled)

## Key Directories
- Application: /opt/tradpal
- Logs: /opt/tradpal/logs
- Output: /opt/tradpal/output
- Config: /opt/tradpal/config

## Management Commands
- Start: sudo systemctl start tradpal
- Stop: sudo systemctl stop tradpal
- Restart: sudo systemctl restart tradpal
- Logs: sudo systemctl status tradpal
- Docker logs: docker-compose logs -f

## Monitoring
- System metrics: htop, iotop
- Application metrics: http://localhost:8000/metrics (if Prometheus enabled)
- CloudWatch: Check AWS Console for detailed metrics

## Backup
- Daily backups run at 2 AM
- Backup location: /opt/tradpal/backups
- S3 upload: Configured automatically (if S3 bucket exists)

## Security Notes
- SSH key authentication enabled
- Security groups configured for minimal access
- Secrets managed via AWS Secrets Manager
- Regular system updates enabled

For support, check the logs or contact the administrator.
EOF

# Signal completion
/opt/aws/bin/cfn-signal -e $? --stack tradpal-indicator --resource EC2Instance --region us-east-1 || true