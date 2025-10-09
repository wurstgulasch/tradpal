#!/bin/bash
# AWS EC2 Deployment Script for TradPal Indicator
# This script sets up TradPal on AWS EC2 with proper security and monitoring

set -e

# Configuration
STACK_NAME="tradpal-indicator"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
KEY_NAME="${KEY_NAME:-tradpal-key}"
VPC_ID="${VPC_ID:-}"
SUBNET_ID="${SUBNET_ID:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install it first."
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install it first."
        exit 1
    fi

    log "Prerequisites check passed."
}

# Create VPC and subnets if not provided
create_vpc() {
    if [ -z "$VPC_ID" ]; then
        log "Creating VPC..."
        VPC_ID=$(aws ec2 create-vpc --cidr-block 10.0.0.0/16 --region $REGION --query 'Vpc.VpcId' --output text)
        aws ec2 create-tags --resources $VPC_ID --tags Key=Name,Value=tradpal-vpc --region $REGION

        # Create subnet
        SUBNET_ID=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.1.0/24 --region $REGION --query 'Subnet.SubnetId' --output text)
        aws ec2 create-tags --resources $SUBNET_ID --tags Key=Name,Value=tradpal-subnet --region $REGION

        # Create Internet Gateway
        IGW_ID=$(aws ec2 create-internet-gateway --region $REGION --query 'InternetGateway.InternetGatewayId' --output text)
        aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID --region $REGION

        # Create route table
        RT_ID=$(aws ec2 create-route-table --vpc-id $VPC_ID --region $REGION --query 'RouteTable.RouteTableId' --output text)
        aws ec2 create-route --route-table-id $RT_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID --region $REGION
        aws ec2 associate-route-table --subnet-id $SUBNET_ID --route-table-id $RT_ID --region $REGION

        log "VPC created: $VPC_ID"
        log "Subnet created: $SUBNET_ID"
    fi
}

# Create security group
create_security_group() {
    log "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name tradpal-sg \
        --description "Security group for TradPal Indicator" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' \
        --output text)

    # Allow SSH
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    # Allow HTTP for monitoring
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    # Allow HTTPS
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    # Allow Prometheus metrics
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    log "Security group created: $SG_ID"
}

# Create IAM role for EC2
create_iam_role() {
    log "Creating IAM role..."

    # Create role
    aws iam create-role \
        --role-name tradpal-ec2-role \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "ec2.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }'

    # Attach policies
    aws iam attach-role-policy \
        --role-name tradpal-ec2-role \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess

    aws iam attach-role-policy \
        --role-name tradpal-ec2-role \
        --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

    # Create instance profile
    aws iam create-instance-profile --instance-profile-name tradpal-instance-profile
    aws iam add-role-to-instance-profile \
        --instance-profile-name tradpal-instance-profile \
        --role-name tradpal-ec2-role

    log "IAM role created: tradpal-ec2-role"
}

# Create EC2 instance
create_ec2_instance() {
    log "Creating EC2 instance..."

    # Get latest Amazon Linux 2 AMI
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text \
        --region $REGION)

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --subnet-id $SUBNET_ID \
        --iam-instance-profile Name=tradpal-instance-profile \
        --user-data file://aws/user-data.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tradpal-indicator},{Key=Environment,Value=production}]" \
        --region $REGION \
        --query 'Instances[0].InstanceId' \
        --output text)

    log "EC2 instance created: $INSTANCE_ID"

    # Wait for instance to be running
    log "Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    log "Instance is running at: $PUBLIC_IP"
}

# Create CloudWatch alarms
create_cloudwatch_alarms() {
    log "Creating CloudWatch alarms..."

    # CPU utilization alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "tradpal-high-cpu" \
        --alarm-description "CPU utilization is high" \
        --metric-name CPUUtilization \
        --namespace AWS/EC2 \
        --statistic Average \
        --period 300 \
        --threshold 80 \
        --comparison-operator GreaterThanThreshold \
        --dimensions Name=InstanceId,Value=$INSTANCE_ID \
        --evaluation-periods 2 \
        --alarm-actions $SNS_TOPIC_ARN \
        --region $REGION

    # Memory utilization alarm (requires CloudWatch agent)
    aws cloudwatch put-metric-alarm \
        --alarm-name "tradpal-high-memory" \
        --alarm-description "Memory utilization is high" \
        --metric-name mem_used_percent \
        --namespace CWAgent \
        --statistic Average \
        --period 300 \
        --threshold 85 \
        --comparison-operator GreaterThanThreshold \
        --dimensions Name=InstanceId,Value=$INSTANCE_ID \
        --evaluation-periods 2 \
        --alarm-actions $SNS_TOPIC_ARN \
        --region $REGION

    log "CloudWatch alarms created."
}

# Main deployment function
main() {
    log "Starting TradPal Indicator AWS deployment..."

    check_prerequisites
    create_vpc
    create_security_group
    create_iam_role
    create_ec2_instance
    create_cloudwatch_alarms

    log "Deployment completed successfully!"
    log "Instance ID: $INSTANCE_ID"
    log "Public IP: $PUBLIC_IP"
    log ""
    log "Next steps:"
    log "1. SSH into the instance: ssh -i $KEY_NAME.pem ec2-user@$PUBLIC_IP"
    log "2. Check logs: docker logs tradpal-indicator"
    log "3. Access monitoring: http://$PUBLIC_IP:8000"
    log "4. Configure secrets in AWS Secrets Manager"
}

# Run main function
main "$@"