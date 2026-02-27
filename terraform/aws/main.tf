# AWS ECS Infrastructure - Main Terraform Configuration
# Section 5: Infrastructure as Code (IaC)

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # S3 Backend for state management (optional, configure if needed)
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "rag-demo/terraform.tfstate"
  #   region = "ap-northeast-2"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "RAG-Demo"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr

  availability_zones = var.availability_zones
  public_subnets     = var.public_subnets
  private_subnets    = var.private_subnets
}

# ECR Repositories
module "ecr" {
  source = "./modules/ecr"

  project_name = var.project_name
  environment  = var.environment

  repositories = [
    "rag-backend",
    "rag-frontend"
  ]
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"

  project_name = var.project_name
  environment  = var.environment

  vpc_id          = module.vpc.vpc_id
  public_subnets  = module.vpc.public_subnet_ids
  security_groups = [module.vpc.alb_security_group_id]
}

# ECS Cluster and Services
module "ecs" {
  source = "./modules/ecs"

  project_name = var.project_name
  environment  = var.environment

  vpc_id             = module.vpc.vpc_id
  private_subnets    = module.vpc.private_subnet_ids
  security_groups    = [module.vpc.ecs_security_group_id]

  # Backend Service
  backend_image      = "${module.ecr.repository_urls["rag-backend"]}:latest"
  backend_cpu        = 512
  backend_memory     = 1024
  backend_desired_count = 2

  # Frontend Service
  frontend_image     = "${module.ecr.repository_urls["rag-frontend"]}:latest"
  frontend_cpu       = 256
  frontend_memory    = 512
  frontend_desired_count = 2

  # Load Balancer Integration
  backend_target_group_arn  = module.alb.backend_target_group_arn
  frontend_target_group_arn = module.alb.frontend_target_group_arn

  # Secrets (create in AWS Secrets Manager first)
  openai_secret_arn   = var.openai_secret_arn
  pinecone_secret_arn = var.pinecone_secret_arn
}

# Outputs
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.alb_dns_name
}

output "backend_ecr_url" {
  description = "ECR repository URL for backend"
  value       = module.ecr.repository_urls["rag-backend"]
}

output "frontend_ecr_url" {
  description = "ECR repository URL for frontend"
  value       = module.ecr.repository_urls["rag-frontend"]
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = module.ecs.cluster_name
}
