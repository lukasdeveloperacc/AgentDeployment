variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "private_subnets" {
  description = "Private subnet IDs"
  type        = list(string)
}

variable "security_groups" {
  description = "Security group IDs"
  type        = list(string)
}

# Backend Configuration
variable "backend_image" {
  description = "Docker image for backend"
  type        = string
}

variable "backend_cpu" {
  description = "CPU units for backend"
  type        = number
  default     = 512
}

variable "backend_memory" {
  description = "Memory for backend"
  type        = number
  default     = 1024
}

variable "backend_desired_count" {
  description = "Desired count of backend tasks"
  type        = number
  default     = 2
}

variable "backend_target_group_arn" {
  description = "ARN of backend target group"
  type        = string
}

# Frontend Configuration
variable "frontend_image" {
  description = "Docker image for frontend"
  type        = string
}

variable "frontend_cpu" {
  description = "CPU units for frontend"
  type        = number
  default     = 256
}

variable "frontend_memory" {
  description = "Memory for frontend"
  type        = number
  default     = 512
}

variable "frontend_desired_count" {
  description = "Desired count of frontend tasks"
  type        = number
  default     = 2
}

variable "frontend_target_group_arn" {
  description = "ARN of frontend target group"
  type        = string
}

# Secrets
variable "openai_secret_arn" {
  description = "ARN of OpenAI API key secret"
  type        = string
  sensitive   = true
}

variable "pinecone_secret_arn" {
  description = "ARN of Pinecone API key secret"
  type        = string
  sensitive   = true
}
