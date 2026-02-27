variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

# Backend Configuration
variable "backend_image" {
  description = "Docker image for backend"
  type        = string
}

variable "backend_cpu" {
  description = "CPU allocation for backend"
  type        = string
  default     = "2"
}

variable "backend_memory" {
  description = "Memory allocation for backend"
  type        = string
  default     = "2Gi"
}

variable "backend_min_instances" {
  description = "Minimum instances for backend"
  type        = number
  default     = 0
}

variable "backend_max_instances" {
  description = "Maximum instances for backend"
  type        = number
  default     = 10
}

variable "backend_timeout" {
  description = "Request timeout for backend (seconds)"
  type        = number
  default     = 300
}

# Frontend Configuration
variable "frontend_image" {
  description = "Docker image for frontend"
  type        = string
}

variable "frontend_cpu" {
  description = "CPU allocation for frontend"
  type        = string
  default     = "1"
}

variable "frontend_memory" {
  description = "Memory allocation for frontend"
  type        = string
  default     = "512Mi"
}

variable "frontend_min_instances" {
  description = "Minimum instances for frontend"
  type        = number
  default     = 0
}

variable "frontend_max_instances" {
  description = "Maximum instances for frontend"
  type        = number
  default     = 5
}

variable "frontend_timeout" {
  description = "Request timeout for frontend (seconds)"
  type        = number
  default     = 60
}

# Secrets
variable "openai_secret_name" {
  description = "Name of OpenAI API key secret"
  type        = string
}

variable "pinecone_secret_name" {
  description = "Name of Pinecone API key secret"
  type        = string
}
