# GCP Infrastructure Variables

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region for resources"
  type        = string
  default     = "asia-northeast3"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "rag-demo"
}

variable "environment" {
  description = "Environment name (dev, stage, prod)"
  type        = string
  default     = "dev"
}
