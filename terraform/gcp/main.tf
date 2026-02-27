# GCP Cloud Run Infrastructure - Main Terraform Configuration
# Section 5: Infrastructure as Code (IaC)

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # GCS Backend for state management (optional, configure if needed)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "rag-demo/terraform.tfstate"
  # }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Enable required APIs
resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])

  service = each.value

  disable_on_destroy = false
}

# Artifact Registry Module
module "artifact_registry" {
  source = "./modules/artifact-registry"

  project_id   = var.gcp_project_id
  region       = var.gcp_region
  project_name = var.project_name
  environment  = var.environment

  depends_on = [google_project_service.services]
}

# Secret Manager for API Keys
resource "google_secret_manager_secret" "openai_api_key" {
  secret_id = "openai-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret" "pinecone_api_key" {
  secret_id = "pinecone-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

# Secret versions (must be created separately with actual values)
# Use: echo -n "your-api-key" | gcloud secrets versions add openai-api-key --data-file=-

# Cloud Run Module
module "cloud_run" {
  source = "./modules/cloud-run"

  project_id   = var.gcp_project_id
  region       = var.gcp_region
  project_name = var.project_name
  environment  = var.environment

  # Backend Configuration
  backend_image           = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${module.artifact_registry.repository_name}/rag-backend:latest"
  backend_cpu             = "2"
  backend_memory          = "2Gi"
  backend_min_instances   = 0
  backend_max_instances   = 10
  backend_timeout         = 300

  # Frontend Configuration
  frontend_image          = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${module.artifact_registry.repository_name}/rag-frontend:latest"
  frontend_cpu            = "1"
  frontend_memory         = "512Mi"
  frontend_min_instances  = 0
  frontend_max_instances  = 5
  frontend_timeout        = 60

  # Secrets
  openai_secret_name   = google_secret_manager_secret.openai_api_key.secret_id
  pinecone_secret_name = google_secret_manager_secret.pinecone_api_key.secret_id

  depends_on = [module.artifact_registry, google_project_service.services]
}

# Outputs
output "backend_url" {
  description = "URL of the backend Cloud Run service"
  value       = module.cloud_run.backend_url
}

output "frontend_url" {
  description = "URL of the frontend Cloud Run service"
  value       = module.cloud_run.frontend_url
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = module.artifact_registry.repository_url
}
