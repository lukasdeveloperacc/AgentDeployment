# Cloud Run Module - Serverless Container Platform

# Service Account for Cloud Run
resource "google_service_account" "cloud_run" {
  account_id   = "${var.project_name}-${var.environment}-run-sa"
  display_name = "Cloud Run Service Account"
  description  = "Service account for Cloud Run services"
}

# Grant Secret Manager access to Service Account
resource "google_secret_manager_secret_iam_member" "openai_access" {
  secret_id = var.openai_secret_name
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_secret_manager_secret_iam_member" "pinecone_access" {
  secret_id = var.pinecone_secret_name
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Backend Cloud Run Service
resource "google_cloud_run_v2_service" "backend" {
  name     = "${var.project_name}-${var.environment}-backend"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run.email

    scaling {
      min_instance_count = var.backend_min_instances
      max_instance_count = var.backend_max_instances
    }

    timeout = "${var.backend_timeout}s"

    containers {
      image = var.backend_image

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = var.backend_cpu
          memory = var.backend_memory
        }
        cpu_idle = false
        startup_cpu_boost = true
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = var.openai_secret_name
            version = "latest"
          }
        }
      }

      env {
        name = "PINECONE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = var.pinecone_secret_name
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 0
        timeout_seconds       = 1
        period_seconds        = 3
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        period_seconds = 10
      }
    }
  }

  labels = {
    project     = var.project_name
    environment = var.environment
    service     = "backend"
  }

  depends_on = [
    google_secret_manager_secret_iam_member.openai_access,
    google_secret_manager_secret_iam_member.pinecone_access
  ]
}

# Frontend Cloud Run Service
resource "google_cloud_run_v2_service" "frontend" {
  name     = "${var.project_name}-${var.environment}-frontend"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run.email

    scaling {
      min_instance_count = var.frontend_min_instances
      max_instance_count = var.frontend_max_instances
    }

    timeout = "${var.frontend_timeout}s"

    containers {
      image = var.frontend_image

      ports {
        container_port = 80
      }

      resources {
        limits = {
          cpu    = var.frontend_cpu
          memory = var.frontend_memory
        }
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 80
        }
        period_seconds = 10
      }
    }
  }

  labels = {
    project     = var.project_name
    environment = var.environment
    service     = "frontend"
  }
}

# IAM policy to allow unauthenticated access
resource "google_cloud_run_service_iam_member" "backend_noauth" {
  location = google_cloud_run_v2_service.backend.location
  service  = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "frontend_noauth" {
  location = google_cloud_run_v2_service.frontend.location
  service  = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
