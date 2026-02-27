# Artifact Registry Module - Container Registry for GCP

resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = "${var.project_name}-${var.environment}"
  description   = "Docker repository for ${var.project_name}"
  format        = "DOCKER"

  labels = {
    project     = var.project_name
    environment = var.environment
  }
}

# Cleanup policy - keep only last 10 images
resource "google_artifact_registry_repository_cleanup_policy" "policy" {
  repository = google_artifact_registry_repository.main.id
  location   = google_artifact_registry_repository.main.location

  policy_id = "keep-last-10"
  action    = "DELETE"

  condition {
    tag_state = "ANY"
    older_than = "0s"
  }

  most_recent_versions {
    keep_count = 10
  }
}
