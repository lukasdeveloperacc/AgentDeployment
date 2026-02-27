output "repository_name" {
  description = "Name of the Artifact Registry repository"
  value       = google_artifact_registry_repository.main.repository_id
}

output "repository_url" {
  description = "URL of the Artifact Registry repository"
  value       = "${google_artifact_registry_repository.main.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.main.repository_id}"
}
