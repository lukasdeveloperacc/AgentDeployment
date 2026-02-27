#!/bin/bash
# GCP Cloud Run 배포 스크립트
# Section 4: GCP Cloud Run 배포 실습용

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 변수 설정
GCP_PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
GCP_REGION="asia-northeast3"  # Seoul
ARTIFACT_REGISTRY_REPO="rag-demo"
IMAGE_TAG="${1:-latest}"

echo -e "${GREEN}=== GCP Cloud Run 배포 시작 ===${NC}"
echo "Project ID: $GCP_PROJECT_ID"
echo "Region: $GCP_REGION"
echo "Image Tag: $IMAGE_TAG"

# 1. gcloud 인증 확인
echo -e "\n${YELLOW}[1/6] gcloud 인증 확인...${NC}"
gcloud auth list
gcloud config set project $GCP_PROJECT_ID

# 2. Artifact Registry 활성화 및 Repository 생성
echo -e "\n${YELLOW}[2/6] Artifact Registry 설정...${NC}"
gcloud services enable artifactregistry.googleapis.com

# Repository 생성 (없으면)
gcloud artifacts repositories describe $ARTIFACT_REGISTRY_REPO \
  --location=$GCP_REGION \
  --format="value(name)" 2>/dev/null || \
gcloud artifacts repositories create $ARTIFACT_REGISTRY_REPO \
  --repository-format=docker \
  --location=$GCP_REGION \
  --description="RAG Demo Docker Images"

# 3. gcloud Docker 인증
echo -e "\n${YELLOW}[3/6] Docker 인증 설정...${NC}"
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# 4. Docker 이미지 빌드
echo -e "\n${YELLOW}[4/6] Docker 이미지 빌드...${NC}"
docker build -t rag-backend:$IMAGE_TAG ./backend
docker build -t rag-frontend:$IMAGE_TAG ./frontend

# 5. 이미지 태깅
echo -e "\n${YELLOW}[5/6] 이미지 태깅...${NC}"
docker tag rag-backend:$IMAGE_TAG ${GCP_REGION}-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/rag-backend:$IMAGE_TAG
docker tag rag-frontend:$IMAGE_TAG ${GCP_REGION}-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/rag-frontend:$IMAGE_TAG

# 6. Artifact Registry에 푸시
echo -e "\n${YELLOW}[6/6] Artifact Registry에 이미지 푸시...${NC}"
docker push ${GCP_REGION}-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/rag-backend:$IMAGE_TAG
docker push ${GCP_REGION}-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/rag-frontend:$IMAGE_TAG

echo -e "\n${GREEN}=== 배포 완료! ===${NC}"
echo "Backend Image: ${GCP_REGION}-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/rag-backend:$IMAGE_TAG"
echo "Frontend Image: ${GCP_REGION}-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/rag-frontend:$IMAGE_TAG"
echo -e "\n${YELLOW}다음 단계: Cloud Run 서비스 배포${NC}"
echo "gcloud run deploy rag-backend --image ${GCP_REGION}-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/rag-backend:$IMAGE_TAG --region $GCP_REGION"
