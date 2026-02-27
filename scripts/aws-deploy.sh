#!/bin/bash
# AWS ECS/Fargate 배포 스크립트
# Section 3: AWS ECS/Fargate 배포 실습용

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 변수 설정
AWS_REGION="ap-northeast-2"  # Seoul
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_BACKEND_REPO="rag-backend"
ECR_FRONTEND_REPO="rag-frontend"
IMAGE_TAG="${1:-latest}"

echo -e "${GREEN}=== AWS ECS/Fargate 배포 시작 ===${NC}"
echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "Image Tag: $IMAGE_TAG"

# 1. ECR 로그인
echo -e "\n${YELLOW}[1/5] ECR 로그인...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 2. ECR Repository 생성 (없으면)
echo -e "\n${YELLOW}[2/5] ECR Repository 생성 확인...${NC}"
aws ecr describe-repositories --repository-names $ECR_BACKEND_REPO --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_BACKEND_REPO --region $AWS_REGION

aws ecr describe-repositories --repository-names $ECR_FRONTEND_REPO --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_FRONTEND_REPO --region $AWS_REGION

# 3. Docker 이미지 빌드
echo -e "\n${YELLOW}[3/5] Docker 이미지 빌드...${NC}"
docker build -t $ECR_BACKEND_REPO:$IMAGE_TAG ./backend
docker build -t $ECR_FRONTEND_REPO:$IMAGE_TAG ./frontend

# 4. 이미지 태깅
echo -e "\n${YELLOW}[4/5] 이미지 태깅...${NC}"
docker tag $ECR_BACKEND_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:$IMAGE_TAG
docker tag $ECR_FRONTEND_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:$IMAGE_TAG

# 5. ECR에 푸시
echo -e "\n${YELLOW}[5/5] ECR에 이미지 푸시...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:$IMAGE_TAG

echo -e "\n${GREEN}=== 배포 완료! ===${NC}"
echo "Backend Image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:$IMAGE_TAG"
echo "Frontend Image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:$IMAGE_TAG"
echo -e "\n${YELLOW}다음 단계: AWS Console에서 ECS Service 생성${NC}"
