# Section 3: AWS ECS/Fargate 배포

## 학습 목표
- AWS 컨테이너 서비스 핵심 개념 이해 (ECR, ECS, Fargate, ALB)
- 프로덕션 배포 아키텍처 설계
- ECR에 컨테이너 이미지 업로드
- ECS/Fargate로 실제 서비스 배포
- CloudWatch 모니터링 및 장애 대응

## 새로 추가된 파일
```
scripts/
  └── aws-deploy.sh                    # AWS 배포 자동화 스크립트

docs/aws/
  ├── task-definition-backend.json    # Backend ECS Task Definition
  └── task-definition-frontend.json   # Frontend ECS Task Definition
```

## AWS 서비스 구성도

```
┌─────────────────────────────────────────────────┐
│                   Internet                      │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │  Application Load     │
        │  Balancer (ALB)       │
        └───────┬───────┬───────┘
                │       │
        ┌───────▼──┐ ┌──▼────────┐
        │ Target   │ │ Target    │
        │ Group    │ │ Group     │
        │ (FE:80)  │ │ (BE:8000) │
        └───────┬──┘ └──┬────────┘
                │       │
        ┌───────▼──┐ ┌──▼────────┐
        │ ECS      │ │ ECS       │
        │ Service  │ │ Service   │
        │ (Frontend)│ │ (Backend)│
        └───────┬──┘ └──┬────────┘
                │       │
        ┌───────▼──┐ ┌──▼────────┐
        │ Fargate  │ │ Fargate   │
        │ Task     │ │ Task      │
        └──────────┘ └───────────┘
```

## 배포 단계

### 1. AWS CLI 설정
```bash
# AWS CLI 설치 확인
aws --version

# AWS 자격증명 설정
aws configure
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region: ap-northeast-2
# Default output format: json
```

### 2. ECR에 이미지 푸시
```bash
# 배포 스크립트 실행
chmod +x scripts/aws-deploy.sh
./scripts/aws-deploy.sh latest

# 또는 수동으로:
# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin {ACCOUNT_ID}.dkr.ecr.ap-northeast-2.amazonaws.com

# 이미지 빌드 및 태깅
docker build -t rag-backend:latest ./backend
docker tag rag-backend:latest {ACCOUNT_ID}.dkr.ecr.ap-northeast-2.amazonaws.com/rag-backend:latest

# ECR 푸시
docker push {ACCOUNT_ID}.dkr.ecr.ap-northeast-2.amazonaws.com/rag-backend:latest
```

### 3. Secrets Manager에 API Key 저장
```bash
# OpenAI API Key 저장
aws secretsmanager create-secret \
  --name prod/openai-api-key \
  --secret-string "your-openai-api-key" \
  --region ap-northeast-2

# Pinecone API Key 저장
aws secretsmanager create-secret \
  --name prod/pinecone-api-key \
  --secret-string "your-pinecone-api-key" \
  --region ap-northeast-2
```

### 4. ECS Cluster 생성
```bash
# AWS Console > ECS > Clusters > Create Cluster
# - Cluster name: rag-demo-cluster
# - Infrastructure: AWS Fargate (serverless)
```

### 5. Task Definition 등록
```bash
# Task Definition 파일 수정 ({ACCOUNT_ID} 교체)
sed -i "s/{ACCOUNT_ID}/$(aws sts get-caller-identity --query Account --output text)/g" docs/aws/task-definition-backend.json

# Task Definition 등록
aws ecs register-task-definition \
  --cli-input-json file://docs/aws/task-definition-backend.json

aws ecs register-task-definition \
  --cli-input-json file://docs/aws/task-definition-frontend.json
```

### 6. ALB 및 Target Group 생성
```bash
# AWS Console에서 생성:
# 1. Application Load Balancer 생성
# 2. Target Group 생성 (IP 타입, Frontend:80, Backend:8000)
# 3. Listener Rules 설정
```

### 7. ECS Service 생성
```bash
# AWS Console > ECS > Cluster > Create Service
# - Launch type: Fargate
# - Task Definition: rag-backend-task
# - Service name: rag-backend-service
# - Desired tasks: 2
# - Load balancer: Application Load Balancer
# - Target group: rag-backend-tg
```

## CloudWatch 모니터링

### 로그 확인
```bash
# CloudWatch Logs로 이동
aws logs tail /ecs/rag-backend --follow

# 특정 기간 로그 조회
aws logs filter-log-events \
  --log-group-name /ecs/rag-backend \
  --start-time $(date -u -d '1 hour ago' +%s)000
```

### 메트릭 확인
- CPU 사용률
- 메모리 사용률
- 네트워크 In/Out
- 헬스체크 상태

### 알람 설정
```bash
# CPU 사용률 80% 초과 시 알람
aws cloudwatch put-metric-alarm \
  --alarm-name rag-backend-high-cpu \
  --alarm-description "Backend CPU > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold
```

## 트러블슈팅

### Task가 계속 재시작되는 경우
1. CloudWatch Logs 확인
2. 헬스체크 설정 확인
3. 환경변수/시크릿 주입 확인
4. 리소스(CPU/Memory) 부족 확인

### API Key 관련 오류
1. Secrets Manager 시크릿 이름 확인
2. Task Role IAM 권한 확인
3. 시크릿 ARN이 정확한지 확인

### 네트워크 연결 실패
1. Security Group 설정 확인
2. Subnet이 Public인지 확인
3. NAT Gateway 설정 확인 (Private Subnet 사용 시)

## 비용 최적화

### Fargate Spot 사용
- On-Demand 대비 최대 70% 할인
- 중단 가능한 워크로드에 적합

### Auto Scaling 설정
```bash
# Target Tracking Scaling Policy
# - Target: CPU 70%
# - Min tasks: 2
# - Max tasks: 10
```

## 완료 체크리스트
- [ ] AWS CLI가 설정되었는가?
- [ ] ECR에 이미지가 푸시되었는가?
- [ ] Secrets Manager에 API Key가 저장되었는가?
- [ ] ECS Cluster가 생성되었는가?
- [ ] Task Definition이 등록되었는가?
- [ ] ALB와 Target Group이 설정되었는가?
- [ ] ECS Service가 실행 중인가?
- [ ] Public URL로 접속이 가능한가?
- [ ] CloudWatch Logs에서 로그가 확인되는가?
- [ ] 헬스체크가 통과하는가?
