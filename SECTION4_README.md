# Section 4: GCP Cloud Run 배포

## 학습 목표
- GCP 컨테이너 서비스를 AWS와 비교하며 이해
- Artifact Registry에 이미지 업로드
- Cloud Run으로 서버리스 컨테이너 배포
- Secret Manager로 시크릿 관리
- Cloud Logging 모니터링

## 새로 추가된 파일
```
scripts/
  └── gcp-deploy.sh                   # GCP 배포 자동화 스크립트

docs/gcp/
  ├── cloud-run-backend.yaml         # Backend Cloud Run 서비스 설정
  └── cloud-run-frontend.yaml        # Frontend Cloud Run 서비스 설정
```

## AWS ↔ GCP 서비스 매핑

| AWS Service | GCP Service | 용도 |
|-------------|-------------|------|
| ECR | Artifact Registry | 컨테이너 이미지 저장소 |
| ECS/Fargate | Cloud Run | 컨테이너 실행 |
| Secrets Manager | Secret Manager | 시크릿 관리 |
| CloudWatch Logs | Cloud Logging | 로그 수집/분석 |
| CloudWatch Metrics | Cloud Monitoring | 메트릭 모니터링 |
| ALB | Cloud Load Balancing | 로드 밸런싱 |

## 배포 단계

### 1. gcloud CLI 설정
```bash
# gcloud CLI 설치 확인
gcloud version

# 인증
gcloud auth login

# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID

# 기본 리전 설정
gcloud config set run/region asia-northeast3
```

### 2. 필요한 API 활성화
```bash
# Cloud Run API
gcloud services enable run.googleapis.com

# Artifact Registry API
gcloud services enable artifactregistry.googleapis.com

# Secret Manager API
gcloud services enable secretmanager.googleapis.com
```

### 3. Secret Manager에 API Key 저장
```bash
# OpenAI API Key 저장
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key \
  --data-file=- \
  --replication-policy="automatic"

# Pinecone API Key 저장
echo -n "your-pinecone-api-key" | gcloud secrets create pinecone-api-key \
  --data-file=- \
  --replication-policy="automatic"

# Secret 접근 권한 부여 (Cloud Run Service Account에)
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### 4. Artifact Registry에 이미지 푸시
```bash
# 배포 스크립트 실행
export GCP_PROJECT_ID=YOUR_PROJECT_ID
chmod +x scripts/gcp-deploy.sh
./scripts/gcp-deploy.sh latest
```

### 5. Cloud Run 서비스 배포

#### Backend 배포
```bash
gcloud run deploy rag-backend \
  --image asia-northeast3-docker.pkg.dev/YOUR_PROJECT_ID/rag-demo/rag-backend:latest \
  --region asia-northeast3 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO" \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest" \
  --cpu 2 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300 \
  --port 8000
```

#### Frontend 배포
```bash
gcloud run deploy rag-frontend \
  --image asia-northeast3-docker.pkg.dev/YOUR_PROJECT_ID/rag-demo/rag-frontend:latest \
  --region asia-northeast3 \
  --platform managed \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 5 \
  --port 80
```

### 6. 배포 확인
```bash
# 서비스 URL 확인
gcloud run services describe rag-backend \
  --region asia-northeast3 \
  --format="value(status.url)"

# 브라우저에서 접속
# https://rag-backend-HASH-an.a.run.app
```

## Cloud Run 주요 기능

### Auto Scaling
- **min-instances**: 최소 인스턴스 수 (0 = 완전 서버리스)
- **max-instances**: 최대 인스턴스 수
- **트래픽 없을 때**: 자동으로 0으로 축소 → 비용 0원

### Cold Start 최적화
```yaml
# cloud-run-backend.yaml 설정
annotations:
  run.googleapis.com/startup-cpu-boost: 'true'  # 시작 시 CPU 부스트
startupProbe:
  initialDelaySeconds: 0
  periodSeconds: 3
  failureThreshold: 10
```

### 트래픽 분할 (Canary Deployment)
```bash
# 새 버전 배포 (트래픽 없음)
gcloud run deploy rag-backend \
  --image ...latest \
  --no-traffic

# 트래픽 10%만 새 버전으로
gcloud run services update-traffic rag-backend \
  --to-revisions LATEST=10,PREVIOUS=90

# 문제 없으면 100% 전환
gcloud run services update-traffic rag-backend \
  --to-latest
```

## Cloud Logging 모니터링

### 로그 확인
```bash
# 실시간 로그 스트리밍
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=rag-backend" --format=json

# 최근 1시간 로그 조회
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-backend" \
  --limit 50 \
  --format json \
  --freshness 1h
```

### 로그 쿼리 (Cloud Console)
```
resource.type="cloud_run_revision"
resource.labels.service_name="rag-backend"
severity="ERROR"
```

### 메트릭 대시보드
- Request Count (요청 수)
- Request Latency (응답 시간)
- Container Instance Count (실행 중인 인스턴스 수)
- Billable Instance Time (과금 시간)

## 비용 최적화

### Cloud Run 비용 구조
```
비용 = (vCPU 시간 × vCPU 가격) + (Memory 시간 × Memory 가격) + (Request 수 × Request 가격)
```

### 최적화 전략
1. **min-instances=0**: 트래픽 없을 때 완전 종료
2. **적절한 리소스 할당**: 과도한 CPU/Memory 할당 피하기
3. **Request timeout 최적화**: 불필요하게 긴 timeout 피하기
4. **Cold Start 허용**: 비용 vs 성능 트레이드오프 고려

### 예상 비용 (서울 리전)
- vCPU: $0.00002400/vCPU-second
- Memory: $0.00000250/GiB-second
- Requests: $0.40/million requests
- 무료 할당량: 월 200만 요청, 36만 vCPU-second, 10만 GiB-second

## 트러블슈팅

### Cold Start 지연
- **증상**: 첫 요청이 느림 (2-5초)
- **해결**:
  - min-instances=1 설정 (비용 증가)
  - startup-cpu-boost 활성화
  - 이미지 경량화

### Secret 접근 실패
- **증상**: "Permission denied" 오류
- **해결**:
  - Secret Manager API 활성화 확인
  - Service Account IAM 권한 확인
  - Secret 이름 정확한지 확인

### 503 Service Unavailable
- **증상**: 서비스 접속 불가
- **해결**:
  - 컨테이너 로그 확인
  - Health check 설정 확인
  - max-instances 제한 확인

## 완료 체크리스트
- [ ] gcloud CLI가 설정되었는가?
- [ ] Artifact Registry에 이미지가 푸시되었는가?
- [ ] Secret Manager에 API Key가 저장되었는가?
- [ ] Cloud Run 서비스가 배포되었는가?
- [ ] Public URL로 접속이 가능한가?
- [ ] Cloud Logging에서 로그가 확인되는가?
- [ ] Auto Scaling이 동작하는가?
- [ ] Cold Start 시간이 허용 범위 내인가?
- [ ] 트래픽 분할 테스트를 해보았는가?
- [ ] 비용 모니터링을 설정했는가?
