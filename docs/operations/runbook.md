# 운영 런북 (Operations Runbook)

## 📚 목차
1. [일상 운영 절차](#일상-운영-절차)
2. [장애 대응](#장애-대응)
3. [배포 관리](#배포-관리)
4. [성능 모니터링](#성능-모니터링)
5. [보안 관리](#보안-관리)
6. [비용 최적화](#비용-최적화)

---

## 일상 운영 절차

### 매일 체크리스트

#### 1. 서비스 헬스 체크
```bash
# AWS ECS 서비스 상태 확인
aws ecs describe-services \
  --cluster rag-demo-cluster \
  --services rag-backend-service rag-frontend-service \
  --region ap-northeast-2 \
  --query 'services[*].[serviceName,status,runningCount,desiredCount]' \
  --output table

# GCP Cloud Run 서비스 상태 확인
gcloud run services list \
  --region asia-northeast3 \
  --format="table(SERVICE,STATUS,URL)"
```

**정상 상태:**
- Status: ACTIVE
- Running Count = Desired Count
- Health check: 200 OK

#### 2. 로그 모니터링
```bash
# AWS CloudWatch 최근 에러 로그
aws logs filter-log-events \
  --log-group-name /ecs/rag-backend \
  --filter-pattern "ERROR" \
  --start-time $(date -u -d '1 hour ago' +%s)000 \
  --region ap-northeast-2

# GCP Cloud Run 에러 로그
gcloud logging read "resource.type=cloud_run_revision \
  AND severity>=ERROR" \
  --limit 50 \
  --format json
```

**주의 사항:**
- ERROR 로그 급증 → 즉시 조사
- WARN 로그 패턴 → 잠재적 문제 파악
- 반복되는 에러 → 근본 원인 분석

#### 3. 리소스 사용량 확인
```bash
# AWS ECS 태스크 CPU/Memory 사용량
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=rag-backend-service \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average \
  --region ap-northeast-2

# GCP Cloud Run 인스턴스 수
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/instance_count"' \
  --format="table(resource.labels.service_name,points[0].value.int64_value)"
```

**임계값:**
- CPU: 평균 70% 이하 유지
- Memory: 80% 이하 유지
- 지속적으로 높으면 스케일업 고려

#### 4. 비용 모니터링
```bash
# AWS 일일 비용 확인
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '1 day ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=SERVICE

# GCP 프로젝트 비용 확인
gcloud billing projects describe $(gcloud config get-value project) \
  --format="table(billingAccountName,billingEnabled)"
```

### 주간 체크리스트

#### 1. 백업 확인
```bash
# RDS 스냅샷 확인
aws rds describe-db-snapshots \
  --db-instance-identifier rag-demo-db \
  --region ap-northeast-2

# 스냅샷 수동 생성
aws rds create-db-snapshot \
  --db-instance-identifier rag-demo-db \
  --db-snapshot-identifier rag-demo-manual-$(date +%Y%m%d)
```

#### 2. 보안 업데이트
```bash
# Docker 이미지 취약점 스캔
docker scout cves <image-name>

# ECR 이미지 스캔 결과 확인
aws ecr describe-image-scan-findings \
  --repository-name rag-backend \
  --image-id imageTag=latest \
  --region ap-northeast-2
```

#### 3. 성능 리포트 생성
```bash
# CloudWatch 대시보드 스크린샷
# GCP Monitoring 대시보드 리뷰
# 주간 성능 트렌드 분석
```

### 월간 체크리스트

#### 1. 비용 최적화 리뷰
- 미사용 리소스 식별 및 제거
- 예약 인스턴스 / Committed Use 검토
- 스토리지 라이프사이클 정책 확인

#### 2. 보안 감사
- IAM 권한 리뷰
- 시크릿 로테이션
- 방화벽 규칙 검토

#### 3. DR 테스트
- 백업 복원 테스트
- 장애 복구 시나리오 실행
- 런북 업데이트

---

## 장애 대응

### 장애 등급 정의

| 등급 | 정의 | 대응 시간 | 예시 |
|------|------|----------|------|
| P0 - Critical | 서비스 전체 다운 | 즉시 (15분 이내) | API 완전 불통 |
| P1 - High | 핵심 기능 장애 | 1시간 이내 | 검색 기능 오류 |
| P2 - Medium | 부분 기능 장애 | 4시간 이내 | 특정 엔드포인트 느림 |
| P3 - Low | 경미한 문제 | 1일 이내 | UI 표시 오류 |

### 장애 대응 절차

#### Step 1: 장애 감지 및 확인
```bash
# 1. Health check 상태 확인
curl -f https://api.example.com/health || echo "Health check failed"

# 2. 서비스 상태 확인 (AWS)
aws ecs describe-services \
  --cluster rag-demo-cluster \
  --services rag-backend-service

# 3. 최근 배포 확인
git log -1 --oneline
```

#### Step 2: 영향 범위 파악
```bash
# 1. 에러 로그 확인
aws logs tail /ecs/rag-backend --follow --since 10m

# 2. 메트릭 확인
# - 요청 수 변화
# - 에러율 증가
# - 응답 시간 증가

# 3. 사용자 영향 파악
# - 에러 발생 사용자 수
# - 영향받는 기능
```

#### Step 3: 긴급 복구
```bash
# Option 1: 이전 버전으로 롤백
git revert HEAD
git push origin main

# Option 2: 트래픽 차단 (임시)
# AWS ALB에서 503 반환 설정
# 또는 Cloud Run 트래픽 0으로 설정

# Option 3: 스케일 조정
aws ecs update-service \
  --cluster rag-demo-cluster \
  --service rag-backend-service \
  --desired-count 5
```

#### Step 4: 근본 원인 분석
```bash
# 1. 상세 로그 분석
aws logs get-log-events \
  --log-group-name /ecs/rag-backend \
  --log-stream-name <stream-name>

# 2. 메트릭 상관관계 분석
# - 배포 시간과 장애 시간 비교
# - CPU/Memory spike 확인
# - 외부 의존성 문제 확인

# 3. 코드 변경 리뷰
git diff HEAD~1 HEAD
```

#### Step 5: 영구 수정 및 문서화
```bash
# 1. 수정 코드 개발 및 테스트
# 2. PR 생성 및 리뷰
# 3. 배포 후 모니터링
# 4. 포스트모템 작성
```

### 일반적인 장애 시나리오

#### 시나리오 1: 503 Service Unavailable

**증상:**
```bash
$ curl https://api.example.com/health
503 Service Unavailable
```

**원인 및 해결:**
1. **모든 태스크가 Unhealthy**
   ```bash
   # Health check 로그 확인
   aws logs tail /ecs/rag-backend --filter-pattern "health"

   # 해결: Health check 엔드포인트 수정 또는 타임아웃 증가
   ```

2. **Target Group에 등록된 타겟 없음**
   ```bash
   # Target Group 상태 확인
   aws elbv2 describe-target-health \
     --target-group-arn <arn>

   # 해결: ECS 서비스 desired count 증가
   ```

3. **컨테이너 크래시 반복**
   ```bash
   # 크래시 로그 확인
   aws ecs describe-tasks --cluster rag-demo-cluster --tasks <task-id>

   # 해결: 애플리케이션 코드 수정 또는 리소스 증가
   ```

#### 시나리오 2: 높은 응답 시간

**증상:**
```bash
$ time curl https://api.example.com/api/query
# 10초 이상 소요
```

**원인 및 해결:**
1. **CPU/Memory 과부하**
   ```bash
   # 리소스 사용량 확인
   aws cloudwatch get-metric-statistics ...

   # 해결: 스케일아웃 또는 인스턴스 크기 증가
   aws ecs update-service --desired-count 10
   ```

2. **데이터베이스 느림**
   ```bash
   # RDS 모니터링
   aws rds describe-db-instances --db-instance-identifier rag-demo-db

   # 해결: 쿼리 최적화, 인덱스 추가, Read Replica 추가
   ```

3. **외부 API 타임아웃**
   ```bash
   # 애플리케이션 로그에서 타임아웃 확인

   # 해결: 타임아웃 설정 조정, 재시도 로직 개선, Circuit Breaker 패턴 적용
   ```

#### 시나리오 3: 메모리 부족 (OOM)

**증상:**
```bash
# ECS 태스크가 계속 재시작됨
Task stopped at: 2024-01-01T12:00:00Z
Stop reason: Essential container in task exited
Exit code: 137 (OOM killed)
```

**해결:**
1. **메모리 제한 증가**
   ```bash
   # Task Definition 업데이트
   # memory: 512 → 1024
   ```

2. **메모리 누수 조사**
   ```python
   # 애플리케이션에서 메모리 프로파일링
   import tracemalloc
   tracemalloc.start()
   # 코드 실행
   snapshot = tracemalloc.take_snapshot()
   ```

3. **캐시 크기 조정**
   ```python
   # LRU 캐시 크기 제한
   from functools import lru_cache
   @lru_cache(maxsize=100)  # 기존 1000에서 감소
   ```

#### 시나리오 4: 배포 실패

**증상:**
```bash
# GitHub Actions 배포 Job 실패
Error: Service rag-backend-service did not stabilize
```

**해결:**
1. **롤링 업데이트 실패**
   ```bash
   # 이전 Task Definition으로 롤백
   aws ecs update-service \
     --cluster rag-demo-cluster \
     --service rag-backend-service \
     --task-definition rag-backend:10  # 이전 버전
   ```

2. **Health check 실패**
   ```bash
   # Health check 설정 확인
   # - 경로: /health
   # - 타임아웃: 5초
   # - Interval: 30초

   # 임시 해결: Health check 비활성화 (주의!)
   ```

3. **이미지 pull 실패**
   ```bash
   # ECR 권한 확인
   aws ecr get-login-password --region ap-northeast-2

   # 이미지 존재 확인
   aws ecr describe-images \
     --repository-name rag-backend \
     --image-ids imageTag=latest
   ```

---

## 배포 관리

### 정상 배포 절차

#### 1. 배포 전 체크리스트
- [ ] 코드 리뷰 완료
- [ ] 모든 테스트 통과
- [ ] Staging 환경 배포 및 검증
- [ ] 배포 영향 범위 파악
- [ ] 롤백 계획 수립
- [ ] 모니터링 대시보드 준비

#### 2. 배포 실행
```bash
# 수동 배포 (필요시)
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3

# GitHub Actions 워크플로우 수동 실행
# GitHub → Actions → Deploy → Run workflow
```

#### 3. 배포 후 검증
```bash
# 1. Health check
curl https://api.example.com/health

# 2. 기능 테스트
curl -X POST https://api.example.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# 3. 로그 모니터링 (10분간)
aws logs tail /ecs/rag-backend --follow

# 4. 메트릭 확인
# - 에러율 정상
# - 응답 시간 정상
# - CPU/Memory 정상
```

### 롤백 절차

#### Quick Rollback (긴급)
```bash
# Option 1: Git Revert
git revert HEAD
git push origin main

# Option 2: 이전 Task Definition으로 복구 (AWS)
aws ecs update-service \
  --cluster rag-demo-cluster \
  --service rag-backend-service \
  --task-definition rag-backend:10

# Option 3: 이전 revision으로 복구 (GCP)
gcloud run services update-traffic rag-backend \
  --to-revisions=rag-backend-00010-abc=100 \
  --region asia-northeast3
```

#### Gradual Rollback (단계적)
```bash
# 트래픽 점진적 이동 (Canary)
# AWS ALB weighted target groups 사용
# 또는 GCP Cloud Run traffic splitting

# 예: 90% → 이전 버전, 10% → 새 버전
gcloud run services update-traffic rag-backend \
  --to-revisions=rag-backend-00010-abc=90,rag-backend-00011-xyz=10 \
  --region asia-northeast3

# 문제 없으면 점진적으로 비율 조정
# 100% → 이전 버전
```

### Blue-Green 배포

#### AWS ECS 방식
```bash
# 1. 새로운 Task Definition 배포 (Green)
# 2. Target Group 생성 (Green)
# 3. ALB Listener Rule 업데이트 (Green으로 트래픽 전환)
# 4. 검증 완료 후 Blue 환경 제거
```

#### GCP Cloud Run 방식
```bash
# 1. 새로운 revision 배포 (no-traffic)
gcloud run deploy rag-backend \
  --image=$IMAGE \
  --no-traffic

# 2. 트래픽 전환 테스트 (10%)
gcloud run services update-traffic rag-backend \
  --to-revisions=LATEST=10

# 3. 검증 후 100% 전환
gcloud run services update-traffic rag-backend \
  --to-latest

# 4. 이전 revision 제거 (선택)
gcloud run revisions delete <old-revision>
```

---

## 성능 모니터링

### 핵심 메트릭

#### 1. 애플리케이션 메트릭
```yaml
# 추적해야 할 메트릭
request_count: 초당 요청 수
error_rate: 에러율 (%)
response_time_p50: 50th percentile 응답 시간
response_time_p95: 95th percentile 응답 시간
response_time_p99: 99th percentile 응답 시간
```

**정상 범위:**
- Error rate: < 1%
- P50 response time: < 200ms
- P95 response time: < 500ms
- P99 response time: < 1000ms

#### 2. 인프라 메트릭
```yaml
cpu_utilization: CPU 사용률 (%)
memory_utilization: 메모리 사용률 (%)
network_in: 네트워크 입력 (bytes)
network_out: 네트워크 출력 (bytes)
disk_io: 디스크 I/O
```

**정상 범위:**
- CPU: < 70% (평균)
- Memory: < 80%
- Network: 대역폭 한계의 70% 이내

#### 3. 비즈니스 메트릭
```yaml
active_users: 활성 사용자 수
query_count: 검색 쿼리 수
query_success_rate: 검색 성공률
average_query_time: 평균 쿼리 시간
```

### 알림 설정

#### CloudWatch Alarms (AWS)
```bash
# CPU 사용률 알림
aws cloudwatch put-metric-alarm \
  --alarm-name rag-backend-high-cpu \
  --alarm-description "Alert when CPU exceeds 70%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 70 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=ServiceName,Value=rag-backend-service

# 에러율 알림
aws cloudwatch put-metric-alarm \
  --alarm-name rag-backend-high-error-rate \
  --metric-name 4XXError \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 60 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

#### GCP Monitoring Alerts
```bash
# Uptime check 생성
gcloud monitoring uptime create \
  --display-name="RAG Backend Health" \
  --resource-type=uptime-url \
  --host=rag-backend-xyz.run.app \
  --path=/health

# Alert policy 생성 (YAML)
cat > alert-policy.yaml <<EOF
displayName: "High Error Rate"
conditions:
  - displayName: "Error rate > 5%"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count"'
      comparison: COMPARISON_GT
      thresholdValue: 0.05
      duration: 60s
notificationChannels:
  - projects/PROJECT_ID/notificationChannels/CHANNEL_ID
EOF

gcloud alpha monitoring policies create --policy-from-file=alert-policy.yaml
```

### 대시보드 구성

#### CloudWatch Dashboard (AWS)
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", {"stat": "Average"}],
          [".", "MemoryUtilization"]
        ],
        "period": 300,
        "region": "ap-northeast-2",
        "title": "ECS Resources"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ApplicationELB", "TargetResponseTime", {"stat": "p99"}]
        ],
        "period": 60,
        "title": "Response Time P99"
      }
    }
  ]
}
```

#### GCP Monitoring Dashboard
- Cloud Run request count
- Cloud Run request latencies
- Cloud Run instance count
- Cloud Run billable instance time

---

## 보안 관리

### 시크릿 로테이션

#### AWS Secrets Manager
```bash
# 1. 새로운 API 키 생성
NEW_KEY="new-openai-api-key-value"

# 2. Secret 업데이트
aws secretsmanager update-secret \
  --secret-id openai-api-key \
  --secret-string "$NEW_KEY" \
  --region ap-northeast-2

# 3. ECS 서비스 재배포 (새 시크릿 적용)
aws ecs update-service \
  --cluster rag-demo-cluster \
  --service rag-backend-service \
  --force-new-deployment
```

#### GCP Secret Manager
```bash
# 1. 새 버전 추가
echo -n "new-api-key-value" | \
  gcloud secrets versions add openai-api-key --data-file=-

# 2. Cloud Run 서비스 재배포
gcloud run services update rag-backend \
  --region asia-northeast3 \
  --update-secrets=OPENAI_API_KEY=openai-api-key:latest
```

### 취약점 스캔

#### Docker 이미지 스캔
```bash
# Trivy로 로컬 스캔
trivy image --severity HIGH,CRITICAL \
  rag-backend:latest

# ECR 자동 스캔 활성화
aws ecr put-image-scanning-configuration \
  --repository-name rag-backend \
  --image-scanning-configuration scanOnPush=true

# 스캔 결과 확인
aws ecr describe-image-scan-findings \
  --repository-name rag-backend \
  --image-id imageTag=latest
```

#### 의존성 스캔
```bash
# Python 의존성 취약점 스캔
pip install safety
safety check -r requirements.txt

# GitHub Dependabot 활성화
# Settings → Security → Dependabot alerts
```

### 접근 제어

#### IAM 최소 권한 원칙
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeServices",
        "ecs:DescribeTasks"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Deny",
      "Action": [
        "ecs:DeleteCluster",
        "ecs:DeleteService"
      ],
      "Resource": "*"
    }
  ]
}
```

#### 네트워크 보안
```bash
# Security Group 규칙 최소화
aws ec2 describe-security-groups \
  --group-ids sg-xxx

# 불필요한 포트 차단
# Only allow:
# - 443 (HTTPS) from ALB
# - 8000 (App) from ALB security group
```

---

## 비용 최적화

### 비용 절감 전략

#### 1. 컴퓨팅 최적화
```bash
# Fargate Spot 사용 (최대 70% 할인)
# Task Definition에서 capacityProviderStrategy 설정

# GCP Cloud Run 최소 인스턴스 0으로 설정
gcloud run services update rag-backend \
  --min-instances=0  # 트래픽 없을 때 비용 0

# 오토스케일링 정책 최적화
# - 적절한 CPU/Memory 임계값 설정
# - Scale-down cooldown 시간 조정
```

#### 2. 스토리지 최적화
```bash
# ECR 이미지 라이프사이클 정책
cat > lifecycle-policy.json <<EOF
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {
        "tagStatus": "any",
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}
EOF

aws ecr put-lifecycle-policy \
  --repository-name rag-backend \
  --lifecycle-policy-text file://lifecycle-policy.json

# CloudWatch Logs 보존 기간 설정
aws logs put-retention-policy \
  --log-group-name /ecs/rag-backend \
  --retention-in-days 30
```

#### 3. 네트워크 최적화
```bash
# NAT Gateway 대신 NAT Instance 사용 (소규모)
# 또는 VPC Endpoint 사용 (S3, ECR 등)

# CloudFront 캐싱 활용
# - Static assets
# - API responses (cache-control 헤더 활용)
```

### 비용 모니터링

#### AWS Cost Explorer
```bash
# 일일 비용 확인
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=SERVICE

# 비용 예측
aws ce get-cost-forecast \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --metric UNBLENDED_COST \
  --granularity MONTHLY
```

#### GCP Billing
```bash
# 프로젝트 비용 확인
gcloud billing accounts list

# BigQuery로 상세 비용 분석
bq query --use_legacy_sql=false '
SELECT
  service.description,
  SUM(cost) as total_cost
FROM `billing-export.gcp_billing_export_v1_XXX`
WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY service.description
ORDER BY total_cost DESC
'
```

### 예산 알림 설정

#### AWS Budget
```bash
aws budgets create-budget \
  --account-id 123456789012 \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

#### GCP Budget Alert
```bash
# Google Cloud Console
# Billing → Budgets & alerts → Create budget
# - Set budget amount
# - Configure alert thresholds (50%, 90%, 100%)
# - Add notification emails
```

---

## 📞 에스컬레이션

### 연락처
- **DevOps 팀**: devops@example.com
- **인프라 담당**: infra@example.com
- **보안 팀**: security@example.com
- **On-call**: +82-10-xxxx-xxxx

### 에스컬레이션 기준
- P0 장애 15분 내 미해결
- P1 장애 1시간 내 미해결
- 보안 사고 즉시 에스컬레이션
- 비정상적인 비용 급증

---

## 📚 참고 자료

### AWS
- [ECS Troubleshooting](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/troubleshooting.html)
- [CloudWatch Logs Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AnalyzingLogData.html)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

### GCP
- [Cloud Run Troubleshooting](https://cloud.google.com/run/docs/troubleshooting)
- [Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [GCP Best Practices](https://cloud.google.com/docs/enterprise/best-practices-for-enterprise-organizations)

### 일반
- [SRE Book (Google)](https://sre.google/sre-book/table-of-contents/)
- [The Site Reliability Workbook](https://sre.google/workbook/table-of-contents/)
- [12 Factor App](https://12factor.net/)
