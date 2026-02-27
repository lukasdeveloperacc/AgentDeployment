# Section 7: 운영 및 모니터링

## 📚 학습 목표

이번 섹션에서는 프로덕션 환경의 운영과 모니터링 방법을 학습합니다.

- 로그 수집 및 분석 방법 이해
- 메트릭 모니터링 및 대시보드 구성
- 알림(Alert) 설정 및 관리
- 장애 대응 절차 및 복구 방법
- 성능 최적화 및 튜닝
- 보안 관리 및 컴플라이언스

## 🎯 Section 7에서 다루는 내용

### 1. 로그 관리
- 중앙 집중식 로그 수집
- 로그 검색 및 분석
- 로그 보존 정책

### 2. 메트릭 모니터링
- 핵심 성능 지표(KPI)
- 대시보드 구성
- 트렌드 분석

### 3. 알림 시스템
- Alert 정책 설정
- 알림 채널 구성
- On-call 관리

### 4. 장애 대응
- 장애 감지 및 분류
- 대응 절차
- 포스트모템

### 5. 성능 최적화
- 병목 지점 식별
- 리소스 최적화
- 비용 효율화

### 6. 보안 운영
- 시크릿 관리
- 취약점 스캔
- 접근 제어

---

## 📊 로그 관리

### AWS CloudWatch Logs

#### 로그 그룹 구조
```
/ecs/rag-backend         # Backend 애플리케이션 로그
/ecs/rag-frontend        # Frontend 애플리케이션 로그
/aws/lambda/xxx          # Lambda 함수 로그 (사용시)
/aws/rds/xxx             # RDS 로그 (사용시)
```

#### 로그 확인
```bash
# 실시간 로그 스트리밍
aws logs tail /ecs/rag-backend --follow

# 특정 시간대 로그 조회
aws logs filter-log-events \
  --log-group-name /ecs/rag-backend \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --end-time $(date +%s)000

# 에러 로그만 필터링
aws logs filter-log-events \
  --log-group-name /ecs/rag-backend \
  --filter-pattern "ERROR"

# 특정 패턴 검색
aws logs filter-log-events \
  --log-group-name /ecs/rag-backend \
  --filter-pattern '[time, request_id, level = ERROR, msg]'
```

#### CloudWatch Logs Insights

**쿼리 예시 1: 에러 빈도 분석**
```sql
fields @timestamp, @message
| filter @message like /ERROR/
| stats count() by bin(5m)
| sort @timestamp desc
```

**쿼리 예시 2: 느린 요청 분석**
```sql
fields @timestamp, @message
| parse @message /response_time: (?<duration>\d+)ms/
| filter duration > 1000
| stats avg(duration), max(duration), count() by bin(5m)
```

**쿼리 예시 3: 사용자별 요청 분석**
```sql
fields @timestamp, @message
| parse @message /user_id: (?<user>[^\s]+)/
| stats count() by user
| sort count desc
| limit 20
```

**쿼리 예시 4: API 엔드포인트별 성능**
```sql
fields @timestamp, @message
| parse @message /GET (?<endpoint>\/api\/[^\s]+) .*response_time: (?<duration>\d+)ms/
| stats avg(duration), count() by endpoint
| sort avg desc
```

#### 로그 보존 정책 설정
```bash
# 30일 보존 설정
aws logs put-retention-policy \
  --log-group-name /ecs/rag-backend \
  --retention-in-days 30

# 보존 기간 옵션: 1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
```

### GCP Cloud Logging

#### 로그 뷰어 사용
```bash
# 최근 로그 조회
gcloud logging read "resource.type=cloud_run_revision" \
  --limit 50 \
  --format json

# 에러 로그 필터링
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" \
  --limit 50 \
  --format json

# 특정 서비스 로그
gcloud logging read "resource.type=cloud_run_revision \
  AND resource.labels.service_name=rag-backend" \
  --limit 50 \
  --format json

# 시간 범위 지정
gcloud logging read "resource.type=cloud_run_revision" \
  --freshness=1h \
  --limit 100
```

#### 로그 기반 메트릭 생성
```bash
# 에러 카운트 메트릭
gcloud logging metrics create error_count \
  --description="Count of error logs" \
  --log-filter='resource.type="cloud_run_revision" AND severity>=ERROR'

# 특정 패턴 메트릭
gcloud logging metrics create slow_query_count \
  --description="Count of slow queries" \
  --log-filter='resource.type="cloud_run_revision" AND jsonPayload.duration>1000'
```

#### 로그 익스포트 (BigQuery)
```bash
# BigQuery로 로그 익스포트 설정
gcloud logging sinks create rag-logs-export \
  bigquery.googleapis.com/projects/PROJECT_ID/datasets/logs_dataset \
  --log-filter='resource.type="cloud_run_revision"'
```

#### 로그 분석 쿼리 (BigQuery)
```sql
-- 시간대별 에러 분포
SELECT
  TIMESTAMP_TRUNC(timestamp, HOUR) as hour,
  COUNT(*) as error_count
FROM `PROJECT_ID.logs_dataset.cloud_run_revision_*`
WHERE severity = 'ERROR'
GROUP BY hour
ORDER BY hour DESC

-- API 엔드포인트별 평균 응답시간
SELECT
  httpRequest.requestUrl as endpoint,
  AVG(CAST(jsonPayload.duration AS INT64)) as avg_duration_ms,
  COUNT(*) as request_count
FROM `PROJECT_ID.logs_dataset.cloud_run_revision_*`
WHERE httpRequest.requestUrl IS NOT NULL
GROUP BY endpoint
ORDER BY avg_duration_ms DESC
```

---

## 📈 메트릭 모니터링

### 핵심 메트릭 (Golden Signals)

#### 1. Latency (지연시간)
응답 시간의 분포:
- **P50**: 50% 사용자 경험
- **P95**: 95% 사용자 경험
- **P99**: 99% 사용자 경험 (꼬리 지연)

**목표값:**
- P50: < 200ms
- P95: < 500ms
- P99: < 1000ms

#### 2. Traffic (트래픽)
시스템 부하:
- 초당 요청 수 (RPS)
- 동시 연결 수
- 대역폭 사용량

**모니터링:**
- 평상시 트래픽 패턴 파악
- 급격한 증가/감소 감지
- 용량 계획

#### 3. Errors (에러)
실패율:
- 4XX 에러 (클라이언트 에러)
- 5XX 에러 (서버 에러)
- 에러율 (%)

**목표값:**
- 전체 에러율: < 1%
- 5XX 에러율: < 0.1%

#### 4. Saturation (포화도)
리소스 사용률:
- CPU 사용률
- 메모리 사용률
- 네트워크 사용률
- 스토리지 사용률

**목표값:**
- CPU: < 70% (평균)
- Memory: < 80%
- Network: < 70%

### AWS CloudWatch 대시보드

#### 대시보드 생성
```bash
# 대시보드 JSON 파일 생성
aws cloudwatch put-dashboard \
  --dashboard-name rag-demo-dashboard \
  --dashboard-body file://docs/monitoring/cloudwatch-dashboard.json
```

**대시보드 구성 요소:**
1. **CPU/Memory 사용률** - 리소스 모니터링
2. **Response Time** - P50, P95, P99 지연시간
3. **Request Count** - 트래픽 추이
4. **Error Count** - 4XX, 5XX 에러
5. **Task Count** - 실행 중인 태스크 수
6. **Recent Errors** - 최근 에러 로그

#### 커스텀 메트릭 발행
```python
# backend/app/monitoring.py
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch', region_name='ap-northeast-2')

def publish_metric(metric_name, value, unit='None'):
    """CloudWatch에 커스텀 메트릭 발행"""
    cloudwatch.put_metric_data(
        Namespace='RAG/Application',
        MetricData=[
            {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow(),
                'Dimensions': [
                    {
                        'Name': 'Environment',
                        'Value': 'production'
                    }
                ]
            }
        ]
    )

# 사용 예시
publish_metric('QueryLatency', 150, 'Milliseconds')
publish_metric('CacheHitRate', 85, 'Percent')
publish_metric('ActiveUsers', 42, 'Count')
```

### GCP Cloud Monitoring

#### 대시보드 생성
```bash
# Cloud Console에서 대시보드 생성
# Monitoring → Dashboards → Create Dashboard

# 주요 위젯:
# - Cloud Run Request Count
# - Cloud Run Request Latencies
# - Cloud Run Instance Count
# - Cloud Run Billable Instance Time
```

#### 커스텀 메트릭 발행
```python
# backend/app/monitoring.py
from google.cloud import monitoring_v3
import time

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{PROJECT_ID}"

def publish_metric(metric_type, value):
    """Cloud Monitoring에 커스텀 메트릭 발행"""
    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/{metric_type}"
    series.resource.type = "cloud_run_revision"
    series.resource.labels["project_id"] = PROJECT_ID
    series.resource.labels["service_name"] = "rag-backend"

    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10 ** 9)
    interval = monitoring_v3.TimeInterval(
        {"end_time": {"seconds": seconds, "nanos": nanos}}
    )

    point = monitoring_v3.Point(
        {"interval": interval, "value": {"double_value": value}}
    )
    series.points = [point]

    client.create_time_series(name=project_name, time_series=[series])

# 사용 예시
publish_metric("query_latency", 150)
publish_metric("cache_hit_rate", 0.85)
```

---

## 🚨 알림 시스템

### 알림 정책 설정

#### AWS CloudWatch Alarms

**CPU 사용률 알림:**
```bash
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
  --dimensions Name=ServiceName,Value=rag-backend-service \
  --alarm-actions arn:aws:sns:ap-northeast-2:ACCOUNT_ID:devops-alerts
```

**메모리 사용률 알림:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name rag-backend-high-memory \
  --alarm-description "Alert when memory exceeds 80%" \
  --metric-name MemoryUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=ServiceName,Value=rag-backend-service
```

**에러율 알림:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name rag-alb-high-error-rate \
  --alarm-description "Alert when 5XX errors exceed 10 per minute" \
  --metric-name HTTPCode_Target_5XX_Count \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 60 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:ap-northeast-2:ACCOUNT_ID:critical-alerts
```

**응답 시간 알림:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name rag-alb-slow-response \
  --alarm-description "Alert when P99 latency exceeds 1 second" \
  --metric-name TargetResponseTime \
  --namespace AWS/ApplicationELB \
  --extended-statistic p99 \
  --period 60 \
  --threshold 1.0 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3
```

#### SNS 토픽 생성 및 구독
```bash
# SNS 토픽 생성
aws sns create-topic \
  --name devops-alerts \
  --region ap-northeast-2

# 이메일 구독 추가
aws sns subscribe \
  --topic-arn arn:aws:sns:ap-northeast-2:ACCOUNT_ID:devops-alerts \
  --protocol email \
  --notification-endpoint devops@example.com

# SMS 구독 추가
aws sns subscribe \
  --topic-arn arn:aws:sns:ap-northeast-2:ACCOUNT_ID:critical-alerts \
  --protocol sms \
  --notification-endpoint "+821012345678"

# Slack 웹훅 구독 (Lambda 필요)
aws sns subscribe \
  --topic-arn arn:aws:sns:ap-northeast-2:ACCOUNT_ID:devops-alerts \
  --protocol https \
  --notification-endpoint https://hooks.slack.com/services/xxx/yyy/zzz
```

#### GCP Cloud Monitoring Alerts

**Uptime Check 생성:**
```bash
# Health check 설정
gcloud monitoring uptime create rag-backend-uptime \
  --display-name="RAG Backend Health Check" \
  --resource-type=uptime-url \
  --host=rag-backend-xyz.run.app \
  --path=/health \
  --port=443 \
  --check-interval=60s \
  --timeout=10s
```

**Alert Policy 생성 (YAML):**
```yaml
# alert-policy.yaml
displayName: "High 5XX Error Rate"
documentation:
  content: "5XX error rate exceeds 5% for 2 minutes"
  mimeType: "text/markdown"
conditions:
  - displayName: "Error rate > 5%"
    conditionThreshold:
      filter: |
        resource.type="cloud_run_revision"
        AND metric.type="run.googleapis.com/request_count"
        AND metric.labels.response_code_class="5xx"
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_RATE
      comparison: COMPARISON_GT
      thresholdValue: 0.05
      duration: 120s
notificationChannels:
  - projects/PROJECT_ID/notificationChannels/CHANNEL_ID
alertStrategy:
  autoClose: 1800s  # 30분 후 자동 종료
```

**Alert 생성:**
```bash
gcloud alpha monitoring policies create \
  --policy-from-file=alert-policy.yaml
```

**Notification Channel 생성:**
```bash
# 이메일 채널
gcloud alpha monitoring channels create \
  --display-name="DevOps Email" \
  --type=email \
  --channel-labels=email_address=devops@example.com

# Slack 채널
gcloud alpha monitoring channels create \
  --display-name="Slack Alerts" \
  --type=slack \
  --channel-labels=url=https://hooks.slack.com/services/xxx/yyy/zzz
```

### 알림 레벨 전략

| 레벨 | 조건 | 대응 시간 | 알림 채널 |
|------|------|----------|----------|
| **Critical** | 서비스 다운, 전체 장애 | 즉시 (15분) | Email + SMS + Slack + PagerDuty |
| **High** | 높은 에러율, 심각한 성능 저하 | 1시간 | Email + Slack |
| **Medium** | 중간 수준 문제 | 4시간 | Email |
| **Low** | 경미한 문제, 트렌드 | 1일 | Email (일일 digest) |

---

## 🛠️ 장애 대응

### 장애 대응 프로세스

#### 1단계: 감지 및 확인
```bash
# Health check 확인
curl -f https://api.example.com/health

# 서비스 상태 확인 (AWS)
aws ecs describe-services \
  --cluster rag-demo-cluster \
  --services rag-backend-service \
  --query 'services[0].[status,runningCount,desiredCount,healthCheckGracePeriodSeconds]'

# 서비스 상태 확인 (GCP)
gcloud run services describe rag-backend \
  --region asia-northeast3 \
  --format="table(status.conditions,status.url)"

# 최근 배포 확인
git log -1 --oneline
```

#### 2단계: 영향 범위 파악
```bash
# 에러 로그 확인
aws logs tail /ecs/rag-backend --since 10m --filter-pattern "ERROR"

# 메트릭 확인
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=rag-backend-service \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum

# 트래픽 패턴 확인
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name RequestCount \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Sum
```

#### 3단계: 긴급 복구

**Option 1: 이전 버전으로 롤백**
```bash
# Git 롤백
git revert HEAD
git push origin main

# 또는 직접 이전 버전으로 배포 (AWS)
aws ecs update-service \
  --cluster rag-demo-cluster \
  --service rag-backend-service \
  --task-definition rag-backend:10

# GCP Cloud Run 롤백
gcloud run services update-traffic rag-backend \
  --to-revisions=rag-backend-00010-abc=100 \
  --region asia-northeast3
```

**Option 2: 스케일 조정**
```bash
# AWS ECS 스케일아웃
aws ecs update-service \
  --cluster rag-demo-cluster \
  --service rag-backend-service \
  --desired-count 10

# GCP Cloud Run 최대 인스턴스 증가
gcloud run services update rag-backend \
  --max-instances 20 \
  --region asia-northeast3
```

**Option 3: 트래픽 차단 (임시)**
```bash
# AWS ALB 타겟 그룹 deregister
aws elbv2 deregister-targets \
  --target-group-arn <arn> \
  --targets Id=<instance-id>

# GCP Cloud Run 트래픽 0으로 설정
gcloud run services update-traffic rag-backend \
  --to-revisions=rag-backend-00010-abc=0 \
  --region asia-northeast3
```

#### 4단계: 근본 원인 분석

**로그 분석:**
```bash
# CloudWatch Logs Insights
aws logs start-query \
  --log-group-name /ecs/rag-backend \
  --start-time $(date -d '2 hours ago' +%s) \
  --end-time $(date +%s) \
  --query-string '
    fields @timestamp, @message
    | filter @message like /ERROR/
    | stats count() by bin(5m)
  '

# 코드 변경 확인
git diff HEAD~1 HEAD

# 의존성 변경 확인
git diff HEAD~1 HEAD -- requirements.txt package.json
```

**메트릭 상관관계 분석:**
- 배포 시간과 장애 발생 시간 비교
- CPU/Memory spike 확인
- 외부 의존성 문제 확인 (API, DB)
- 트래픽 패턴 변화 확인

#### 5단계: 영구 수정 및 포스트모템

**포스트모템 템플릿:**
```markdown
# Incident Postmortem: [제목]

## 요약
- **발생 시간**: 2024-01-01 14:30 KST
- **해결 시간**: 2024-01-01 15:45 KST
- **영향 범위**: 전체 사용자의 30%
- **심각도**: P1 (High)

## 타임라인
- 14:30 - 배포 완료
- 14:35 - 에러율 증가 알림
- 14:40 - 장애 확인 및 대응 시작
- 14:50 - 이전 버전으로 롤백
- 15:00 - 서비스 정상화 확인
- 15:45 - 근본 원인 파악 완료

## 근본 원인
- 새로운 코드에서 데이터베이스 연결 풀 설정 오류
- 동시 연결 수 제한 초과로 인한 연결 실패

## 해결 방법
- 즉시: 이전 버전으로 롤백
- 영구: 연결 풀 크기 증가 및 재시도 로직 추가

## 재발 방지
1. 스테이징 환경에서 부하 테스트 필수화
2. 데이터베이스 연결 모니터링 강화
3. Circuit Breaker 패턴 적용

## 액션 아이템
- [ ] 부하 테스트 자동화 (담당: DevOps, 마감: 1/15)
- [ ] 연결 풀 모니터링 대시보드 추가 (담당: Backend, 마감: 1/10)
- [ ] Circuit Breaker 라이브러리 도입 (담당: Backend, 마감: 1/20)
```

### 일반적인 장애 시나리오 및 해결

**시나리오 1: 503 Service Unavailable**
```bash
# 원인 확인
aws ecs describe-services --cluster rag-demo-cluster --services rag-backend-service

# Health check 실패 확인
aws logs filter-log-events \
  --log-group-name /ecs/rag-backend \
  --filter-pattern "health"

# 해결:
# 1. Health check 엔드포인트 수정
# 2. Health check 타임아웃 증가
# 3. Desired count 증가
```

**시나리오 2: 높은 응답 시간**
```bash
# CPU/Memory 확인
aws cloudwatch get-metric-statistics ...

# 해결:
# 1. 스케일아웃 (인스턴스 수 증가)
# 2. 스케일업 (인스턴스 크기 증가)
# 3. 코드 최적화 (쿼리, 알고리즘)
# 4. 캐싱 추가
```

**시나리오 3: 메모리 부족 (OOM)**
```bash
# Task 종료 로그 확인
# Exit code: 137 (OOM killed)

# 해결:
# 1. 메모리 제한 증가
# 2. 메모리 누수 조사 및 수정
# 3. 캐시 크기 조정
```

---

## ⚡ 성능 최적화

### 병목 지점 식별

#### 1. 애플리케이션 프로파일링
```python
# Python 프로파일링
import cProfile
import pstats

def profile_function(func):
    """함수 성능 프로파일링 데코레이터"""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

        return result
    return wrapper

@profile_function
def slow_function():
    # 느린 함수
    pass
```

#### 2. 데이터베이스 쿼리 최적화
```python
# 느린 쿼리 로깅
import time
import logging

def log_slow_query(query, duration_ms):
    if duration_ms > 100:  # 100ms 이상
        logging.warning(f"Slow query ({duration_ms}ms): {query}")

# 쿼리 실행 시간 측정
start = time.time()
result = db.execute(query)
duration_ms = (time.time() - start) * 1000
log_slow_query(query, duration_ms)
```

#### 3. 캐싱 전략
```python
# Redis 캐싱
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(ttl=300):
    """결과 캐싱 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # 캐시 확인
            cached = redis_client.get(key)
            if cached:
                return cached

            # 캐시 미스 - 실행 및 저장
            result = func(*args, **kwargs)
            redis_client.setex(key, ttl, result)
            return result
        return wrapper
    return decorator

@cache_result(ttl=600)
def expensive_query(param):
    # 비용이 큰 쿼리
    pass
```

### 리소스 최적화

#### CPU 최적화
```yaml
# 적절한 CPU 할당
AWS ECS:
  cpu: 1024  # 1 vCPU
  memory: 2048  # 2 GB

GCP Cloud Run:
  cpu: 2
  memory: 2Gi
```

#### 메모리 최적화
```python
# 메모리 효율적인 코드
# Bad: 전체 리스트 로드
def process_all_items():
    items = db.query("SELECT * FROM items")  # 메모리 부족 위험
    return [process(item) for item in items]

# Good: 스트리밍 처리
def process_items_streaming():
    for item in db.query("SELECT * FROM items").yield_per(100):
        yield process(item)
```

#### 네트워크 최적화
```python
# Connection pooling
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# HTTP 연결 재사용
import requests
session = requests.Session()
session.get(url)  # 연결 재사용
```

### 오토스케일링 정책

#### AWS ECS Auto Scaling
```bash
# Target Tracking Scaling Policy
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/rag-demo-cluster/rag-backend-service \
  --min-capacity 2 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/rag-demo-cluster/rag-backend-service \
  --policy-name cpu-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

#### GCP Cloud Run Auto Scaling
```bash
# 자동 스케일링 설정
gcloud run services update rag-backend \
  --min-instances 1 \
  --max-instances 10 \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 80 \
  --region asia-northeast3
```

---

## 🔐 보안 운영

### 시크릿 관리

#### AWS Secrets Manager
```bash
# 시크릿 생성
aws secretsmanager create-secret \
  --name openai-api-key \
  --secret-string "sk-xxx" \
  --region ap-northeast-2

# 시크릿 조회
aws secretsmanager get-secret-value \
  --secret-id openai-api-key \
  --region ap-northeast-2

# 시크릿 업데이트
aws secretsmanager update-secret \
  --secret-id openai-api-key \
  --secret-string "sk-new-key" \
  --region ap-northeast-2

# 시크릿 로테이션 설정
aws secretsmanager rotate-secret \
  --secret-id openai-api-key \
  --rotation-lambda-arn arn:aws:lambda:... \
  --rotation-rules AutomaticallyAfterDays=30
```

#### GCP Secret Manager
```bash
# 시크릿 생성
echo -n "sk-xxx" | gcloud secrets create openai-api-key --data-file=-

# 시크릿 버전 추가 (업데이트)
echo -n "sk-new-key" | gcloud secrets versions add openai-api-key --data-file=-

# 시크릿 조회
gcloud secrets versions access latest --secret="openai-api-key"

# 이전 버전 조회
gcloud secrets versions access 1 --secret="openai-api-key"

# 버전 삭제
gcloud secrets versions destroy 1 --secret="openai-api-key"
```

### 취약점 스캔

#### Docker 이미지 스캔
```bash
# Trivy로 이미지 스캔
trivy image --severity HIGH,CRITICAL rag-backend:latest

# ECR 자동 스캔 활성화
aws ecr put-image-scanning-configuration \
  --repository-name rag-backend \
  --image-scanning-configuration scanOnPush=true

# 스캔 결과 확인
aws ecr describe-image-scan-findings \
  --repository-name rag-backend \
  --image-id imageTag=latest

# Artifact Registry 스캔 (GCP)
gcloud artifacts docker images scan <IMAGE_URL>
```

#### 의존성 스캔
```bash
# Python 의존성 스캔
pip install safety
safety check -r requirements.txt

# npm audit (Node.js)
npm audit
npm audit fix

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
        "ecs:DescribeTasks",
        "ecs:ListTasks"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Deny",
      "Action": [
        "ecs:DeleteCluster",
        "ecs:DeleteService",
        "ecs:UpdateService"
      ],
      "Resource": "*"
    }
  ]
}
```

#### 네트워크 보안
```bash
# Security Group 최소 규칙
# Only allow:
# - 443 (HTTPS) from ALB security group
# - 8000 (Application) from ALB security group
# - NO direct internet access

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp \
  --port 8000 \
  --source-group sg-alb-xxx
```

---

## 💰 비용 최적화

### 비용 모니터링

#### AWS Cost Explorer
```bash
# 일일 비용 확인
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=SERVICE

# 서비스별 비용
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --group-by Type=SERVICE \
  --filter '{
    "Dimensions": {
      "Key": "SERVICE",
      "Values": ["Amazon Elastic Container Service", "Amazon EC2 Container Registry"]
    }
  }'
```

#### GCP Billing
```bash
# 프로젝트 비용 확인
gcloud billing projects describe $(gcloud config get-value project)

# BigQuery로 상세 분석
bq query --use_legacy_sql=false '
SELECT
  service.description,
  sku.description,
  SUM(cost) as total_cost,
  SUM(usage.amount) as total_usage
FROM `PROJECT_ID.billing.gcp_billing_export_v1_XXX`
WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY service.description, sku.description
ORDER BY total_cost DESC
LIMIT 20
'
```

### 비용 절감 전략

#### 1. Fargate Spot (최대 70% 할인)
```json
{
  "capacityProviderStrategy": [
    {
      "capacityProvider": "FARGATE_SPOT",
      "weight": 1
    },
    {
      "capacityProvider": "FARGATE",
      "weight": 1,
      "base": 1
    }
  ]
}
```

#### 2. Cloud Run 최소 인스턴스 0
```bash
gcloud run services update rag-backend \
  --min-instances=0 \
  --region asia-northeast3
```

#### 3. 이미지 라이프사이클 정책
```json
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
```

#### 4. 로그 보존 기간 최적화
```bash
# 30일로 제한
aws logs put-retention-policy \
  --log-group-name /ecs/rag-backend \
  --retention-in-days 30
```

---

## ✅ 완료 체크리스트

Section 7를 완료하기 전에 다음 항목을 확인하세요:

### 로그 관리
- [ ] CloudWatch Logs 또는 Cloud Logging 설정
- [ ] 로그 보존 정책 설정
- [ ] 로그 필터 및 검색 쿼리 작성
- [ ] 로그 기반 메트릭 생성 (선택)

### 모니터링
- [ ] 대시보드 생성 (CPU, Memory, Latency, Errors)
- [ ] 핵심 메트릭 모니터링 설정
- [ ] 커스텀 메트릭 발행 (선택)

### 알림
- [ ] CloudWatch Alarms 또는 GCP Alerts 설정
- [ ] SNS 토픽 또는 Notification Channels 생성
- [ ] 알림 수신 테스트
- [ ] On-call 일정 설정 (선택)

### 장애 대응
- [ ] 런북(Runbook) 문서 작성
- [ ] 롤백 절차 문서화
- [ ] 포스트모템 템플릿 준비
- [ ] 장애 대응 훈련 실시 (권장)

### 성능 최적화
- [ ] 병목 지점 식별 및 문서화
- [ ] 오토스케일링 정책 설정
- [ ] 캐싱 전략 적용 (권장)

### 보안
- [ ] 시크릿 로테이션 계획 수립
- [ ] 취약점 스캔 자동화 설정
- [ ] IAM 권한 최소화 검토
- [ ] 네트워크 보안 그룹 최적화

### 비용 최적화
- [ ] 비용 모니터링 설정
- [ ] 예산 알림 설정
- [ ] 리소스 최적화 검토
- [ ] 미사용 리소스 정리

---

## 🎓 학습 정리

### 핵심 개념
1. **로그 관리**: 중앙 집중식 로그 수집 및 분석
2. **메트릭 모니터링**: Golden Signals (Latency, Traffic, Errors, Saturation)
3. **알림 시스템**: 적절한 임계값과 알림 채널 설정
4. **장애 대응**: 체계적인 대응 절차와 포스트모템
5. **성능 최적화**: 병목 지점 식별 및 개선
6. **보안 운영**: 시크릿 관리 및 취약점 스캔
7. **비용 최적화**: 효율적인 리소스 사용과 비용 모니터링

### 실무 적용
- 프로덕션 서비스의 안정성 확보
- 신속한 장애 대응 및 복구
- 데이터 기반 성능 개선
- 보안 위협 사전 차단
- 비용 효율적인 운영

### 전체 과정 완료
축하합니다! Section 0-7까지 모두 완료하셨습니다.

**학습 여정:**
- Section 0: 로컬 개발 환경
- Section 1: 환경 변수 관리
- Section 2: Docker 컨테이너화
- Section 3: AWS ECS 배포
- Section 4: GCP Cloud Run 배포
- Section 5: Terraform IaC
- Section 6: CI/CD 파이프라인
- Section 7: 운영 및 모니터링 ✅

이제 여러분은 AI 서비스를 개발, 배포, 운영할 수 있는
완전한 DevOps 역량을 갖추셨습니다!

---

## 📚 추가 학습 자료

### 운영 및 모니터링
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [The Site Reliability Workbook](https://sre.google/workbook/table-of-contents/)
- [AWS Well-Architected Framework - Operational Excellence](https://docs.aws.amazon.com/wellarchitected/latest/operational-excellence-pillar/welcome.html)

### 로그 및 모니터링
- [CloudWatch Logs Insights Query Syntax](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax.html)
- [GCP Cloud Logging Query Language](https://cloud.google.com/logging/docs/view/logging-query-language)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)

### 장애 대응
- [Incident Response Best Practices](https://response.pagerduty.com/)
- [Post-Incident Review Template](https://www.atlassian.com/incident-management/postmortem/templates)

### 성능 최적화
- [High Performance Browser Networking](https://hpbn.co/)
- [Database Performance Tuning](https://use-the-index-luke.com/)

---

**Section 7 및 전체 과정 완료를 축하합니다! 🎉🚀**

이제 여러분은 프로덕션 AI 서비스를 완벽하게
개발, 배포, 운영할 수 있는 풀스택 AI 엔지니어입니다!
