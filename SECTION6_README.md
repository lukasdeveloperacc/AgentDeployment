# Section 6: CI/CD 파이프라인 구성

## 📚 학습 목표

이번 섹션에서는 GitHub Actions를 활용한 CI/CD 파이프라인을 구축합니다.

- GitHub Actions 기본 개념과 워크플로우 작성법 이해
- 자동화된 테스트 및 코드 품질 검사 파이프라인 구성
- Docker 이미지 자동 빌드 및 레지스트리 푸시
- AWS ECS와 GCP Cloud Run 자동 배포 설정
- Terraform 인프라 자동화 파이프라인 구성
- Pull Request 검증 자동화

## 🎯 Section 6에서 다루는 내용

### 1. CI/CD 개념
- Continuous Integration (지속적 통합)
- Continuous Deployment (지속적 배포)
- 자동화의 이점과 Best Practices

### 2. GitHub Actions 기초
- 워크플로우, 잡(Job), 스텝(Step) 구조
- 트리거 이벤트 (push, pull_request, workflow_dispatch)
- GitHub Secrets 관리
- 환경(Environment) 설정

### 3. 구현된 파이프라인
1. **Pull Request 검증** (`.github/workflows/pr-checks.yml`)
2. **AWS 배포 파이프라인** (`.github/workflows/aws-deploy.yml`)
3. **GCP 배포 파이프라인** (`.github/workflows/gcp-deploy.yml`)
4. **Terraform AWS 인프라** (`.github/workflows/terraform-aws.yml`)
5. **Terraform GCP 인프라** (`.github/workflows/terraform-gcp.yml`)

---

## 📖 CI/CD 개념

### Continuous Integration (CI)
코드 변경사항을 자동으로 테스트하고 검증하는 프로세스:
- 코드 푸시 시 자동 테스트 실행
- 코드 품질 검사 (Linting, Formatting)
- 보안 취약점 스캔
- 빌드 테스트

**이점:**
- 버그 조기 발견
- 코드 품질 유지
- 팀 협업 효율 향상
- 통합 충돌 최소화

### Continuous Deployment (CD)
테스트를 통과한 코드를 자동으로 프로덕션에 배포:
- 자동 Docker 이미지 빌드
- 컨테이너 레지스트리 푸시
- 클라우드 서비스 배포
- 배포 검증 및 롤백

**이점:**
- 빠른 배포 주기
- 수동 오류 제거
- 일관된 배포 프로세스
- 빠른 피드백 루프

---

## 🛠️ GitHub Actions 기초

### 워크플로우 구조

```yaml
name: 워크플로우 이름

# 트리거 조건
on:
  push:
    branches:
      - main
  pull_request:

# 환경 변수
env:
  NODE_ENV: production

jobs:
  # Job 1
  job_name:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run command
        run: echo "Hello World"
```

### 주요 구성 요소

#### 1. Triggers (on)
워크플로우를 실행할 이벤트:

```yaml
on:
  push:
    branches: [main, develop]
    paths:
      - 'backend/**'
  pull_request:
    branches: [main]
  workflow_dispatch:  # 수동 실행
  schedule:
    - cron: '0 0 * * *'  # 매일 자정
```

#### 2. Jobs
병렬 또는 순차 실행되는 작업 단위:

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps: [...]

  deploy:
    needs: build  # build가 성공해야 실행
    runs-on: ubuntu-latest
    steps: [...]
```

#### 3. Steps
Job 내에서 실행되는 개별 작업:

```yaml
steps:
  # GitHub Action 사용
  - uses: actions/checkout@v4

  # Shell 명령 실행
  - name: Run tests
    run: pytest tests/

  # 환경 변수 설정
  - name: Set variable
    run: echo "VERSION=1.0.0" >> $GITHUB_ENV
```

#### 4. Secrets
민감한 정보 관리:

```yaml
steps:
  - name: Login to AWS
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

## 🔐 GitHub Secrets 설정

### 필요한 Secrets

파이프라인 실행을 위해 다음 Secrets을 GitHub 저장소에 설정해야 합니다:

#### AWS 관련
```
AWS_ACCESS_KEY_ID          # AWS IAM 사용자 액세스 키
AWS_SECRET_ACCESS_KEY      # AWS IAM 사용자 시크릿 키
```

#### GCP 관련
```
GCP_PROJECT_ID            # GCP 프로젝트 ID
GCP_SA_KEY                # GCP 서비스 계정 JSON 키 (전체 내용)
```

### Secrets 설정 방법

1. **GitHub 저장소 접속**
   ```
   Settings → Secrets and variables → Actions → New repository secret
   ```

2. **AWS Secrets 생성**

   **AWS_ACCESS_KEY_ID 생성:**
   ```bash
   # AWS IAM에서 생성한 액세스 키 ID 입력
   ```

   **AWS_SECRET_ACCESS_KEY 생성:**
   ```bash
   # AWS IAM에서 생성한 시크릿 액세스 키 입력
   ```

3. **GCP Secrets 생성**

   **GCP_PROJECT_ID 생성:**
   ```bash
   # GCP 프로젝트 ID 입력 (예: my-project-12345)
   ```

   **GCP_SA_KEY 생성:**
   ```bash
   # 서비스 계정 JSON 키 파일 전체 내용 복사하여 붙여넣기
   # (gcp-service-account-key.json 파일 내용)
   ```

### IAM 권한 설정

#### AWS IAM 정책
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:*",
        "ecs:*",
        "iam:PassRole",
        "logs:*"
      ],
      "Resource": "*"
    }
  ]
}
```

#### GCP 서비스 계정 역할
```bash
# 필요한 역할
- Artifact Registry Administrator
- Cloud Run Admin
- Service Account User
- Secret Manager Secret Accessor
```

---

## 📋 파이프라인 상세 설명

### 1. Pull Request 검증 파이프라인

**파일:** `.github/workflows/pr-checks.yml`

#### 목적
Pull Request가 생성될 때 자동으로 코드 품질을 검증합니다.

#### 실행 조건
```yaml
on:
  pull_request:
    branches:
      - main
      - develop
```

#### Jobs 구성

**Job 1: Code Quality (코드 품질 검사)**
```yaml
code-quality:
  runs-on: ubuntu-latest
  steps:
    - Checkout code
    - Set up Python
    - Install dependencies
    - Run Ruff (linting)
    - Run Black (formatting check)
    - Run MyPy (type checking)
```

**검사 항목:**
- **Ruff**: Python 코드 스타일 및 잠재적 버그 검사
- **Black**: 코드 포맷팅 일관성 검증
- **MyPy**: 타입 힌트 검증 (실패해도 계속 진행)

**Job 2: Security (보안 검사)**
```yaml
security:
  runs-on: ubuntu-latest
  steps:
    - Checkout code
    - Run Trivy vulnerability scanner
    - Upload results to GitHub Security tab
```

**검사 항목:**
- **Trivy**: 의존성 취약점 스캔
- 결과를 GitHub Security 탭에 업로드

**Job 3: Tests (테스트)**
```yaml
tests:
  runs-on: ubuntu-latest
  steps:
    - Checkout code
    - Set up Python
    - Install dependencies
    - Run tests with coverage
    - Upload coverage to Codecov
```

**검사 항목:**
- **pytest**: 단위 테스트 및 통합 테스트 실행
- **coverage**: 코드 커버리지 측정 및 리포트 생성

**Job 4: Docker Build (Docker 빌드 테스트)**
```yaml
docker-build:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      service: [backend, frontend]
  steps:
    - Checkout code
    - Set up Docker Buildx
    - Build Docker image (push하지 않음)
```

**검사 항목:**
- Backend와 Frontend Docker 이미지가 정상적으로 빌드되는지 검증
- 실제로 푸시하지는 않음 (테스트만)

**Job 5: PR Size (PR 크기 검사)**
```yaml
pr-size:
  runs-on: ubuntu-latest
  steps:
    - Check PR size
    - Fail if changes > 500 lines
```

**검사 항목:**
- PR의 변경 라인 수가 500줄 이하인지 확인
- 너무 큰 PR은 리뷰가 어려우므로 실패 처리

**Job 6: Summary (결과 요약)**
```yaml
summary:
  needs: [code-quality, security, tests, docker-build]
  runs-on: ubuntu-latest
  if: always()
  steps:
    - Create summary comment on PR
```

**기능:**
- 모든 검사 결과를 요약하여 PR에 코멘트로 게시
- 각 Job의 성공/실패 상태를 이모지로 표시

#### 사용 예시

```bash
# 1. 새로운 브랜치 생성
git checkout -b feature/new-feature

# 2. 코드 변경 후 커밋
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# 3. GitHub에서 Pull Request 생성
# → pr-checks.yml이 자동으로 실행됨

# 4. PR 페이지에서 검사 결과 확인
# → 모든 검사가 통과하면 ✅ 표시
# → 실패한 검사가 있으면 ❌ 표시 및 상세 로그 확인
```

---

### 2. AWS 배포 파이프라인

**파일:** `.github/workflows/aws-deploy.yml`

#### 목적
코드가 main 또는 develop 브랜치에 푸시되면 자동으로 AWS ECS에 배포합니다.

#### 실행 조건
```yaml
on:
  push:
    branches:
      - main
      - develop
    paths:
      - 'backend/**'
      - 'frontend/**'
      - '.github/workflows/aws-deploy.yml'
  workflow_dispatch:  # 수동 실행도 가능
```

#### Jobs 구성

**Job 1: Test (테스트)**
```yaml
test:
  runs-on: ubuntu-latest
  steps:
    - Run backend tests
    - Run linting
```

**Job 2: Build and Push (빌드 및 푸시)**
```yaml
build-and-push:
  needs: test
  runs-on: ubuntu-latest
  steps:
    - Checkout code
    - Configure AWS credentials
    - Login to Amazon ECR
    - Build and push backend image
    - Build and push frontend image
```

**주요 기능:**
- 테스트가 성공해야만 실행됨 (`needs: test`)
- Git commit SHA를 이미지 태그로 사용 (추적 가능)
- ECR에 이미지 푸시 (sha_short 태그 + latest 태그)
- 다음 Job에서 사용할 이미지 URL을 output으로 전달

**Job 3: Deploy (배포)**
```yaml
deploy:
  needs: build-and-push
  runs-on: ubuntu-latest
  environment:
    name: production
    url: http://${{ steps.get-alb-dns.outputs.alb_dns }}
  steps:
    - Update backend ECS service
    - Update frontend ECS service
    - Wait for service stability
    - Get ALB DNS name
```

**주요 기능:**
- `force-new-deployment`로 새 이미지 배포 강제
- ECS 서비스 안정화 대기 (최대 15분)
- 배포 완료 후 ALB DNS 출력

**Job 4: Notify (알림)**
```yaml
notify:
  needs: [test, build-and-push, deploy]
  runs-on: ubuntu-latest
  if: always()
  steps:
    - Check deployment status
    - Display success/failure message
```

#### 환경 변수 설정
```yaml
env:
  AWS_REGION: ap-northeast-2
  ECR_BACKEND_REPO: rag-backend
  ECR_FRONTEND_REPO: rag-frontend
  ECS_CLUSTER: rag-demo-cluster
  ECS_BACKEND_SERVICE: rag-backend-service
  ECS_FRONTEND_SERVICE: rag-frontend-service
```

#### 사용 예시

```bash
# 1. 코드 변경 및 커밋
git add .
git commit -m "Update backend API"
git push origin main

# 2. GitHub Actions 자동 실행
# → 테스트 → 빌드 → 배포 순서로 진행

# 3. 배포 상태 확인
# GitHub → Actions → 최신 워크플로우 클릭
# → 각 Job의 로그 확인 가능

# 4. 수동 실행 (필요시)
# GitHub → Actions → AWS ECS Deployment → Run workflow
```

#### 배포 검증

```bash
# 1. ECS 서비스 상태 확인
aws ecs describe-services \
  --cluster rag-demo-cluster \
  --services rag-backend-service rag-frontend-service \
  --region ap-northeast-2

# 2. ALB DNS로 접속 테스트
curl http://<ALB-DNS>/health

# 3. CloudWatch 로그 확인
aws logs tail /ecs/rag-backend --follow --region ap-northeast-2
```

---

### 3. GCP 배포 파이프라인

**파일:** `.github/workflows/gcp-deploy.yml`

#### 목적
코드가 main 또는 develop 브랜치에 푸시되면 자동으로 GCP Cloud Run에 배포합니다.

#### 실행 조건
```yaml
on:
  push:
    branches:
      - main
      - develop
    paths:
      - 'backend/**'
      - 'frontend/**'
      - '.github/workflows/gcp-deploy.yml'
  workflow_dispatch:
```

#### Jobs 구성

**Job 1: Test (테스트)**
- AWS 파이프라인과 동일

**Job 2: Build and Push (빌드 및 푸시)**
```yaml
build-and-push:
  needs: test
  runs-on: ubuntu-latest
  steps:
    - Checkout code
    - Authenticate to Google Cloud
    - Configure Docker for Artifact Registry
    - Build and push backend image
    - Build and push frontend image
```

**주요 차이점 (AWS와 비교):**
- GCP 인증: Service Account JSON 키 사용
- Artifact Registry: GCP의 컨테이너 레지스트리
- 이미지 URL 형식: `{region}-docker.pkg.dev/{project}/{repo}/{image}:{tag}`

**Job 3: Deploy (배포)**
```yaml
deploy:
  needs: build-and-push
  runs-on: ubuntu-latest
  environment:
    name: production
    url: https://${{ steps.deploy-backend.outputs.url }}
  steps:
    - Deploy backend to Cloud Run
    - Deploy frontend to Cloud Run
    - Wait for services to be ready
    - Health check
```

**Cloud Run 배포 설정:**
```bash
gcloud run deploy rag-backend \
  --image $IMAGE \
  --region asia-northeast3 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=production" \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest" \
  --cpu 2 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300 \
  --port 8000
```

**주요 설정:**
- `--allow-unauthenticated`: 공개 접근 허용
- `--set-secrets`: Secret Manager에서 시크릿 주입
- `--min-instances 0`: 트래픽 없을 때 스케일 다운
- `--max-instances 10`: 최대 10개 인스턴스까지 오토스케일링

**Job 4: Notify (알림)**
- AWS 파이프라인과 동일

#### 환경 변수 설정
```yaml
env:
  GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  GCP_REGION: asia-northeast3
  ARTIFACT_REGISTRY_REPO: rag-demo
  BACKEND_SERVICE: rag-backend
  FRONTEND_SERVICE: rag-frontend
```

#### 사용 예시

```bash
# 1. 코드 변경 및 커밋
git add .
git commit -m "Update backend API"
git push origin main

# 2. GitHub Actions 자동 실행

# 3. 배포 상태 확인
# GitHub → Actions → GCP Cloud Run Deployment

# 4. Cloud Run 서비스 URL 확인
gcloud run services describe rag-backend \
  --region asia-northeast3 \
  --format="value(status.url)"
```

#### 배포 검증

```bash
# 1. Cloud Run 서비스 상태 확인
gcloud run services list --region asia-northeast3

# 2. Health check
BACKEND_URL=$(gcloud run services describe rag-backend \
  --region asia-northeast3 \
  --format="value(status.url)")
curl $BACKEND_URL/health

# 3. 로그 확인
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=rag-backend" \
  --limit 50 \
  --format json
```

---

### 4. Terraform AWS 인프라 파이프라인

**파일:** `.github/workflows/terraform-aws.yml`

#### 목적
Terraform 코드 변경 시 인프라를 자동으로 프로비저닝합니다.

#### 실행 조건
```yaml
on:
  push:
    branches:
      - main
    paths:
      - 'terraform/aws/**'
      - '.github/workflows/terraform-aws.yml'
  pull_request:
    paths:
      - 'terraform/aws/**'
  workflow_dispatch:
```

#### Jobs 구성

**Job 1: Terraform Plan (계획 단계)**
```yaml
terraform-plan:
  runs-on: ubuntu-latest
  defaults:
    run:
      working-directory: terraform/aws
  steps:
    - Checkout code
    - Setup Terraform
    - Configure AWS credentials
    - Terraform Format Check
    - Terraform Init
    - Terraform Validate
    - Terraform Plan
    - Comment PR (PR인 경우)
```

**주요 단계:**

1. **Format Check**
   ```bash
   terraform fmt -check -recursive
   ```
   - 코드 포맷팅 검증
   - 실패해도 계속 진행 (`continue-on-error: true`)

2. **Init**
   ```bash
   terraform init
   ```
   - Provider 다운로드
   - Backend 초기화

3. **Validate**
   ```bash
   terraform validate
   ```
   - 문법 및 구성 검증

4. **Plan**
   ```bash
   terraform plan -no-color
   ```
   - 변경 사항 미리보기
   - 생성/수정/삭제될 리소스 확인

5. **Comment PR**
   - PR인 경우 Plan 결과를 코멘트로 게시
   - 리뷰어가 변경사항을 쉽게 파악 가능

**Job 2: Terraform Apply (적용 단계)**
```yaml
terraform-apply:
  needs: terraform-plan
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  runs-on: ubuntu-latest
  environment: production
  steps:
    - Checkout code
    - Setup Terraform
    - Configure AWS credentials
    - Terraform Init
    - Terraform Apply
```

**실행 조건:**
- main 브랜치에 푸시된 경우만 실행
- PR에서는 실행되지 않음 (Plan만 실행)
- `environment: production` 설정으로 승인 필요 (선택사항)

**Apply 명령:**
```bash
terraform apply -auto-approve
```
- 자동으로 승인하여 적용
- 수동 승인이 필요한 경우 GitHub Environment 설정 사용

#### 환경 변수
```yaml
env:
  AWS_REGION: ap-northeast-2
  TF_VERSION: 1.6.0
```

#### 사용 예시

```bash
# 1. Terraform 코드 변경
cd terraform/aws
vim main.tf

# 2. Pull Request 생성
git checkout -b infra/update-ecs-config
git add .
git commit -m "Update ECS task CPU to 1024"
git push origin infra/update-ecs-config

# 3. PR 생성 → Plan 자동 실행
# → PR 코멘트에 Plan 결과 표시

# 4. PR 리뷰 및 머지
# → main 브랜치로 머지

# 5. Apply 자동 실행
# → 실제 인프라 변경 적용
```

#### 주의사항

⚠️ **중요:**
- Apply는 main 브랜치에서만 실행됩니다
- Production 환경 변경은 신중하게 검토 후 머지하세요
- Terraform State는 S3 Backend에 저장되어 있어야 합니다
- 여러 명이 동시에 Apply하면 충돌 발생 가능 (State Lock 필요)

---

### 5. Terraform GCP 인프라 파이프라인

**파일:** `.github/workflows/terraform-gcp.yml`

#### 목적
Terraform 코드 변경 시 GCP 인프라를 자동으로 프로비저닝합니다.

#### 실행 조건 및 구조
- AWS Terraform 워크플로우와 동일한 구조
- GCP 인증 방식만 다름:

```yaml
- name: Authenticate to Google Cloud
  uses: google-github-actions/auth@v2
  with:
    credentials_json: ${{ secrets.GCP_SA_KEY }}

- name: Set up Cloud SDK
  uses: google-github-actions/setup-gcloud@v2
```

#### 환경 변수
```yaml
env:
  GCP_REGION: asia-northeast3
  TF_VERSION: 1.6.0
```

#### 사용 예시
- AWS Terraform 워크플로우와 동일
- `terraform/gcp` 디렉토리에서 작업

---

## 🎬 워크플로우 수동 실행

모든 워크플로우는 `workflow_dispatch` 트리거를 지원하여 수동 실행이 가능합니다.

### 수동 실행 방법

1. **GitHub 저장소 접속**
   ```
   GitHub → Actions → 워크플로우 선택
   ```

2. **Run workflow 버튼 클릭**
   ```
   우측 상단 "Run workflow" 버튼
   → 브랜치 선택
   → "Run workflow" 확인
   ```

3. **실행 상태 확인**
   ```
   워크플로우 실행 목록에서 상태 확인
   → 클릭하여 상세 로그 확인
   ```

### 수동 실행이 유용한 경우

- **재배포**: 코드 변경 없이 재배포가 필요한 경우
- **롤백**: 이전 커밋으로 배포하고 싶은 경우
- **테스트**: 특정 브랜치에서 파이프라인 테스트
- **인프라 변경**: Terraform Apply를 수동으로 실행

---

## 📊 워크플로우 결과 확인

### GitHub Actions 페이지

```
GitHub → Actions
```

**주요 정보:**
- ✅ 성공한 워크플로우 (녹색 체크)
- ❌ 실패한 워크플로우 (빨간 X)
- 🔄 실행 중인 워크플로우 (노란색 점)
- ⏱️ 실행 시간 및 커밋 정보

### 상세 로그 확인

1. **워크플로우 클릭**
   ```
   워크플로우 실행 목록에서 항목 클릭
   ```

2. **Job 선택**
   ```
   좌측 사이드바에서 Job 선택
   ```

3. **Step 로그 확인**
   ```
   각 Step 클릭하여 상세 로그 확인
   ```

### PR에서 확인

Pull Request 페이지에서 CI/CD 상태를 직접 확인할 수 있습니다:

```
PR → Checks 탭
→ 각 워크플로우의 성공/실패 상태
→ "Details" 클릭하여 상세 로그 확인
```

### 알림 설정

GitHub 알림 설정:
```
Settings → Notifications → Actions
→ 실패 시 이메일 알림 활성화
```

---

## 🔧 트러블슈팅

### 1. Secrets 관련 오류

**증상:**
```
Error: The secrets.AWS_ACCESS_KEY_ID is not set
```

**해결방법:**
```bash
# 1. Settings → Secrets and variables → Actions 확인
# 2. 필요한 Secrets이 모두 등록되어 있는지 확인
# 3. Secret 이름 철자 확인 (대소문자 구분)
```

### 2. AWS 권한 오류

**증상:**
```
Error: User is not authorized to perform: ecs:UpdateService
```

**해결방법:**
```bash
# IAM 정책 확인
aws iam list-attached-user-policies --user-name github-actions

# 필요한 권한 추가
# - ecr:*
# - ecs:UpdateService
# - ecs:DescribeServices
# - iam:PassRole
```

### 3. GCP 인증 오류

**증상:**
```
Error: Failed to create cluster: Service account key is invalid
```

**해결방법:**
```bash
# 1. 서비스 계정 키 재생성
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@PROJECT_ID.iam.gserviceaccount.com

# 2. JSON 파일 전체 내용을 GCP_SA_KEY Secret에 저장
cat key.json

# 3. Secret 업데이트 후 워크플로우 재실행
```

### 4. Docker 빌드 실패

**증상:**
```
Error: failed to solve: failed to fetch oauth token
```

**해결방법:**
```bash
# 1. Dockerfile 문법 확인
docker build -t test .

# 2. 의존성 설치 오류인 경우
# - requirements.txt 또는 package.json 확인
# - 버전 호환성 확인

# 3. 캐시 초기화
docker system prune -a
```

### 5. ECS 배포 타임아웃

**증상:**
```
Error: Waited 15 minutes for service to stabilize but failed
```

**해결방법:**
```bash
# 1. ECS 서비스 이벤트 확인
aws ecs describe-services \
  --cluster rag-demo-cluster \
  --services rag-backend-service

# 2. Task Definition 로그 확인
aws ecs describe-tasks \
  --cluster rag-demo-cluster \
  --tasks <task-arn>

# 3. CloudWatch Logs 확인
aws logs tail /ecs/rag-backend --follow

# 4. Health check 설정 확인
# - 올바른 경로 (/health)
# - 적절한 타임아웃 설정
```

### 6. Terraform State Lock

**증상:**
```
Error: Error acquiring the state lock
```

**해결방법:**
```bash
# 1. 다른 사용자가 Apply 중인지 확인
# 2. 강제로 Lock 해제 (주의!)
terraform force-unlock <LOCK_ID>

# 3. S3 Backend 확인
aws s3 ls s3://terraform-state-bucket/
```

### 7. 워크플로우 디버깅

**로그 상세도 높이기:**
```yaml
# .github/workflows/aws-deploy.yml
env:
  ACTIONS_RUNNER_DEBUG: true
  ACTIONS_STEP_DEBUG: true
```

**특정 Step만 실행:**
```yaml
steps:
  - name: Debug step
    if: github.event_name == 'workflow_dispatch'
    run: |
      echo "Debug information"
      env
```

---

## 🎯 Best Practices

### 1. 브랜치 전략

**권장 구조:**
```
main (프로덕션)
├── develop (개발)
│   ├── feature/new-feature
│   └── feature/bug-fix
└── hotfix/critical-bug
```

**배포 전략:**
- `feature/*` → `develop` PR: PR 검증만
- `develop` → `main` PR: Staging 배포
- `main` 머지: Production 배포

### 2. 환경 분리

**Environment 설정:**
```yaml
# Production 환경
deploy-prod:
  environment:
    name: production
    url: https://api.example.com
  steps: [...]

# Staging 환경
deploy-staging:
  environment:
    name: staging
    url: https://staging.example.com
  steps: [...]
```

**이점:**
- 배포 승인 프로세스 추가 가능
- 환경별 Secrets 분리
- 배포 이력 추적

### 3. 캐싱 활용

**의존성 캐싱:**
```yaml
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

**Docker 레이어 캐싱:**
```yaml
- name: Build Docker image
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### 4. Parallel Jobs

**병렬 실행으로 속도 향상:**
```yaml
jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps: [...]

  test-frontend:
    runs-on: ubuntu-latest
    steps: [...]

  deploy:
    needs: [test-backend, test-frontend]
    runs-on: ubuntu-latest
    steps: [...]
```

### 5. Matrix 전략

**여러 버전 테스트:**
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
```

### 6. 보안

**Secrets 사용:**
```yaml
# ✅ 올바른 사용
env:
  API_KEY: ${{ secrets.API_KEY }}

# ❌ 잘못된 사용
run: echo "API_KEY=secret123"
```

**CODEOWNERS 설정:**
```
# .github/CODEOWNERS
.github/workflows/* @team-devops
terraform/* @team-infrastructure
```

### 7. 알림

**Slack 알림 추가:**
```yaml
- name: Notify Slack
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "Deployment failed: ${{ github.repository }}"
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 📈 모니터링 및 최적화

### 워크플로우 실행 시간 추적

```yaml
- name: Measure execution time
  run: |
    echo "::set-output name=start::$(date +%s)"
    # 작업 수행
    DURATION=$(($(date +%s) - $START_TIME))
    echo "Duration: $DURATION seconds"
```

### 비용 최적화

**실행 시간 줄이기:**
1. 캐싱 적극 활용
2. 불필요한 Step 제거
3. Parallel Jobs 활용
4. Self-hosted runners 고려 (대규모 프로젝트)

**실행 빈도 최적화:**
```yaml
# 특정 경로 변경 시만 실행
on:
  push:
    paths:
      - 'src/**'
      - '!docs/**'  # docs는 제외
```

### 성능 메트릭

**추적할 메트릭:**
- 평균 빌드 시간
- 성공률
- 배포 빈도
- 평균 배포 소요 시간
- 롤백 빈도

---

## ✅ 완료 체크리스트

Section 6를 완료하기 전에 다음 항목을 확인하세요:

### GitHub Secrets 설정
- [ ] `AWS_ACCESS_KEY_ID` 등록
- [ ] `AWS_SECRET_ACCESS_KEY` 등록
- [ ] `GCP_PROJECT_ID` 등록
- [ ] `GCP_SA_KEY` 등록

### IAM 권한 설정
- [ ] AWS IAM 사용자 생성 및 정책 연결
- [ ] GCP 서비스 계정 생성 및 역할 부여

### 워크플로우 테스트
- [ ] PR 생성 → PR Checks 실행 확인
- [ ] main 브랜치 푸시 → AWS 배포 확인
- [ ] main 브랜치 푸시 → GCP 배포 확인
- [ ] Terraform 변경 → Plan/Apply 확인

### 배포 검증
- [ ] AWS ECS 서비스 정상 동작 확인
- [ ] GCP Cloud Run 서비스 정상 동작 확인
- [ ] Health check 엔드포인트 응답 확인
- [ ] 로그 정상 출력 확인

### 문서화
- [ ] 팀원들에게 Secrets 설정 방법 공유
- [ ] 배포 프로세스 문서화
- [ ] 트러블슈팅 가이드 작성
- [ ] 롤백 절차 문서화

---

## 🎓 학습 정리

### 핵심 개념
1. **CI/CD**: 지속적 통합 및 배포로 개발 속도와 품질 향상
2. **GitHub Actions**: YAML 기반 워크플로우 자동화
3. **Infrastructure as Code**: Terraform으로 인프라 자동화
4. **Multi-Cloud**: AWS와 GCP 동시 배포 전략

### 실무 적용
- 코드 품질 자동 검증으로 버그 조기 발견
- 자동 배포로 수동 작업 제거 및 일관성 확보
- IaC로 인프라 버전 관리 및 재현 가능성 확보
- 멀티 클라우드로 vendor lock-in 방지

### 다음 단계
Section 7에서는 운영 및 모니터링을 다룹니다:
- 로그 수집 및 분석
- 메트릭 모니터링 및 알림
- 장애 대응 및 복구
- 성능 최적화

---

## 📚 추가 학습 자료

### GitHub Actions
- [GitHub Actions 공식 문서](https://docs.github.com/en/actions)
- [Awesome Actions](https://github.com/sdras/awesome-actions)
- [GitHub Actions 마켓플레이스](https://github.com/marketplace?type=actions)

### CI/CD Best Practices
- [The Twelve-Factor App](https://12factor.net/)
- [GitLab CI/CD Guide](https://docs.gitlab.com/ee/ci/)
- [Jenkins Best Practices](https://www.jenkins.io/doc/book/pipeline/pipeline-best-practices/)

### Terraform
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)

### Docker
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Builds](https://docs.docker.com/develop/develop-images/multistage-build/)

---

## 🤝 기여 및 피드백

이 Section에 대한 피드백이나 개선 제안이 있다면:
1. Issue 생성
2. Pull Request 제출
3. 강사에게 직접 문의

**다음 Section**: [Section 7: 운영 및 모니터링](SECTION7_README.md)

---

**Section 6 완료를 축하합니다! 🎉**

이제 CI/CD 파이프라인을 통해 자동화된 배포 프로세스를 구축했습니다.
Section 7에서는 운영 단계에서 필요한 모니터링과 관리 방법을 학습합니다.
