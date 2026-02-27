# Section 5: Terraform 기반 Infrastructure as Code (IaC)

## 학습 목표
- IaC의 핵심 개념과 Terraform 기초 이해
- Terraform으로 AWS ECS 인프라 자동 프로비저닝
- Terraform으로 GCP Cloud Run 인프라 자동 프로비저닝
- 모듈화를 통한 재사용 가능한 인프라 코드 작성
- Terraform State 관리 및 협업 전략

## 새로 추가된 파일
```
terraform/
  ├── aws/
  │   ├── main.tf                        # AWS 메인 구성
  │   ├── variables.tf                   # AWS 변수 정의
  │   ├── terraform.tfvars.example       # AWS 변수 예시
  │   └── modules/
  │       ├── vpc/                       # VPC 모듈
  │       │   ├── main.tf
  │       │   ├── variables.tf
  │       │   └── outputs.tf
  │       ├── ecr/                       # ECR 모듈
  │       │   ├── main.tf
  │       │   ├── variables.tf
  │       │   └── outputs.tf
  │       ├── alb/                       # ALB 모듈
  │       │   ├── main.tf
  │       │   ├── variables.tf
  │       │   └── outputs.tf
  │       └── ecs/                       # ECS 모듈
  │           ├── main.tf
  │           ├── variables.tf
  │           └── outputs.tf
  │
  └── gcp/
      ├── main.tf                        # GCP 메인 구성
      ├── variables.tf                   # GCP 변수 정의
      ├── terraform.tfvars.example       # GCP 변수 예시
      └── modules/
          ├── artifact-registry/         # Artifact Registry 모듈
          │   ├── main.tf
          │   ├── variables.tf
          │   └── outputs.tf
          └── cloud-run/                 # Cloud Run 모듈
              ├── main.tf
              ├── variables.tf
              └── outputs.tf
```

## Infrastructure as Code (IaC)란?

### 기존 방식 vs IaC
```
기존 방식 (ClickOps):
1. AWS Console 접속
2. VPC 생성 클릭
3. Subnet 설정 입력
4. Security Group 설정...
❌ 재현 불가능, 문서화 어려움, 휴먼 에러 발생

IaC (Terraform):
1. .tf 파일 작성
2. terraform apply
✅ 재현 가능, 버전 관리, 자동화, 협업 용이
```

### Terraform 핵심 개념
- **Provider**: 클라우드 제공자 (AWS, GCP, Azure 등)
- **Resource**: 생성할 인프라 리소스 (VPC, VM, LB 등)
- **Module**: 재사용 가능한 인프라 구성 단위
- **State**: 현재 인프라 상태 추적 파일
- **Plan**: 변경 사항 미리보기 (실행 전 검증)
- **Apply**: 실제 인프라 변경 적용

## Terraform 설치

### macOS
```bash
# Homebrew 이용
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# 설치 확인
terraform version
```

### Linux
```bash
# Ubuntu/Debian
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

## AWS 인프라 프로비저닝

### 1. 사전 준비
```bash
# AWS 자격증명 설정
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-northeast-2"

# Secrets Manager에 API Key 미리 저장 (Section 3 참고)
aws secretsmanager create-secret \
  --name prod/openai-api-key \
  --secret-string "your-openai-api-key"

aws secretsmanager create-secret \
  --name prod/pinecone-api-key \
  --secret-string "your-pinecone-api-key"
```

### 2. Terraform 변수 설정
```bash
cd terraform/aws

# terraform.tfvars 파일 생성
cp terraform.tfvars.example terraform.tfvars

# terraform.tfvars 편집
cat > terraform.tfvars <<EOF
aws_region   = "ap-northeast-2"
project_name = "rag-demo"
environment  = "dev"

vpc_cidr           = "10.0.0.0/16"
availability_zones = ["ap-northeast-2a", "ap-northeast-2c"]
public_subnets     = ["10.0.1.0/24", "10.0.2.0/24"]
private_subnets    = ["10.0.11.0/24", "10.0.12.0/24"]

openai_secret_arn   = "arn:aws:secretsmanager:ap-northeast-2:123456789012:secret:prod/openai-api-key-xxxxx"
pinecone_secret_arn = "arn:aws:secretsmanager:ap-northeast-2:123456789012:secret:prod/pinecone-api-key-xxxxx"
EOF
```

### 3. Terraform 실행
```bash
# 초기화 (Provider 다운로드)
terraform init

# 실행 계획 확인 (실제 변경 전 미리보기)
terraform plan

# 인프라 생성
terraform apply

# 확인 프롬프트에서 'yes' 입력
```

### 4. 출력 확인
```bash
# 생성된 리소스 출력
terraform output

# 예시 출력:
# alb_dns_name = "rag-demo-dev-alb-123456789.ap-northeast-2.elb.amazonaws.com"
# backend_ecr_url = "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/rag-demo-rag-backend"
# ecs_cluster_name = "rag-demo-dev-cluster"
```

### 5. 이미지 푸시 및 서비스 업데이트
```bash
# Section 3의 스크립트 사용
cd ../..
./scripts/aws-deploy.sh latest

# ECS 서비스는 자동으로 최신 이미지로 업데이트됨
```

## GCP 인프라 프로비저닝

### 1. 사전 준비
```bash
# gcloud 인증
gcloud auth application-default login

# 프로젝트 설정
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID

# Secret Manager에 API Key 저장 (Section 4 참고)
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your-pinecone-api-key" | gcloud secrets create pinecone-api-key --data-file=-
```

### 2. Terraform 변수 설정
```bash
cd terraform/gcp

# terraform.tfvars 파일 생성
cp terraform.tfvars.example terraform.tfvars

# terraform.tfvars 편집
cat > terraform.tfvars <<EOF
gcp_project_id = "your-gcp-project-id"
gcp_region     = "asia-northeast3"
project_name   = "rag-demo"
environment    = "dev"
EOF
```

### 3. Terraform 실행
```bash
# 초기화
terraform init

# 실행 계획 확인
terraform plan

# 인프라 생성
terraform apply

# 확인 프롬프트에서 'yes' 입력
```

### 4. 출력 확인
```bash
# 생성된 리소스 출력
terraform output

# 예시 출력:
# backend_url = "https://rag-demo-dev-backend-abc123-an.a.run.app"
# frontend_url = "https://rag-demo-dev-frontend-def456-an.a.run.app"
# artifact_registry_url = "asia-northeast3-docker.pkg.dev/your-project-id/rag-demo-dev"
```

### 5. 이미지 푸시
```bash
# Section 4의 스크립트 사용
cd ../..
export GCP_PROJECT_ID="your-project-id"
./scripts/gcp-deploy.sh latest

# Cloud Run 서비스는 자동으로 최신 이미지로 업데이트됨
```

## Terraform 주요 명령어

### 기본 워크플로우
```bash
# 1. 초기화 (최초 1회, Provider 설치)
terraform init

# 2. 포맷팅 (코드 정리)
terraform fmt

# 3. 유효성 검사
terraform validate

# 4. 실행 계획 확인 (dry-run)
terraform plan

# 5. 변경 적용
terraform apply

# 6. 특정 리소스만 적용
terraform apply -target=module.vpc

# 7. 리소스 삭제
terraform destroy
```

### State 관리
```bash
# State 목록 보기
terraform state list

# 특정 리소스 상태 보기
terraform state show module.vpc.aws_vpc.main

# State에서 리소스 제거 (실제 리소스는 유지)
terraform state rm module.vpc.aws_vpc.main

# State 백업
cp terraform.tfstate terraform.tfstate.backup
```

### 출력 관리
```bash
# 모든 출력 보기
terraform output

# 특정 출력만 보기
terraform output alb_dns_name

# JSON 포맷으로 출력
terraform output -json
```

## 모듈 구조 설명

### AWS 모듈 구성
```
terraform/aws/
├── main.tf              # 메인 구성, 모듈 조합
├── variables.tf         # 입력 변수 정의
├── outputs.tf           # 출력 변수 정의 (없음)
└── modules/
    ├── vpc/            # 네트워크 인프라
    │   ├── VPC, Subnet, NAT Gateway
    │   ├── Security Groups
    │   └── Route Tables
    ├── ecr/            # 컨테이너 레지스트리
    │   └── ECR Repositories
    ├── alb/            # 로드 밸런서
    │   ├── Application Load Balancer
    │   ├── Target Groups
    │   └── Listener Rules
    └── ecs/            # 컨테이너 오케스트레이션
        ├── ECS Cluster
        ├── Task Definitions
        ├── ECS Services
        └── IAM Roles
```

### GCP 모듈 구성
```
terraform/gcp/
├── main.tf              # 메인 구성, 모듈 조합
├── variables.tf         # 입력 변수 정의
├── outputs.tf           # 출력 변수 정의 (없음)
└── modules/
    ├── artifact-registry/  # 컨테이너 레지스트리
    │   └── Artifact Registry
    └── cloud-run/          # 서버리스 컨테이너
        ├── Cloud Run Services
        ├── Service Accounts
        └── IAM Policies
```

## Terraform State 관리

### State란?
- Terraform이 관리하는 인프라의 현재 상태
- `terraform.tfstate` 파일에 JSON 형식으로 저장
- **매우 중요**: 이 파일이 없으면 인프라 관리 불가능

### State 저장 위치

#### 로컬 State (기본)
```hcl
# 별도 설정 없음 → terraform.tfstate 파일로 로컬 저장
# ❌ 협업 불가능, 동기화 문제, 백업 필요
```

#### 원격 State (프로덕션 권장)
```hcl
# AWS S3 Backend
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "rag-demo/terraform.tfstate"
    region         = "ap-northeast-2"
    dynamodb_table = "terraform-locks"  # Lock 방지
    encrypt        = true
  }
}

# GCP GCS Backend
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "rag-demo/terraform.tfstate"
  }
}
```

### State Lock
- 여러 사람이 동시에 `terraform apply` 실행 방지
- AWS: DynamoDB 테이블 사용
- GCP: 자동 Lock 지원

## 변수 관리 전략

### 1. 환경별 변수 분리
```bash
# 디렉토리 구조
terraform/aws/
├── environments/
│   ├── dev/
│   │   └── terraform.tfvars
│   ├── stage/
│   │   └── terraform.tfvars
│   └── prod/
│       └── terraform.tfvars
```

### 2. Sensitive 변수 보호
```hcl
variable "openai_api_key" {
  type      = string
  sensitive = true  # Plan/Apply 출력에서 마스킹
}
```

### 3. 환경변수 사용
```bash
# TF_VAR_ 접두사로 변수 주입
export TF_VAR_gcp_project_id="your-project-id"
terraform apply
```

## 비용 관리

### AWS 비용 추정
```
VPC: $0 (무료)
NAT Gateway: $0.045/hour (~$32/month)
ALB: $0.0225/hour (~$16/month)
ECS Fargate: 사용량 기반
  - vCPU: $0.04856/vCPU/hour
  - Memory: $0.00532/GB/hour
```

### GCP 비용 추정
```
Artifact Registry: $0.10/GB/month (스토리지)
Cloud Run:
  - vCPU: $0.00002400/vCPU-second
  - Memory: $0.00000250/GiB-second
  - Requests: $0.40/million requests
무료 할당량: 월 200만 요청, 36만 vCPU-second
```

### 비용 최적화 팁
1. **개발 환경**: 필요 시에만 인프라 생성
   ```bash
   # 작업 시작
   terraform apply

   # 작업 완료 후 삭제
   terraform destroy
   ```

2. **Auto Scaling 활용**: 트래픽에 따라 자동 조절
3. **Spot Instances**: AWS Fargate Spot 사용 (최대 70% 할인)
4. **리소스 크기 최적화**: 필요한 만큼만 할당

## 트러블슈팅

### State Lock 에러
```
Error: Error acquiring the state lock
```
**해결**:
```bash
# Lock 강제 해제 (주의: 다른 사람이 실행 중이 아닌지 확인!)
terraform force-unlock LOCK_ID
```

### Provider 버전 충돌
```
Error: Incompatible provider version
```
**해결**:
```bash
# .terraform 디렉토리 삭제 후 재초기화
rm -rf .terraform
terraform init
```

### 변수 누락 에러
```
Error: No value for required variable
```
**해결**:
```bash
# terraform.tfvars 파일 확인
cat terraform.tfvars

# 또는 명령줄에서 직접 전달
terraform apply -var="gcp_project_id=your-project-id"
```

### 리소스 삭제 실패
```
Error: Error deleting VPC: DependencyViolation
```
**해결**:
```bash
# 의존 관계 확인
terraform state list

# 특정 리소스만 먼저 삭제
terraform destroy -target=module.ecs
terraform destroy -target=module.alb
```

## 모범 사례

### 1. 모듈화
- 재사용 가능한 단위로 분리
- 각 모듈은 단일 책임만 가짐
- Input/Output을 명확히 정의

### 2. 변수 검증
```hcl
variable "environment" {
  type = string
  validation {
    condition     = contains(["dev", "stage", "prod"], var.environment)
    error_message = "Environment must be dev, stage, or prod."
  }
}
```

### 3. 리소스 태깅
```hcl
default_tags {
  tags = {
    Project     = "RAG-Demo"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Team        = "Platform"
    CostCenter  = "Engineering"
  }
}
```

### 4. 코드 리뷰
```bash
# 포맷 자동 수정
terraform fmt -recursive

# 유효성 검사
terraform validate

# Plan 결과 공유
terraform plan -out=tfplan
```

### 5. 문서화
```hcl
# 변수에 설명 추가
variable "vpc_cidr" {
  description = "CIDR block for VPC. 10.0.0.0/16 provides 65,536 IP addresses."
  type        = string
  default     = "10.0.0.0/16"
}
```

## Terraform vs 수동 배포 비교

| 항목 | 수동 배포 (Section 3-4) | Terraform (Section 5) |
|------|-------------------------|------------------------|
| 재현성 | ❌ 매번 다를 수 있음 | ✅ 100% 재현 가능 |
| 협업 | ❌ 문서화 어려움 | ✅ 코드로 공유 |
| 롤백 | ❌ 수동으로 되돌리기 | ✅ 이전 버전 적용 |
| 변경 추적 | ❌ 어려움 | ✅ Git으로 추적 |
| 리소스 정리 | ❌ 수동으로 하나씩 | ✅ `terraform destroy` |
| 테스트 | ❌ 프로덕션에서만 | ✅ 격리된 환경 생성 |
| 속도 | ⚠️ 느림 (클릭 많음) | ✅ 빠름 (자동화) |

## 다음 단계

Section 5를 완료했다면:
- ✅ Terraform으로 인프라를 코드로 관리하는 방법 습득
- ✅ AWS와 GCP 인프라를 모듈화하여 재사용 가능하게 구성
- ✅ State 관리 및 협업 전략 이해

**Section 6 예고**: GitHub Actions를 이용한 CI/CD 파이프라인
- 코드 푸시 → 자동 테스트 → 자동 빌드 → 자동 배포
- Terraform과 CI/CD 통합
- 프로덕션 환경 안전 배포 전략

## 완료 체크리스트
- [ ] Terraform이 설치되었는가?
- [ ] AWS 인프라가 Terraform으로 프로비저닝되었는가?
- [ ] GCP 인프라가 Terraform으로 프로비저닝되었는가?
- [ ] 모듈 구조를 이해했는가?
- [ ] State 파일의 중요성을 이해했는가?
- [ ] 환경별 변수 관리 전략을 적용했는가?
- [ ] `terraform plan`과 `terraform apply`의 차이를 이해했는가?
- [ ] 비용 관리 전략을 수립했는가?
- [ ] 인프라 변경 사항을 Git으로 추적하고 있는가?
- [ ] `terraform destroy`로 리소스를 정리할 수 있는가?
