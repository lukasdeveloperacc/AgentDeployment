# AgentDeployment - AI 서비스 통합 & 배포 강의

RAG/Agent 시스템의 로컬 개발부터 클라우드 배포까지 전 과정을 학습하는 실습 프로젝트입니다.

## 📚 강의 개요

- **대상**: 1-2년차 개발자, 컴공과 학생
- **목표**: Docker 컨테이너 기반 AI 서비스 배포 실습 (AWS + GCP)
- **시간**: 30시간 녹화 → 14시간 편집본
- **레벨**: 초급 ~ 중급

## 📁 주요 파일

- **[`section_details.md`](./section_details.md)**: 세부 커리큘럼 (클립 구성, 이론 파일, 체크리스트)
- **[`lecture_requirements/RAG_Agent_Deployment_Curriculum.xlsx`](./lecture_requirements/RAG_Agent_Deployment_Curriculum.xlsx)**: 전체 커리큘럼 Excel (11개 시트)
  - 0. 전체 개요
  - Section 0~7 (각 섹션별 클립, 이론 파일, 실습)
  - 이론 파일 목록 (51개 PDF)
  - 완주 기준 (DoD)

## 🎯 학습 목표

1. ✅ RAG/Agent 시스템 구축 및 로컬 실행 (Pinecone 연동)
2. ✅ Docker 컨테이너화 및 멀티 컨테이너 구성 (FE + BE)
3. ✅ AWS ECS/Fargate 배포
4. ✅ GCP Cloud Run 배포
5. ✅ Terraform 인프라 코드화 (AWS)
6. ✅ CI/CD 파이프라인 구축 (앱 + 인프라 자동화)
7. ✅ 실전 운영 (비용 최적화, 모니터링, Incident 대응)

## 📚 강의 목차

### Section 0: Backend + Frontend 로컬 실행 (1.5h)
- Python/uv 환경 설정
- FastAPI 서버 실행 및 Swagger UI 확인
- Frontend 연동 및 통합 테스트
- Pinecone 초기화 및 Vector DB 연동
- 3가지 탭 테스트: Ask / RAG / Agent

### Section 1: Docker & 환경변수 기초 (1.5h)
- Docker 기초 개념 (컨테이너 vs VM, 이미지 vs 컨테이너)
- AI 서비스 통합 패턴 (CORS, 프록시)
- **⭐ Streaming SSE 구현** (Server-Sent Events)
- 환경변수 설계 (dev/stage/prod)
- **⭐ LLM API 키 보안** (유출 방지 체크리스트)

### Section 2: 멀티 컨테이너 로컬 구성 (1.5h)
- AI 서비스 패키징 전략
- Backend Dockerfile 작성 (FastAPI)
- Frontend Dockerfile 작성 (Nginx)
- **⭐ docker-compose 구성** (FE + BE 2개 컨테이너)
- Pinecone 클라우드 서비스 연동
- 시크릿 주입 패턴 실습

### Section 3: AWS ECS/Fargate 배포 (2h)
- **⭐ AWS 기초 개념** (VPC, ECS/Fargate, ECR)
- AWS 배포 구성도 (FE + BE)
- ECR 이미지 푸시
- ECS 배포 실습 (Task Definition, Service, ALB)
- **⭐ Pinecone 연동 확인** (Secrets Manager)
- **⭐ AWS 관측 최소 세트** (CloudWatch, 비용 알람)

### Section 4: GCP Cloud Run 배포 (2h)
- **⭐ GCP 기초 개념** (Cloud Run, Artifact Registry)
- AWS ↔ GCP 서비스 매핑
- Artifact Registry 이미지 푸시
- Cloud Run 배포 실습 (FE + BE)
- Secret Manager 시크릿 주입
- Cloud Logging 및 모니터링
- 트래픽 분할 및 롤백

### Section 5: Terraform으로 인프라 관리 (2h)
- Terraform 기초 개념 (IaC, HCL 문법)
- Terraform State 관리 (S3 Backend, DynamoDB Lock)
- AWS ECS Terraform 모듈 작성
- IAM 및 Secrets Manager Terraform 작성
- terraform plan/apply 배포 실습

### Section 6: CI/CD 파이프라인 (2h)
- CI/CD 기초 개념 (GitHub Actions)
- **⭐ 앱 배포 파이프라인 구축** (빌드, 푸시, 배포 자동화)
- Terraform 자동화 (plan on PR, apply on merge)
- 배포 스크립트 템플릿 활용
- 배포 검증 및 모니터링
- 버전 관리 전략

### Section 7: 실전 운영 (1.5h)
- Incident 대응 프로세스
- **⭐ Incident #1**: API Key 유출/누락
- **⭐ Incident #2**: Vector DB 장애
- **⭐ Incident #3**: Streaming 타임아웃
- **⭐ Incident #4**: Agent 무한 루프
- **⭐ Incident #5**: 비용 폭증
- **⭐ Incident #6**: 배포 장애
- 최종 데모 시연 및 운영 문서 작성

> **⭐ 표시**: RAG/Agent AI 서비스 특화 클립

### 📊 전체 통계
- **총 시간**: 14시간
- **이론 파일**: 51개 PPT
- **실습 체크리스트**: Section별 완료 조건
- **산출물**: 배포된 RAG/Agent 서비스 + 운영 문서

> **상세 커리큘럼**: [`section_details.md`](./section_details.md) 참조

## 🛠️ 개발 환경 설정

### 사전 요구사항

- **Python 3.11+** 설치
- **uv** 설치: https://docs.astral.sh/uv/
- **OpenAI API Key**: https://platform.openai.com/api-keys
- **Pinecone API Key**: https://app.pinecone.io/
- **Git** 설치

## ⚠️ 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- API Key는 타인과 공유하지 마세요
- 프로덕션 배포 시 환경변수는 Secrets Manager 사용

## 📄 라이센스

MIT License - 교육 목적으로 자유롭게 사용 가능합니다.

## 🙋 문의

- 강의 관련 문의: 강의 플랫폼 Q&A
- 버그 제보: GitHub Issues
