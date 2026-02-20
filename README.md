# AgentDeployment - AI 서비스 통합 & 배포 강의

RAG/Agent 시스템의 로컬 개발부터 클라우드 배포까지 전 과정을 학습하는 실습 프로젝트입니다.

## 📚 강의 개요

- **대상**: 1-2년차 개발자, 컴공과 학생
- **목표**: Docker 컨테이너 기반 AI 서비스 배포 실습
- **시간**: 30시간 녹화 → 10시간 편집본
- **레벨**: 초급 ~ 중급

## 🎯 학습 목표

1. ✅ RAG/Agent 시스템 구축 및 로컬 실행
2. ✅ Docker 컨테이너화 및 멀티 컨테이너 구성
3. ✅ AWS ECS/Fargate 배포
4. ✅ GCP Cloud Run + Pinecone 배포
5. ✅ CI/CD 파이프라인 구축
6. ✅ 실전 운영 (비용 최적화, 모니터링, Auto Scaling)

## 📚 강의 목차

### Section 0: Backend + Frontend 로컬 실행 (1.5h)
- Python/uv 환경 설정, FastAPI 서버 실행, Frontend 연동, Pinecone 초기화

### Section 1: Docker & 환경변수 기초 (1.5h)
- 컨테이너 개념, Dockerfile, 환경변수 관리

### Section 2: 멀티 컨테이너 로컬 구성 (1.5h)
- docker-compose로 Backend + Frontend

### Section 3: AWS ECS/Fargate 배포 (2h)
- ECR, Task Definition, ECS Service, ALB

### Section 4: GCP Cloud Run + Pinecone (2h)
- Artifact Registry, Cloud Run, Pinecone 인덱스 운영

### Section 5: CI/CD 파이프라인 (1.5h)
- GitHub Actions로 AWS/GCP 자동 배포

### Section 6: 실전 운영 (1.5h)
- 비용 최적화, Auto Scaling, 로깅/모니터링, 보안

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
