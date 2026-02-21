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
4. ✅ Terraform 인프라 코드화 (AWS)
5. ✅ CI/CD 파이프라인 구축 (앱 + 인프라 자동화)
6. ✅ 실전 운영 (비용 최적화, 모니터링)

## 📚 강의 목차

### Section 0: Backend + Frontend 로컬 실행 (1.5h)
- Python/uv 환경 설정, FastAPI 서버 실행, Frontend 연동, Pinecone 초기화

### Section 1: Docker & 환경변수 기초 (1.5h)
- 컨테이너 개념, Dockerfile, 환경변수 관리

### Section 2: 멀티 컨테이너 로컬 구성 (1.5h)
- docker-compose로 Backend + Frontend

### Section 3: AWS ECS/Fargate 배포 (2h)
- ECR, Task Definition, ECS Service, ALB

### Section 4: Terraform으로 인프라 관리 (2h)
- Terraform 기초 (HCL, state, plan/apply)
- AWS 인프라 코드화 (ECS, ALB, ECR, IAM)

### Section 5: CI/CD 파이프라인 (2h)
- GitHub Actions로 앱 배포 + Terraform 인프라 자동화

### Section 6: 실전 운영 (1.5h)
- 비용 최적화, 모니터링

## 📁 프로젝트 구조

```
AgentDeployment/
├─ backend/               # FastAPI RAG/Agent 백엔드
│  ├─ app.py             # FastAPI 메인 애플리케이션
│  ├─ init_pinecone.py   # Pinecone 초기화 스크립트
│  ├─ pyproject.toml     # Python 의존성 (uv)
│  ├─ .env.example       # 환경변수 템플릿
│  ├─ docs/              # RAG용 샘플 문서
│  └─ README.md          # Backend 상세 가이드
├─ frontend/             # Vanilla JS 프론트엔드
│  ├─ index.html         # 메인 UI (3개 탭)
│  ├─ app.js             # SSE 클라이언트 로직
│  ├─ style.css          # 다크 모드 스타일
│  └─ README.md          # Frontend 상세 가이드
└─ README.md             # 이 파일 (전체 프로젝트 개요)
```

## 🚀 빠른 시작 (로컬 실행)

### 사전 요구사항

- **Python 3.11+** 설치
- **uv** 설치: https://docs.astral.sh/uv/
- **OpenAI API Key**: https://platform.openai.com/api-keys
- **Pinecone API Key** (선택사항): https://app.pinecone.io/
- **Git** 설치

### 1단계: Backend 실행

```bash
# backend 디렉토리로 이동
cd backend

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OpenAI API Key 입력 (필수)
# PINECONE_API_KEY는 RAG 기능 사용 시 필요 (선택)

# 의존성 설치
uv sync

# Pinecone 초기화 (선택사항 - RAG 사용 시)
# Pinecone 콘솔에서 인덱스 생성 후:
uv run python init_pinecone.py

# FastAPI 서버 실행
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**접속**: http://localhost:8000/docs (Swagger UI)

> **참고**: `/ask` 엔드포인트는 Pinecone 없이도 작동합니다. RAG 기능(`/rag`, `/agent`)을 사용하려면 Pinecone 설정이 필요합니다.

### 2단계: Frontend 실행 (새 터미널)

```bash
# frontend 디렉토리로 이동
cd frontend

# Python HTTP 서버 실행
python3 -m http.server 3000
```

**접속**: http://localhost:3000

> **대안**: Node.js `http-server` 또는 VS Code Live Server 사용 가능

### 3단계: 테스트

1. **Ask 탭**: "안녕하세요" 입력 → LLM 직접 응답 확인
2. **RAG 탭**: "RAG란 무엇인가요?" 입력 → 문서 검색 기반 응답 확인 (Pinecone 설정 시)
3. **Agent 탭**: 질문 입력 → 자동 분류 경로 확인

## 📖 상세 가이드

- **Backend 설정**: [`backend/README.md`](./backend/README.md)
  - Pinecone 설정, API 엔드포인트, 트러블슈팅
- **Frontend 설정**: [`frontend/README.md`](./frontend/README.md)
  - SSE 통신 방식, UI 사용법, 브라우저 설정

## 🛠️ 기술 스택

### Backend
- **Python 3.11+**: 최신 타입 힌트 및 성능 개선
- **FastAPI**: 비동기 웹 프레임워크
- **LangChain**: LLM 체인 구성
- **Pinecone**: 클라우드 Vector Database (Managed Service)
- **OpenAI API**: GPT-4o-mini, text-embedding-3-small
- **uv**: 빠른 Python 패키지 관리자

### Frontend
- **Vanilla JavaScript**: 프레임워크 없는 순수 JS
- **EventSource API**: Server-Sent Events (SSE)
- **CSS3**: 다크 모드, 반응형 디자인
- **HTML5**: 시맨틱 마크업

## 🔧 트러블슈팅

### Backend 서버가 시작되지 않음
```bash
# OpenAI API Key 확인
cat backend/.env | grep OPENAI_API_KEY

# 포트 충돌 확인 (8000번 포트)
lsof -i :8000

# 다른 포트로 실행
uv run uvicorn app:app --reload --port 8001
```

### Frontend에서 Backend 연결 안 됨 (CORS 에러)
```bash
# Backend .env 파일에서 CORS 설정 확인
CORS_ORIGINS=http://localhost:3000

# Frontend를 다른 포트로 실행 중이라면 해당 포트 추가
CORS_ORIGINS=http://localhost:3000,http://localhost:5500
```

### RAG/Agent 탭이 작동하지 않음
- Pinecone 인덱스가 생성되어 있는지 확인
- Backend 로그에서 "Pinecone not available" 경고 확인
- `/ask` 엔드포인트는 Pinecone 없이도 작동합니다

## ⚠️ 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- API Key는 타인과 공유하지 마세요
- Pinecone 무료 플랜은 1개 인덱스만 생성 가능
- 프로덕션 배포 시 환경변수는 AWS Secrets Manager 사용

## 📄 라이센스

MIT License - 교육 목적으로 자유롭게 사용 가능합니다.

## 🙋 문의

- 강의 관련 문의: 강의 플랫폼 Q&A
- 버그 제보: GitHub Issues
