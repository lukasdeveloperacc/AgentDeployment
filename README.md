# AgentDeployment - AI 서비스 통합 & 배포 강의

RAG/Agent 시스템의 로컬 개발부터 클라우드 배포까지 전 과정을 학습하는 실습 프로젝트입니다.

## 📚 강의 개요

- **대상**: 1-2년차 개발자, 컴공과 학생
- **목표**: Docker 컨테이너 기반 AI 서비스 배포 실습
- **시간**: 30시간 녹화 → 10시간 편집본
- **레벨**: 초급 ~ 중급

## 🎯 학습 목표

1. ✅ RAG/Agent 시스템 구축 및 로컬 실행
2. ✅ Docker 멀티 컨테이너 구성
3. ✅ AWS ECS/Fargate 배포
4. ✅ GCP Cloud Run + Pinecone 배포
5. ✅ CI/CD 파이프라인 구축
6. ✅ 실전 운영 (비용 최적화, 모니터링, Auto Scaling)

## 📁 프로젝트 구조

```
AgentDeployment/
├── backend/              # FastAPI RAG/Agent 백엔드
│   ├── app.py           # FastAPI 메인 애플리케이션
│   ├── init_chroma.py   # ChromaDB 초기화
│   ├── pyproject.toml   # uv 패키지 관리
│   ├── .env.example     # 환경변수 템플릿
│   └── docs/            # RAG용 샘플 문서 5개
│
├── frontend/             # Vanilla JS 프론트엔드
│   ├── index.html       # 메인 UI (3개 탭)
│   ├── app.js           # SSE 클라이언트
│   └── style.css        # 다크 모드 스타일
│
└── lecture_requirements/ # 강의 자료
    └── 강의목차_확정안.md

```

## 🚀 빠른 시작

### 1. Backend 실행

```bash
cd backend

# .env 파일 생성 및 API Key 입력
cp .env.example .env
# .env 파일 열어서 OPENAI_API_KEY 입력

# uv 의존성 설치
uv sync

# ChromaDB 초기화
uv run python init_chroma.py

# FastAPI 서버 실행
uv run uvicorn app:app --reload --port 8000
```

**접속**: http://localhost:8000/docs

### 2. Frontend 실행

```bash
cd frontend

# Python HTTP 서버 (간단)
python3 -m http.server 3000

# 또는 Node.js http-server
npm install -g http-server
http-server -p 3000 -c-1
```

**접속**: http://localhost:3000

## 💻 기술 스택

### Backend
- **Python 3.11+** / **uv** (패키지 관리)
- **FastAPI** (비동기 웹 프레임워크)
- **LangChain / LangGraph** (LLM 체인 & Agent)
- **ChromaDB** (로컬 Vector Database)
- **OpenAI API** (LLM & Embedding)

### Frontend
- **Vanilla JavaScript** (프레임워크 없이)
- **EventSource API** (SSE 클라이언트)
- **CSS3** (다크 모드, 반응형)

## 📖 주요 기능

### 3가지 인터페이스

1. **Ask 탭**: LLM 직접 호출
   - OpenAI API로 일반 질문 응답
   - 실시간 Streaming

2. **RAG 탭**: 문서 검색 기반 답변
   - ChromaDB에서 관련 문서 검색
   - 검색 결과 + LLM 답변 생성
   - 출처 문서 표시

3. **Agent 탭**: 자동 분류
   - LangGraph Agent가 질문 분류
   - RAG 필요 여부 자동 판단
   - 적절한 경로로 라우팅

### 실시간 Streaming (SSE)
- Server-Sent Events 기반
- 토큰 단위 실시간 응답
- ChatGPT 스타일 타이핑 효과

## 📚 강의 목차

상세 목차는 `lecture_requirements/강의목차_확정안.md` 참조

### Section 0: Docker & 환경변수 기초 (1.5h)
- 컨테이너 개념, Dockerfile, 환경변수 관리

### Section 1: 멀티 컨테이너 로컬 구성 (1.5h)
- docker-compose로 Backend + Frontend + ChromaDB

### Section 2: AWS ECS/Fargate 배포 (2h)
- ECR, Task Definition, ECS Service, ALB

### Section 3: GCP Cloud Run + Pinecone (2h)
- Artifact Registry, Cloud Run, ChromaDB → Pinecone 마이그레이션

### Section 4: CI/CD 파이프라인 (1.5h)
- GitHub Actions로 AWS/GCP 자동 배포

### Section 5: 실전 운영 (1.5h)
- 비용 최적화, Auto Scaling, 로깅/모니터링, 보안

## 🛠️ 개발 환경 설정

### 사전 요구사항

- **Python 3.11+** 설치
- **uv** 설치: https://docs.astral.sh/uv/
- **OpenAI API Key**: https://platform.openai.com/api-keys
- **Git** 설치

### 추천 도구
- **VS Code** (편집기)
- **Docker Desktop** (컨테이너 학습)
- **Postman** or **HTTPie** (API 테스트)

## 📝 환경변수 설정

`backend/.env` 파일:

```bash
# OpenAI API Key (필수!)
OPENAI_API_KEY=sk-proj-your-key-here

# ChromaDB 로컬 저장소
CHROMA_PERSIST_DIR=./chroma_db

# LLM 설정
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# RAG 설정
RAG_TOP_K=3
EMBEDDING_MODEL=text-embedding-3-small

# CORS (Frontend URL)
CORS_ORIGINS=http://localhost:3000
```

## 🧪 테스트

### API 테스트 (curl)

```bash
# Health Check
curl http://localhost:8000/health

# Ask (LLM 직접)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "안녕하세요"}'

# RAG (문서 검색)
curl -X POST "http://localhost:8000/rag" \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG란 무엇인가요?"}'

# Agent (자동 분류)
curl -X POST "http://localhost:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{"question": "Vector Database의 장점은?"}'
```

## 🔧 트러블슈팅

### Backend 관련

**1. `uv` 명령어 없음**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

**2. ChromaDB 초기화 실패**
```bash
# docs 디렉토리 확인
ls backend/docs/
# 5개 문서가 있어야 함
```

**3. OpenAI API Key 에러**
- `.env` 파일에 올바른 키 입력 확인
- `sk-proj-` 또는 `sk-`로 시작하는지 확인

### Frontend 관련

**1. CORS 에러**
- Backend `.env`의 `CORS_ORIGINS` 확인
- Frontend 실행 포트와 일치해야 함

**2. Backend 연결 실패**
- Backend 서버 실행 여부 확인
- `app.js`의 `API_BASE_URL` 확인

## 📖 참고 자료

### Backend
- FastAPI: https://fastapi.tiangolo.com/
- LangChain: https://python.langchain.com/
- ChromaDB: https://docs.trychroma.com/
- uv: https://docs.astral.sh/uv/

### Frontend
- Server-Sent Events: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- EventSource API: https://developer.mozilla.org/en-US/docs/Web/API/EventSource

### 강의 자료
- 강의 목차: `lecture_requirements/강의목차_확정안.md`

## ⚠️ 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- `chroma_db/` 디렉토리는 자동 생성됩니다
- API Key는 타인과 공유하지 마세요
- 프로덕션 배포 시 환경변수는 Secrets Manager 사용

## 📄 라이센스

MIT License - 교육 목적으로 자유롭게 사용 가능합니다.

## 🙋 문의

- 강의 관련 문의: 강의 플랫폼 Q&A
- 버그 제보: GitHub Issues
