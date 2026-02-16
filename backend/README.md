# RAG/Agent Backend - 로컬 실행 가이드

AI 서비스 통합 & 배포 강의를 위한 RAG/Agent 데모 백엔드입니다.

## 📋 주요 기능

- **LLM 직접 호출**: OpenAI API로 일반 질문 응답
- **RAG (Retrieval-Augmented Generation)**: ChromaDB 기반 문서 검색 및 답변 생성
- **LangGraph Agent**: 질문 분류 → RAG/LLM 자동 선택
- **Streaming 응답**: Server-Sent Events (SSE) 기반 실시간 스트리밍

## 🛠️ 기술 스택

- **Python 3.11+**
- **FastAPI**: 비동기 웹 프레임워크
- **LangChain / LangGraph**: LLM 체인 및 Agent 구성
- **ChromaDB**: 로컬 Vector Database
- **OpenAI API**: LLM 및 Embedding

## 📁 프로젝트 구조

```
backend/
├─ app.py                 # FastAPI 메인 애플리케이션
├─ init_chroma.py         # ChromaDB 초기화 스크립트
├─ pyproject.toml         # uv 패키지 관리 설정
├─ .env.example           # 환경변수 템플릿
├─ .env                   # 실제 환경변수 (Git 제외)
├─ docs/                  # RAG용 샘플 문서 5개
│  ├─ 01_RAG_기초.md
│  ├─ 02_Vector_Database.md
│  ├─ 03_LangGraph_Agent.md
│  ├─ 04_Streaming_SSE.md
│  └─ 05_환경변수_관리.md
└─ chroma_db/             # ChromaDB 로컬 저장소 (자동 생성)
```

## 🚀 로컬 실행 방법

### 사전 요구사항

- **Python 3.11 이상** 설치
- **uv** 설치: https://docs.astral.sh/uv/getting-started/installation/
  ```bash
  # macOS/Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **OpenAI API Key** 발급: https://platform.openai.com/api-keys

### 1. 환경변수 설정

```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일을 열어 OpenAI API Key 입력
# macOS/Linux
nano .env

# Windows
notepad .env
```

**`.env` 파일 내용**:
```bash
# OpenAI API Key (필수!)
OPENAI_API_KEY=sk-proj-your-actual-api-key-here

# ChromaDB 설정
CHROMA_PERSIST_DIR=./chroma_db

# LLM 설정
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# RAG 설정
RAG_TOP_K=3
EMBEDDING_MODEL=text-embedding-3-small

# CORS 설정
CORS_ORIGINS=http://localhost:3000
```

### 2. 의존성 설치

```bash
# uv로 가상환경 및 의존성 설치
uv sync
```

### 3. ChromaDB 초기화

```bash
# ChromaDB에 샘플 문서 임베딩
uv run python init_chroma.py
```

**예상 출력**:
```
============================================================
ChromaDB 초기화 시작 (로컬 파일 모드)
============================================================
✓ OpenAI API Key: sk-proj***
✓ ChromaDB persist directory: ./chroma_db

Found 5 markdown files in ./docs
✓ Loaded: 01_RAG_기초.md (8234 characters)
✓ Loaded: 02_Vector_Database.md (7512 characters)
✓ Loaded: 03_LangGraph_Agent.md (9821 characters)
✓ Loaded: 04_Streaming_SSE.md (8934 characters)
✓ Loaded: 05_환경변수_관리.md (7123 characters)

✓ Split 5 documents into 42 chunks

✓ Initializing OpenAI Embeddings...
✓ Creating ChromaDB collection 'ai_service_docs'...
✓ Successfully stored 42 chunks in ./chroma_db

============================================================
검증 테스트
============================================================
Test Query: RAG란 무엇인가요?

✓ Retrieved 2 documents:
[1] Source: 01_RAG_기초.md
    Content: RAG (Retrieval-Augmented Generation) 기초

## RAG란 무엇인가?

RAG(Retrieval-Augmented Generation)는 **검색 증강 생성**을 의미하며...

============================================================
ChromaDB 초기화 완료!
============================================================
```

### 4. FastAPI 서버 실행

```bash
# 개발 서버 실행 (자동 리로드)
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**예상 출력**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
{"time": "2024-02-16 10:00:00", "level": "INFO", "trace_id": "startup", "message": "Application starting..."}
{"time": "2024-02-16 10:00:00", "level": "INFO", "trace_id": "init", "message": "LLM initialized: gpt-4o-mini, API Key: sk-proj***"}
{"time": "2024-02-16 10:00:00", "level": "INFO", "trace_id": "init", "message": "ChromaDB initialized: ./chroma_db"}
{"time": "2024-02-16 10:00:01", "level": "INFO", "trace_id": "startup", "message": "Application started successfully"}
INFO:     Application startup complete.
```

### 5. 접속 및 테스트

- **API Docs (Swagger UI)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Readiness Check**: http://localhost:8000/ready

## 📡 API 엔드포인트

### 1. `/ask` - LLM 직접 호출 (Streaming)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Python의 장점은 무엇인가요?"}'
```

### 2. `/rag` - RAG 기반 답변 (Streaming)
```bash
curl -X POST "http://localhost:8000/rag" \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG란 무엇인가요?"}'
```

### 3. `/agent` - Agent 자동 분류 (Streaming)
```bash
curl -X POST "http://localhost:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{"question": "Vector Database의 장점은?"}'
```

## 🧪 테스트 질문 예시

### Ask 탭 (LLM 직접 호출)
- "안녕하세요"
- "Python에서 리스트와 튜플의 차이는?"
- "오늘 날씨 어때?"

### RAG 탭 (문서 검색 기반)
- "RAG란 무엇인가요?"
- "Vector Database의 종류는?"
- "LangGraph Agent는 어떻게 동작하나요?"
- "SSE와 WebSocket의 차이는?"
- "환경변수 관리 방법은?"

### Agent 탭 (자동 분류)
- "ChromaDB란?" → RAG 경로
- "안녕하세요" → Direct LLM 경로
- "Streaming은 왜 필요한가요?" → RAG 경로

## 🔧 트러블슈팅

### 1. `uv` 명령어를 찾을 수 없음
```bash
# uv 설치 확인
uv --version

# 설치되지 않았다면
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # 또는 ~/.zshrc
```

### 2. OpenAI API Key 에러
```
✗ Error: OPENAI_API_KEY not found in environment
```

**해결**:
- `.env` 파일에 올바른 API Key 입력 확인
- API Key가 `sk-proj-` 또는 `sk-`로 시작하는지 확인

### 3. ChromaDB 초기화 실패
```
✗ No documents found in ./docs directory
```

**해결**:
```bash
# docs 디렉토리 확인
ls -la docs/

# 5개 문서가 있는지 확인
# 없다면 프로젝트 다시 clone
```

### 4. 포트 8000이 이미 사용 중
```bash
# 다른 포트로 실행
uv run uvicorn app:app --reload --port 8001
```

## 📚 다음 단계

1. **Frontend 연동**: `../frontend` 디렉토리에서 프론트엔드 실행
2. **Docker 학습**: Docker 컨테이너화 실습
3. **AWS/GCP 배포**: 클라우드 배포 실습

## ⚠️ 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요!
- `chroma_db/` 디렉토리는 자동 생성되므로 삭제하지 마세요
- API Key는 타인과 공유하지 마세요

## 📖 참고 자료

- FastAPI 문서: https://fastapi.tiangolo.com/
- LangChain 문서: https://python.langchain.com/
- ChromaDB 문서: https://docs.trychroma.com/
- uv 문서: https://docs.astral.sh/uv/

## 📄 라이센스

MIT License - 교육 목적으로 자유롭게 사용 가능합니다.
