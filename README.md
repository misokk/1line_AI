# TeachMe AI 서버

학습자료에서 개념을 분석하고 문제 생성·채점·음성 Teach-back·이해도 평가를 제공하는 FastAPI 서버입니다. 메인 백엔드가 내부 API를 호출하는 구조입니다.

- 문서 인식: Upstage, PDF·PNG·JPEG 지원 및 중복 추출 캐시. TXT는 로컬 처리
- 텍스트 AI: Google 공식 `google-genai` SDK와 Gemini Structured Outputs
- 음성 Teach-back: Gemini Live, 한국어 AI 학생과 양방향 대화
- Python 처리: 객관식·정확 일치 채점, 학습량·예상 시간 계산
- 날짜별 학습 계획: Python으로 진도·취약도에 따른 할 일 배치, DB 저장은 백엔드 담당

## 설치와 실행

Python 3.12를 사용하며 프로젝트 루트에서 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

루트에 `.env`를 직접 만들고 키를 설정합니다. 아래는 처음 실행할 때 생성·음성·기록을 꺼둔 예시이며, 현재 로컬 `.env` 값과 다를 수 있습니다. 실제 키는 저장소에 올리지 않습니다. 모델 이름은 사용하는 계정에서 접근 가능한 값으로 설정합니다.

```dotenv
AI_SERVER_API_KEY=YOUR_SERVER_KEY
GEMINI_API_KEY=YOUR_GEMINI_KEY
UPSTAGE_API_KEY=YOUR_UPSTAGE_KEY
GEMINI_TEXT_MODEL=gemini-3.7-flash
GEMINI_LIVE_MODEL=gemini-3.8-live
AI_USE_SERVER_TEXT_KEY=true
AI_ALLOW_UNLISTED_MODELS=true
AI_LIVE_ENABLED=false
GEMINI_LIVE_ENABLED=false
AI_TIMEOUT_SECONDS=120
AI_ENV=development
AI_RECORD_IO=false
AI_RECORD_CONTENT=false
```

텍스트 생성은 `AI_LIVE_ENABLED=true`, 음성 연결은 `GEMINI_LIVE_ENABLED=true`로 켜고 서버를 재시작합니다. Upstage 인식은 이 두 설정과 별개로 실행될 수 있습니다. 실제 호출에는 사용량·비용이 발생할 수 있으며 자동 재시도·모델 변경은 하지 않습니다.

`GEMINI_TEXT_MODEL`은 기본 텍스트 모델이며 요청의 `model_id`가 우선합니다. 음성은 별도의 `GEMINI_LIVE_MODEL`을 사용하며, 생략하면 `gemini-3.8-live`입니다. 출력 상한은 코드 기본값 8192토큰을 사용하므로 `AI_MAX_OUTPUT_TOKENS`는 변경할 때만 추가합니다. 요청·응답 기록을 켜는 방법은 아래 개발용 기록 항목을 참고합니다.

```powershell
.\.venv\Scripts\python.exe -m api
# 개발 중 자동 재시작이 필요할 때
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

- 상태 확인: http://127.0.0.1:8000/health
- API 문서·테스트: http://127.0.0.1:8000/docs
- 로컬 마이크 테스트: http://127.0.0.1:8000/voice (`AI_ENV=development`에서 제공)
- Linux에서는 `python3.12 -m venv .venv` 후 `.venv/bin/python`으로 같은 명령을 실행합니다.
- `python -m api`는 실행 환경의 `HOST`, `PORT`를 사용합니다. 기본값은 `127.0.0.1:8000`이며 서버 환경에서는 `HOST=0.0.0.0`을 지정할 수 있습니다.

## 인증과 모델 선택

Swagger의 **Authorize → ServerKey**에 `AI_SERVER_API_KEY`를 넣습니다. `/health` 외 내부 HTTP API는 `X-AI-API-Key` 헤더가 필요합니다.

`AI_USE_SERVER_TEXT_KEY=true`이면 서버의 `GEMINI_API_KEY`를 사용합니다. 기본값 `false`에서는 요청에 `X-AI-Key`가 필요하며 Swagger의 **ProviderBYOK**에 설정합니다. 이 헤더를 전달하면 서버 키보다 우선합니다. 운영 프론트엔드에 장기 API 키나 서버 인증키를 전달하지 않습니다.

텍스트 요청의 모델 ID는 `gemini:gemini-3.7-flash`처럼 `gemini:` 접두사를 포함합니다. `GEMINI_TEXT_MODEL`이 기본 목록을 구성하며 `AI_MODEL_ALLOWLIST`, `AI_DEFAULT_MODEL_ID`, `AI_MODEL_ROUTES`로 조정할 수 있습니다. `AI_ALLOW_UNLISTED_MODELS=true`는 목록 밖 Gemini 모델 선택을 허용하지만 실제 접근 권한이나 기능 지원을 보장하지 않습니다. GET `/internal/models`는 설정 목록을 보여주며 `/models/validate-key`는 생성 없이 외부 모델 접근을 확인합니다.

## 내부 API와 데이터 흐름

아래 경로는 `/internal` 기준입니다. 정확한 필수 필드와 응답 타입은 `/docs`에서 확인합니다.

백엔드 연동용 [ai-internal-api.yaml](ai-internal-api.yaml)에 HTTP 요청·응답, 인증·오류, 데이터 저장 책임과 음성 WebSocket 규약을 정리했습니다. OpenAPI 3.1 YAML이며 음성 규약은 `x-websocket`에 있습니다. 기존 사용자용 백엔드 명세와는 별도이고, 코드 변경 시 실제 `/openapi.json`과 대조해 갱신합니다.

백엔드에서 전달한 최종 명세 이후 AI 서버에 학습 계획 생성·혼합 난이도·실시간 음성·상세 평가 등의 추가·변경 사항이 있어 현재 구현 기준의 내부 명세를 함께 제공합니다. 백엔드 연동 시 이 파일의 `x-integration-notes.public_backend_contract_differences`를 참고해 기존 명세와의 차이를 확인해 주세요. 기존 백엔드 최종 명세 자체를 수정한 것은 아닙니다.

| 방식·경로 | 역할 |
|---|---|
| GET `/models` | 설정된 모델 목록 |
| POST `/models/validate-key` | 키·모델 접근 확인 |
| POST `/materials/extract` | 문서 추출 및 청크 초안 |
| POST `/materials/analyze` | 단원·개념·체크리스트·근거 초안 |
| POST `/explanations/generate` | 확정 개념의 원문 기반 설명 |
| POST `/questions/generate` | 문제·정답·해설·근거 생성 |
| POST `/questions/grade` | 문제 1개 채점, 필요한 단답형만 의미 평가 |
| POST `/keywords/suggest` | 문제의 키워드 1~5개 추출 |
| POST `/study-time/estimate` | 남은 날짜에 따른 학습량·권장 시간 추정 |
| POST `/study-plan/generate` | 날짜별 학습·문제 풀이·Teach-back·복습 및 오늘 할 일 |
| POST `/evaluations/generate` | 체크리스트 기반 이해도 평가 |
| POST `/evaluations/detailed` | 평가·발화 인용·교정·개선점·이전 결과 비교 |
| POST `/teachback/sessions` | 음성 세션 생성 |
| GET `/teachback/sessions/{id}` | 세션 상태·전사 기록 조회 |
| WS `/teachback/sessions/{id}/audio` | 음성 송수신 |
| POST `/teachback/sessions/{id}/evaluate` | 세션 사용자 발화 평가 |
| POST `/teachback/sessions/{id}/evaluate-detailed` | 세션 상세 평가, 성공한 기존 결과 재사용 |

1. `extract` 출력 전체를 `analyze`에 그대로 전달하지 않습니다. 백엔드가 `chunk_drafts`에 자료·청크 UUID를 결합해 `SourceChunk`를 구성합니다. `src.material.bind_source_chunks`를 사용할 수 있습니다.
2. `analyze`는 `draft`를 반환합니다. 사용자 확인 후 백엔드가 단원·개념·버전·체크리스트 UUID와 확정 시각을 저장하고, 원문 `chunks`를 포함한 `StudyContext`를 다음 기능에 전달합니다. UUID는 v4이며 근거·페이지·확정 버전 관계를 유지해야 합니다.
3. 문제 생성 HTTP 입력은 5~20개로 검증합니다. 내부 분할 생성·회귀 테스트용 `QuestionRequest`는 작은 배치를 허용합니다. `difficulty="mixed"`이면 문항 수에 맞는 `difficulty_plan`이 필요합니다. 예: `["easy","medium","hard","medium","easy"]`.
   Gemini 3.7/3.8 Flash의 문제 생성에만 `thinking_level=low`를 적용해 추론의 출력 한도 소진을 줄입니다. 다른 기능·모델의 추론 설정은 유지합니다. 난이도별 분할·병합과 문항 수·유형·근거 검증은 유지하며, `MAX_TOKENS` 응답을 자르거나 재조립해 성공으로 처리하지 않습니다.
4. 생성 결과의 `StoredQuestion`에는 정답·해설이 포함됩니다. 백엔드가 비공개로 저장하고 풀이 전에는 `public_question`으로 투영한 공개 문제만 전달합니다.
5. 이해도 평가는 사용자 발화만 사용하며 확정 체크리스트·근거를 검증합니다. `weighted-v1` 점수는 Python으로 계산합니다. 이전 결과를 전달해야 비교가 생성됩니다.
6. 시간 계산은 기준 시간과 입력 진도·취약도에 따른 추정이며 실제 학습 속도나 완료를 보장하지 않습니다. `assumptions`는 계산 기준 설명이고 화면 노출은 프론트엔드에서 결정합니다.

계획 생성은 estimate와 같은 `context`, `total_pages`, `exam_date`, 선택 `today`·`progress`를 받습니다. 확정 개념의 실제 버전·근거 페이지를 유지하며 취약·미학습 개념을 먼저 배치합니다. 학습→문제 풀이→Teach-back 이후, 가능하면 마지막 약 20% 날짜에 복습을 배정하는 초기 규칙입니다. 시험 전날까지 배치하며 시험 당일은 오늘에 압축합니다. 과거 시험일·366일 초과 계획은 거절합니다. `days`, `today_tasks`, 날짜별·전체 예상 시간을 반환하며, 완료한 활동은 `progress`에서 남은 수를 0으로 전달합니다. 일정 변경·완료 이력은 저장하지 않습니다.

DB·사용자 권한·영구 이력·비동기 Job·일정 저장은 AI 서버에서 구현하지 않습니다. 메인 백엔드 공개 API와 `/internal/*`는 별도 계약이며, 계획 생성·혼합 난이도·실시간 음성·상세 평가의 공개 연동 방식은 팀 협의가 필요합니다.

## 음성 테스트

`/voice`에 로컬 서버 인증키와 확정 개념의 세션 입력을 넣습니다. `context`가 포함된 문제 생성 입력 JSON 파일을 선택해도 됩니다. 연결 안내를 확인하고 시작한 뒤 **듣고 있어요**가 표시되면 설명합니다. 종료 후 `status=closed`, `provider_finalized=true`를 확인하고 이해도 평가를 실행합니다. 음성 연결과 평가는 각각 사용량이 발생합니다.

- 새로 시작하면 새 세션입니다. 서버 재시작·만료 시 메모리 세션은 사라집니다.
- 운영 연결은 백엔드의 세션 생성 → 프론트엔드의 일회용 토큰 인증 → WebSocket 순서입니다. 접속 토큰은 60초 유효·1회용입니다.
- 클라이언트는 `ready` 이후 모노 PCM16LE/24kHz를 보내고 종료 시 `{"type":"end"}`를 보냅니다. 서버는 음성·`transcript`·`playback.clear`·`closed` 이벤트를 반환합니다.
- 자막 시각은 서버 수신 기준 근사값입니다. 원본 음성은 저장하지 않으며, 미완성 세션은 평가하지 않습니다. 다중 워커·영구 세션 저장은 지원하지 않습니다.

## 개발용 요청·응답 기록

`AI_ENV=development`, `AI_RECORD_IO=true`일 때 실제 FastAPI 요청·응답을 `input/`, `output/`에 기록합니다. 자료·발화 내용까지 확인하려면 `AI_RECORD_CONTENT=true`를 설정합니다. `production`에서는 기록이 비활성화됩니다.

기능별 고정 파일명에 가장 최근 완료 요청을 저장하며 오류도 이전 결과를 덮어씁니다. 예: `analysis_input.json`/`analysis_output.json`, `questions_generate_input.json`/`questions_generate_output.json`, `study_time_input.json`/`study_time_output.json`, `study_plan_input.json`/`study_plan_output.json`. 본문 없는 GET 입력은 JSON `null`입니다. HTTP 응답 구조는 바꾸지 않습니다.

키·인증정보는 마스킹하고 PDF·이미지는 메타데이터만, 음성은 원본 대신 전사·종료 결과만 기록합니다. 기록 실패는 API 결과에 영향을 주지 않습니다. `.cache/diagnostics/gemini.jsonl`에는 개발용 단계·경과 시간·종료 코드·토큰 수 등 안전한 진단을 별도로 기록합니다.

`.env`, `input/`, `output/`, `.cache/`, `test_data/`, 실제 자료가 포함된 예시 JSON은 Git에서 제외합니다. 내용을 포함한 기록은 개발용 민감 자료로 관리합니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check api src tests
.\.venv\Scripts\python.exe -m pip check
```

자동 테스트는 직접 생성한 자료·Mock HTTP/WebSocket을 사용하며 외부 유료 API를 호출하지 않습니다. 실제 API 검증과 구분합니다. 최소 연결 테스트 `python -m tests.smoke_gemini`도 기본은 오프라인이며, 명시적으로 `--live`를 붙일 때만 실제 호출합니다.

공개 YAML 계약 테스트는 루트 또는 `test_data/swagger-api-final.yaml`을 읽습니다. 두 경로 모두 없으면 해당 테스트를 건너뜁니다. `test_data/`는 Git 제외 대상이므로 새 환경에서는 팀 명세 파일을 별도로 제공해야 합니다. 내부 API 응답 스냅샷은 `tests/fixtures/internal_openapi.json`으로 관리합니다. 공개 YAML은 수정하지 않았으며 새 계획 생성 경로는 내부 API에만 추가했습니다.

## 주요 폴더

```text
api/               FastAPI·인증·모델 선택·Gemini/Upstage 클라이언트·기록·음성 화면
src/               AI 기능·프롬프트·Pydantic 모델·근거 검증·Python 계산
tests/             자동 회귀 테스트·직접 생성한 테스트 자료
requirements.txt   실행·테스트 의존성
pyproject.toml     pytest·Ruff 설정
```

`api/openai.py`와 OpenAI SDK·관련 테스트는 이전 구현 보존용이며 현재 FastAPI 제공자는 Gemini입니다. 메인 백엔드 통합·운영 배포 및 모든 입력에 대한 정확도 검증은 별도 작업입니다.
