import importlib.util
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader

from api.config import Settings
from api.gemini import LiveClient, diagnostic_event, diagnostic_scope
from api.recording import IORecordingMiddleware
from api.router import request_router
from api.upstage import UpstageDocumentParser
from src import analysis as concept_analysis
from src import (
    evaluation,
    explanation,
    grading,
    material,
    questions,
    study_time,
    tagging,
    teachback,
)
from src.errors import (
    AIError,
    ConfigurationError,
    DocumentProcessingStateError,
    InputError,
    InvalidResponseError,
    KeyValidationUnavailableError,
    ModelAccessError,
    ModelRequiredError,
    ModelSelectionError,
    ProviderBadGatewayError,
    ProviderError,
    ProviderRequestError,
    ProviderServiceUnavailableError,
    RateLimitError,
    RequestTimeoutError,
)
from src.models import (
    AnalyzeRequest,
    DetailedEvaluationRequest,
    DetailedEvaluationResult,
    DraftAnalysis,
    EvaluationRequest,
    EvaluationResult,
    ExplanationContent,
    ExplanationRequest,
    GeneratedQuestions,
    GradeRequest,
    Keywords,
    ModelRequest,
    QuestionGenerationRequest,
    QuestionResult,
    TagRequest,
)
from src.pdf import MAX_PDF_BYTES


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = MAX_PDF_BYTES if scope["path"] == "/internal/materials/extract" else 2 * 1024 * 1024
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                return await JSONResponse({"detail": "Request body exceeds size limit."}, 413)(
                    scope, receive, send
                )
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env(Path(".env"))
    slots = BoundedSemaphore(2)
    document_parser = UpstageDocumentParser(settings)
    voice_sessions = teachback.VoiceSessions()
    live_client = LiveClient(settings)
    server_header = APIKeyHeader(name="X-AI-API-Key", auto_error=False, scheme_name="ServerKey")
    provider_header = APIKeyHeader(name="X-AI-Key", auto_error=False, scheme_name="ProviderBYOK")

    @asynccontextmanager
    async def lifespan(app):
        if not settings.server_api_key.get_secret_value().strip():
            raise ConfigurationError("Set AI_SERVER_API_KEY before starting the server.")
        yield

    app = FastAPI(title="Learning AI internal processing", version="0.3.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(IORecordingMiddleware, settings=settings)

    def authorize(key: Annotated[str | None, Depends(server_header)]):
        expected = settings.server_api_key.get_secret_value()
        if not expected:
            raise HTTPException(503, "Server authentication is not configured.")
        if key is None or not secrets.compare_digest(key.encode(), expected.encode()):
            raise HTTPException(401, "Invalid AI server API key.")

    def capacity():
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "AI processor busy; no request sent to provider.")
        try:
            yield
        finally:
            slots.release()

    protected = [Depends(authorize), Depends(capacity)]

    @app.exception_handler(AIError)
    async def ai_error(request, exc):
        if isinstance(exc, DocumentProcessingStateError):
            code, status = "DOCUMENT_PROCESSING_STATE", 409
        elif isinstance(exc, KeyValidationUnavailableError):
            code, status = "KEY_VALIDATION_UNAVAILABLE", 503
        elif isinstance(exc, ModelSelectionError):
            code, status = "MODEL_NOT_ALLOWED", 422
        elif isinstance(exc, ModelAccessError):
            code, status = "PROVIDER_AUTH", 422
        elif isinstance(exc, InputError):
            code, status = "VALIDATION_ERROR", 422
        elif isinstance(exc, RateLimitError):
            code, status = "RATE_LIMITED", 429
        elif isinstance(exc, ProviderBadGatewayError):
            code, status = "PROVIDER_BAD_GATEWAY", 502
        elif isinstance(exc, ProviderServiceUnavailableError):
            code, status = "PROVIDER_SERVICE_UNAVAILABLE", 503
        elif isinstance(exc, ProviderRequestError):
            code, status = "PROVIDER_REQUEST_INVALID", 502
        elif isinstance(exc, RequestTimeoutError):
            code, status = "PROVIDER_TIMEOUT", 504
        elif isinstance(exc, InvalidResponseError):
            code, status = "INVALID_OUTPUT", 502
        elif isinstance(exc, ConfigurationError):
            code, status = "CONFIGURATION_ERROR", 503
        else:
            code, status = "PROVIDER_UNAVAILABLE", 502
        headers = {}
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            {"error": {"code": code, "message": str(exc)}}, status_code=status, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "details": [{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()],
                }
            },
            status_code=422,
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/internal/models", dependencies=protected)
    def models():
        rows = []
        for model_id, target in settings.model_allowlist.items():
            module = "google.genai" if target.provider == "gemini" else target.provider
            available = (
                target.provider == "gemini"
                and target.available
                and importlib.util.find_spec(module) is not None
            )
            rows.append(
                {
                    "id": model_id,
                    "provider": target.provider,
                    "label": model_id,
                    "available": available,
                }
            )
        return {"data": rows}

    @app.post("/internal/models/validate-key", dependencies=protected)
    def validate_key(request: ModelRequest, key: Annotated[str | None, Depends(provider_header)]):
        if not request.model_id:
            raise InputError("model_id is required.")
        try:
            with request_router(settings, request.model_id, key) as router:
                target = settings.target_for("material_analysis", request.model_id)
                router.clients[target.provider].check_model_access(target.model)
        except (
            ModelAccessError,
            RateLimitError,
            RequestTimeoutError,
            ProviderBadGatewayError,
            ProviderServiceUnavailableError,
        ):
            raise
        except (ProviderError, ConfigurationError, AttributeError):
            raise KeyValidationUnavailableError(
                "Safe provider key validation is unavailable. No generation was attempted."
            ) from None
        return {"valid": True, "model_id": request.model_id}

    @app.post(
        "/internal/materials/extract",
        response_model=material.Extraction,
        dependencies=protected,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    kind: {"schema": {"type": "string", "format": "binary"}}
                    for kind in (
                        "application/pdf",
                        "text/plain",
                        "image/png",
                        "image/jpeg",
                    )
                },
            }
        },
    )
    async def extract(
        request: Request,
        model_id: Annotated[str | None, Query(include_in_schema=False)] = None,
        filename: Annotated[str, Query(max_length=255)] = "material",
        material_id: UUID | None = None,
        retry_failed: bool = False,
        ocr_pdf: Annotated[bool, Query(include_in_schema=False)] = False,
        max_ocr_pages: Annotated[int, Query(include_in_schema=False)] = 5,
        ocr_pages: Annotated[
            list[int] | None,
            Query(
                description="Deprecated: split PDF before upload to select pages.",
                deprecated=True,
                include_in_schema=False,
            ),
        ] = None,
    ):
        from starlette.concurrency import run_in_threadpool

        data = await request.body()
        mime = request.headers.get("content-type", "").split(";")[0]
        return await run_in_threadpool(
            material.extract_material,
            data,
            mime,
            parser=document_parser,
            settings=settings,
            filename=filename,
            material_id=material_id,
            retry_failed=retry_failed,
            ocr_pages=ocr_pages,
        )

    def run(function, request, key):
        with diagnostic_scope(
            settings,
            f"{function.__module__}.{function.__name__}",
            request.model_id,
            secrets=(key,) if key else (),
            credential_source="request_header"
            if key
            else ("server_environment" if settings.use_server_text_key else "missing"),
        ):
            diagnostic_event(settings, "function_started")
            try:
                if not request.model_id:
                    raise InputError("Internal provider requests require explicit model_id.")
                with request_router(settings, request.model_id, key) as router:
                    result = function(request, router)
            except Exception as exc:
                diagnostic_event(settings, "function_failed", exc=exc)
                raise
            diagnostic_event(settings, "function_completed")
            return result

    @app.post(
        "/internal/explanations/generate", response_model=ExplanationContent, dependencies=protected
    )
    def explain(request: ExplanationRequest, key: Annotated[str | None, Depends(provider_header)]):
        return run(explanation.generate, request, key)

    @app.post(
        "/internal/study-time/estimate",
        response_model=study_time.WorkloadEstimate,
        dependencies=protected,
    )
    def workload(request: study_time.WorkloadRequest):
        return study_time.estimate_workload(request)

    @app.post(
        "/internal/study-plan/generate",
        response_model=study_time.StudyPlan,
        dependencies=protected,
    )
    def study_plan(request: study_time.WorkloadRequest):
        return study_time.generate_plan(request)

    @app.post(
        "/internal/evaluations/detailed",
        response_model=DetailedEvaluationResult,
        dependencies=protected,
    )
    def detailed_evaluation(
        request: DetailedEvaluationRequest, key: Annotated[str | None, Depends(provider_header)]
    ):
        return run(evaluation.evaluate_detailed, request, key)

    @app.post("/internal/materials/analyze", response_model=DraftAnalysis, dependencies=protected)
    def analyze(request: AnalyzeRequest, key: Annotated[str | None, Depends(provider_header)]):
        return run(concept_analysis.analyze, request, key)

    @app.post(
        "/internal/questions/generate", response_model=GeneratedQuestions, dependencies=protected
    )
    def generate(
        request: QuestionGenerationRequest, key: Annotated[str | None, Depends(provider_header)]
    ):
        return run(questions.generate, request, key)

    @app.post("/internal/questions/grade", response_model=QuestionResult, dependencies=protected)
    def grade(request: GradeRequest, key: Annotated[str | None, Depends(provider_header)]):
        try:
            return grading.grade(request)
        except ModelRequiredError:
            return run(grading.grade, request, key)

    @app.post(
        "/internal/teachback/sessions",
        response_model=teachback.VoiceSessionCreated,
        dependencies=protected,
    )
    async def create_voice_session(request: teachback.VoiceSessionRequest):
        live_client.check_configuration()
        return voice_sessions.create(request, settings.voice_max_seconds)

    @app.get(
        "/internal/teachback/sessions/{session_id}",
        response_model=teachback.VoiceSessionResult,
        dependencies=protected,
    )
    async def voice_session_result(session_id: UUID):
        return voice_sessions.get(session_id).result()

    async def evaluate_voice(
        session_id: UUID,
        request: teachback.VoiceEvaluationRequest,
        key: Annotated[str | None, Depends(provider_header)],
    ):
        from starlette.concurrency import run_in_threadpool

        session = voice_sessions.get(session_id)
        if session.evaluation is not None:
            comparison = (
                evaluation.compare_reports(session.evaluation.report, request.previous_result)
                if request.previous_result
                else None
            )
            return session.evaluation.model_copy(update={"comparison": comparison})
        if session.evaluating:
            raise HTTPException(409, "Voice evaluation is already running.")
        base = session.evaluation_request(request.model_id)
        evaluation_request = DetailedEvaluationRequest(
            **base.model_dump(), previous_result=request.previous_result
        )
        session.evaluating = True
        try:
            session.evaluation = await run_in_threadpool(
                run, evaluation.evaluate_detailed, evaluation_request, key
            )
            return session.evaluation
        finally:
            session.evaluating = False

    @app.post(
        "/internal/teachback/sessions/{session_id}/evaluate",
        response_model=EvaluationResult,
        dependencies=protected,
    )
    async def evaluate_voice_session(
        session_id: UUID,
        request: teachback.VoiceEvaluationRequest,
        key: Annotated[str | None, Depends(provider_header)],
    ):
        return (await evaluate_voice(session_id, request, key)).report

    @app.post(
        "/internal/teachback/sessions/{session_id}/evaluate-detailed",
        response_model=DetailedEvaluationResult,
        dependencies=protected,
    )
    async def evaluate_voice_session_detailed(
        session_id: UUID,
        request: teachback.VoiceEvaluationRequest,
        key: Annotated[str | None, Depends(provider_header)],
    ):
        return await evaluate_voice(session_id, request, key)

    @app.websocket("/internal/teachback/sessions/{session_id}/audio")
    async def voice_audio(websocket: WebSocket, session_id: UUID):
        import asyncio

        await websocket.accept()
        acquired = False
        try:
            # Browser never receives a long-lived provider or server key from this socket.
            auth = await asyncio.wait_for(websocket.receive_json(), 10)
            if (
                not isinstance(auth, dict)
                or set(auth) != {"type", "connection_token"}
                or auth["type"] != "authenticate"
                or not isinstance(auth["connection_token"], str)
                or len(auth["connection_token"]) > 200
            ):
                await websocket.close(4401)
                return
            if not slots.acquire(blocking=False):
                await websocket.close(4429)
                return
            acquired = True
            session = voice_sessions.claim(session_id, auth["connection_token"])
            await live_client.bridge(websocket, session)
            await websocket.send_json(
                {
                    "type": "closed",
                    "status": session.status,
                    "provider_finalized": session.provider_finalized,
                    "error_code": session.error_code,
                }
            )
            await websocket.close()
        except (InputError, TimeoutError, ValueError):
            await websocket.close(4401)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if acquired:
                slots.release()

    if settings.environment == "development":

        @app.get("/voice", response_class=HTMLResponse, include_in_schema=False)
        def local_voice_page():
            return HTMLResponse(
                Path(__file__).with_name("voice.html").read_text(encoding="utf-8"),
                headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
            )

    @app.post(
        "/internal/evaluations/generate", response_model=EvaluationResult, dependencies=protected
    )
    def evaluate(request: EvaluationRequest, key: Annotated[str | None, Depends(provider_header)]):
        return run(evaluation.evaluate, request, key)

    @app.post("/internal/keywords/suggest", response_model=Keywords, dependencies=protected)
    def keywords(request: TagRequest, key: Annotated[str | None, Depends(provider_header)]):
        return run(tagging.suggest, request, key)

    return app


app = create_app()
