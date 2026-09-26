"""FastAPI routes, error envelope, static files."""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from intent.config import Settings
from intent.errors import (
    InputError,
    IntentError,
    InternalError,
    ModelUnavailableError,
    RulesError,
    StorageError,
    UnknownIntentError,
    UnknownLanguageError,
    UnsupportedLanguageError,
)
from intent.pipeline import Pipeline
from intent.rules import load_ruleset, validate_document
from intent.schemas import IntentResult
from intent.service import IntentService, explain


class DetectRequest(BaseModel):
    text: str
    require_supported: bool = False
    language_hint: Optional[str] = None


class BatchRequest(BaseModel):
    texts: list[str]
    require_supported: bool = False


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    require_supported: bool = False
    language_hint: Optional[str] = None


class ValidateRulesRequest(BaseModel):
    language: str
    version: str
    intents: list[dict]
    negators: list[str] = []
    negation_boundaries: list[str] = []


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None
    retryable: bool = False


ERROR_MAP = {
    "input_invalid": (422, InputError),
    "unsupported_language": (422, UnsupportedLanguageError),
    "unknown_language": (404, UnknownLanguageError),
    "unknown_intent": (404, UnknownIntentError),
    "invalid_rules": (422, RulesError),
    "model_unavailable": (503, ModelUnavailableError),
    "storage_error": (503, StorageError),
    "internal": (500, InternalError),
}


def create_app() -> FastAPI:
    """Create the FastAPI application with all routes."""
    settings = Settings.from_env()

    from intent.rules import load_all_rulesets
    from intent.language import load_profiles

    rulesets = load_all_rulesets(settings.languages)
    profiles = load_profiles()

    pipeline = Pipeline(
        rulesets=rulesets,
        profiles=profiles,
        settings=settings,
    )

    from intent.chat import Chatbot
    chatbot = Chatbot()

    service = IntentService(
        pipeline=pipeline,
        chatbot=chatbot,
        db_path=settings.db_path,
    )

    app = FastAPI(
        title="Intent Detection Engine",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.exists():
        app.mount("/js", StaticFiles(directory=web_dir / "js"), name="js")
        # Serve static files directly via routes
        @app.get("/", include_in_schema=False)
        async def root():
            return FileResponse(web_dir / "index.html")
        
        @app.get("/styles.css", include_in_schema=False)
        async def styles():
            return FileResponse(web_dir / "styles.css")

    @app.exception_handler(IntentError)
    async def intent_error_handler(request: Request, exc: IntentError):
        status, _ = ERROR_MAP.get(exc.code, (500, InternalError))
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                code=exc.code,
                message=exc.message,
                details=exc.details,
                retryable=exc.retryable,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="internal",
                message="Erro interno do servidor",
                details={"type": type(exc).__name__},
                retryable=False,
            ).model_dump(),
        )

    @app.get("/")
    async def root():
        index_path = web_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "Intent Detection Engine", "docs": "/docs"}

    @app.get("/api/v1/health")
    async def health():
        return service.health()

    @app.post("/api/v1/detect")
    async def detect(request: DetectRequest):
        if not request.text or not request.text.strip():
            raise InputError("empty text", {"min_chars": settings.min_text_chars})
        if len(request.text.strip()) < settings.min_text_chars:
            raise InputError(
                "text too short",
                {"min_chars": settings.min_text_chars, "actual": len(request.text.strip())},
            )
        if len(request.text.strip()) > settings.max_text_chars:
            raise InputError(
                "text too long",
                {"max_chars": settings.max_text_chars, "actual": len(request.text.strip())},
            )

        result = service.detect(request.text.strip())

        if request.require_supported and result.language and not result.language.supported:
            raise UnsupportedLanguageError(
                "Idioma não suportado",
                {
                    "language": result.language.language,
                    "reason": result.language.reason,
                    "script": result.language.script,
                },
            )

        # IntentResult is frozen, so create a new one with explanation
        return IntentResult(
            text=result.text,
            normalised=result.normalised,
            intents=result.intents,
            primary=result.primary,
            multi_intent=result.multi_intent,
            confidence=result.confidence,
            tokens=result.tokens,
            language=result.language,
            explanation=explain(result),
            rules_version=result.rules_version,
            trace=result.trace,
            resources=result.resources,
            duration_ms=result.duration_ms,
        )

    @app.post("/api/v1/detect/batch")
    async def detect_batch(request: BatchRequest):
        if len(request.texts) > settings.batch_max_items:
            raise InputError(
                "batch too large",
                {"max_items": settings.batch_max_items, "actual": len(request.texts)},
            )

        results = service.detect_batch(request.texts)

        if request.require_supported:
            for result in results:
                if result.language and not result.language.supported:
                    raise UnsupportedLanguageError(
                        "Idioma não suportado no lote",
                        {
                            "language": result.language.language,
                            "reason": result.language.reason,
                            "script": result.language.script,
                        },
                    )

        return [
            IntentResult(
                text=r.text,
                normalised=r.normalised,
                intents=r.intents,
                primary=r.primary,
                multi_intent=r.multi_intent,
                confidence=r.confidence,
                tokens=r.tokens,
                language=r.language,
                explanation=explain(r),
                rules_version=r.rules_version,
                trace=r.trace,
                resources=r.resources,
                duration_ms=r.duration_ms,
            )
            for r in results
        ]

    @app.get("/api/v1/intents")
    async def list_intents(language: Optional[str] = None):
        return service.catalogue(language)

    @app.get("/api/v1/rules/{lang}")
    async def get_rules(lang: str):
        rules = service.rules(lang)
        if rules is None:
            raise UnknownLanguageError(f"Idioma '{lang}' não encontrado")
        return rules

    @app.post("/api/v1/rules/validate")
    async def validate_rules(request: ValidateRulesRequest):
        doc = request.model_dump()
        valid, error = service.validate_rules(doc)
        if not valid:
            raise RulesError("Regras inválidas", {"error": error})
        return {"valid": True}

    @app.get("/api/v1/debug/service-db")
    async def debug_service_db():
        return {"db_path": service.db_path, "db_exists": Path(service.db_path).exists() if service.db_path else False}

    @app.post("/api/v1/chat")
    async def chat(request: ChatRequest):
        if not request.message or not request.message.strip():
            raise InputError("empty message", {"min_chars": settings.min_text_chars})

        response = service.chat(
            request.message.strip(),
            request.session_id,
            request.require_supported,
            request.language_hint,
        )
        return response.model_dump()

    @app.get("/api/v1/conversations")
    async def conversations(limit: int = 20):
        return service.history(limit)

    @app.get("/api/v1/rules/validate-file")
    async def validate_rules_file(path: str):
        try:
            ruleset = load_ruleset(path)
            return {"valid": True, "language": ruleset.language, "version": ruleset.version}
        except Exception as e:
            raise RulesError("Arquivo de regras inválido", {"error": str(e)})

    return app