from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import (
    add_turn_messages,
    check_db_health,
    get_session_messages,
    get_session_system_prompt,
    init_db,
    upsert_session,
)
from app.llm_service import (
    AllProvidersUnavailableError,
    generate_chat_completion,
    init_litellm_cache,
)
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db(settings.sqlite_path)
    init_litellm_cache()
    yield


app = FastAPI(title="LLM Chatbot Gateway", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = get_settings()
    db_ok = check_db_health(cfg.sqlite_path)
    keys = cfg.api_keys_list
    if not db_ok or len(keys) == 0:
        raise HTTPException(status_code=503, detail="service not ready")
    return HealthResponse(
        status="ok",
        keys_configured=len(keys),
        cache_enabled=True,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
) -> ChatResponse:
    if not isinstance(x_simulate_failure, str):
        x_simulate_failure = None
    cfg = get_settings()
    db_path = cfg.sqlite_path

    stored_prompt = get_session_system_prompt(db_path, payload.session_id)
    effective_system_prompt = (
        payload.system_prompt
        if payload.system_prompt is not None
        else (stored_prompt if stored_prompt is not None else cfg.default_system_prompt)
    )

    history = get_session_messages(db_path, payload.session_id)

    llm_messages: List[Dict[str, str]] = []
    if effective_system_prompt:
        llm_messages.append({"role": "system", "content": effective_system_prompt})
    for msg in history:
        llm_messages.append({"role": msg["role"], "content": msg["content"]})
    llm_messages.append({"role": "user", "content": payload.message})

    try:
        reply_text, metadata_dict = generate_chat_completion(
            model=cfg.llm_model,
            messages=llm_messages,
            api_keys=cfg.api_keys_list,
            simulate_failure_header=x_simulate_failure,
            enable_failure_simulation=cfg.enable_failure_simulation,
        )
    except AllProvidersUnavailableError:
        raise HTTPException(status_code=503, detail="all providers unavailable")

    upsert_session(db_path, payload.session_id, payload.system_prompt)
    add_turn_messages(
        db_path=db_path,
        session_id=payload.session_id,
        user_content=payload.message,
        assistant_content=reply_text,
        assistant_metadata=metadata_dict,
    )

    return ChatResponse(
        session_id=payload.session_id,
        reply=reply_text,
        metadata=ChatMetadata(**metadata_dict),
    )


@app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
def get_messages(session_id: str) -> SessionMessagesResponse:
    cfg = get_settings()
    messages = get_session_messages(cfg.sqlite_path, session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionMessagesResponse(session_id=session_id, messages=messages)
