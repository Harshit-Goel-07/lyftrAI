import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.exceptions import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.db import init_db
from app.main import chat, get_messages, health
from app.schemas import ChatRequest


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    db_file = tmp_path / "test_app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("LLM_API_KEYS", "fake-key-1,fake-key-2")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-3.5-flash-lite")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("LITELLM_TELEMETRY", "False")
    get_settings.cache_clear()
    init_db(str(db_file))
    yield
    get_settings.cache_clear()


def _make_mock_response(content: str, cache_hit: bool = False, in_tok: int = 12, out_tok: int = 6):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
        _hidden_params={"cache_hit": cache_hit},
    )


def test_health():
    res = health()
    assert res.status == "ok"
    assert res.keys_configured == 2
    assert res.cache_enabled is True


def test_chat_validation_errors():
    with pytest.raises(Exception):
        ChatRequest(session_id="s-1", message="   ")

    with pytest.raises(Exception):
        ChatRequest(session_id="s-1", message="a" * 4001)

    with pytest.raises(Exception):
        ChatRequest(message="hello")


def test_chat_live_and_history():
    with patch("app.llm_service.litellm.completion") as mock_comp:
        mock_comp.return_value = _make_mock_response("Paris", cache_hit=False, in_tok=20, out_tok=5)
        req = ChatRequest(session_id="s-hist", message="What is the capital of France?")
        res = chat(req)
        assert res.session_id == "s-hist"
        assert res.reply == "Paris"
        assert res.metadata.resolution_source == "LIVE_INFERENCE"
        assert res.metadata.key_id == "key_1"
        assert res.metadata.fallback_attempts == 0
        assert res.metadata.input_tokens == 20
        assert res.metadata.output_tokens == 5

    hist_res = get_messages("s-hist")
    assert hist_res.session_id == "s-hist"
    msgs = hist_res.messages
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].content == "What is the capital of France?"
    assert msgs[1].role == "assistant"
    assert msgs[1].content == "Paris"

    with pytest.raises(HTTPException) as exc_info:
        get_messages("unknown-session")
    assert exc_info.value.status_code == 404


def test_chat_cache_hit():
    with patch("app.llm_service.litellm.completion") as mock_comp:
        mock_comp.return_value = _make_mock_response("Paris", cache_hit=True, in_tok=20, out_tok=5)
        req = ChatRequest(session_id="s-cache", message="Capital of France?")
        res = chat(req)
        assert res.metadata.resolution_source == "LLM_CACHE"
        assert res.metadata.key_id is None
        assert res.metadata.fallback_attempts == 0


def test_failover_simulation_partial_and_total():
    with patch("app.llm_service.litellm.completion") as mock_comp:
        mock_comp.return_value = _make_mock_response("Recovered", cache_hit=False)
        req = ChatRequest(session_id="s-fail", message="Hi")
        res = chat(req, x_simulate_failure="key_1=429")
        assert res.metadata.key_id == "key_2"
        assert res.metadata.fallback_attempts == 1

    with pytest.raises(HTTPException) as exc_info:
        req_all = ChatRequest(session_id="s-all-fail", message="Hi")
        chat(req_all, x_simulate_failure="*=429")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "all providers unavailable"

    with pytest.raises(HTTPException) as exc_info404:
        get_messages("s-all-fail")
    assert exc_info404.value.status_code == 404
