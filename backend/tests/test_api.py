import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    db_file = tmp_path / "test_app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("LLM_API_KEYS", "fake-key-1,fake-key-2")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_mock_response(content: str, cache_hit: bool = False, in_tok: int = 12, out_tok: int = 6):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
        _hidden_params={"cache_hit": cache_hit},
    )


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_chat_validation_errors():
    with TestClient(app) as client:
        res_empty = client.post("/chat", json={"session_id": "s-1", "message": "   "})
        assert res_empty.status_code == 422
        assert "detail" in res_empty.json()

        res_long = client.post("/chat", json={"session_id": "s-1", "message": "a" * 4001})
        assert res_long.status_code == 422
        assert "detail" in res_long.json()

        res_missing = client.post("/chat", json={"message": "hello"})
        assert res_missing.status_code == 422
        assert "detail" in res_missing.json()


def test_chat_live_and_history():
    with TestClient(app) as client:
        with patch("app.llm_service.litellm.completion") as mock_comp:
            mock_comp.return_value = _make_mock_response("Paris", cache_hit=False, in_tok=20, out_tok=5)
            res = client.post(
                "/chat",
                json={"session_id": "s-hist", "message": "What is the capital of France?"},
            )
            assert res.status_code == 200
            body = res.json()
            assert body["session_id"] == "s-hist"
            assert body["reply"] == "Paris"
            assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"
            assert body["metadata"]["key_id"] == "key_1"
            assert body["metadata"]["fallback_attempts"] == 0
            assert body["metadata"]["input_tokens"] == 20
            assert body["metadata"]["output_tokens"] == 5

        hist_res = client.get("/sessions/s-hist/messages")
        assert hist_res.status_code == 200
        msgs = hist_res.json()["messages"]
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "What is the capital of France?"}
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Paris"

        not_found = client.get("/sessions/unknown-session/messages")
        assert not_found.status_code == 404
        assert not_found.json() == {"detail": "session not found"}


def test_chat_cache_hit():
    with TestClient(app) as client:
        with patch("app.llm_service.litellm.completion") as mock_comp:
            mock_comp.return_value = _make_mock_response("Paris", cache_hit=True, in_tok=20, out_tok=5)
            res = client.post(
                "/chat",
                json={"session_id": "s-cache", "message": "Capital of France?"},
            )
            assert res.status_code == 200
            meta = res.json()["metadata"]
            assert meta["resolution_source"] == "LLM_CACHE"
            assert meta["key_id"] is None
            assert meta["fallback_attempts"] == 0


def test_failover_simulation_partial_and_total():
    with TestClient(app) as client:
        with patch("app.llm_service.litellm.completion") as mock_comp:
            mock_comp.return_value = _make_mock_response("Recovered", cache_hit=False)
            res = client.post(
                "/chat",
                json={"session_id": "s-fail", "message": "Hi"},
                headers={"X-Simulate-Failure": "key_1=429"},
            )
            assert res.status_code == 200
            meta = res.json()["metadata"]
            assert meta["key_id"] == "key_2"
            assert meta["fallback_attempts"] == 1

        res_all_fail = client.post(
            "/chat",
            json={"session_id": "s-all-fail", "message": "Hi"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert res_all_fail.status_code == 503
        assert res_all_fail.json() == {"detail": "all providers unavailable"}

        hist_res = client.get("/sessions/s-all-fail/messages")
        assert hist_res.status_code == 404
