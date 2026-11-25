# LLM Chatbot with Key Failover and Response Caching

A full-stack AI chatbot application built with **React**, **FastAPI**, **LiteLLM**, and **SQLite**, featuring automatic API key failover, response caching, per-request token tracking, and multi-turn conversation memory.

## 1. Live URLs
- **Frontend:** `<your-deployed-frontend-url>`
- **Backend:** `<your-deployed-backend-url>`

## 2. Run locally

### With Docker (Recommended)
1. Copy `.env.example` to `.env` and fill in your `LLM_API_KEYS` (at least 2 comma-separated keys).
2. Start the stack:
   ```bash
   make up
   ```
   - Frontend: `http://localhost:3000`
   - Backend: `http://localhost:8000`
3. Stop the stack:
   ```bash
   make down
   ```

### Without Docker
**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 3. Design decisions
- **Key fallback:** `LLM_API_KEYS` is parsed into an ordered pool (`key_1`, `key_2`, ...). On HTTP 429, quota errors, or 5xx responses, the gateway catches the error, increments `fallback_attempts`, and retries with the next key. If all keys fail, it returns `503 {"detail": "all providers unavailable"}` without mutating session history.
- **Caching:** LiteLLM's built-in local cache (`Cache(type="local")`) is enabled on `litellm.completion(..., caching=True)`. Cache keys are derived from the model and full message list (system prompt + history + new message). Cache hits return `resolution_source: "LLM_CACHE"`, `key_id: null`, and `fallback_attempts: 0`.
- **SQLite session storage:** Conversation history is persisted in SQLite (`messages` and `sessions` tables) under `DATABASE_URL` (mounted at `/data/app.db` in Docker). Each turn appends the user and assistant messages atomically after a successful LLM response.

## 4. Verification
- **Health check:**
  ```bash
  curl http://localhost:8000/health
  ```
- **Cache hit:** Send the same first message in two different sessions; the second response returns `resolution_source: "LLM_CACHE"` and `key_id: null`.
- **Failover simulation:**
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -H "X-Simulate-Failure: key_1=429" \
    -d '{"session_id":"test-1","message":"Hello"}'
  ```
  Verify `key_id` is `"key_2"` and `fallback_attempts` is `1`. Using `X-Simulate-Failure: *=429` returns `503`.
- **Per-message token usage:** Every assistant reply displays `input_tokens` and `output_tokens` for that single request.

## 5. Setup Used
- Editor: VS Code / Antigravity IDE
- AI Tools: Antigravity

## 6. Bonus features (if any)
- Optional per-session `system_prompt` input persisted in SQLite.
- "New Session" button in the UI to start a fresh conversation without reloading the page.
