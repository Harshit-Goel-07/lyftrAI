import { useState, useRef, useEffect } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function generateSessionId() {
  return 's-' + Math.random().toString(36).substring(2, 10)
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => generateSessionId())
  const [systemPrompt, setSystemPrompt] = useState('')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, errorMsg])

  async function handleSend(e) {
    if (e) e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || loading) return

    setErrorMsg(null)
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }])
    setLoading(true)

    try {
      const payload = {
        session_id: sessionId,
        message: trimmed,
      }
      if (systemPrompt.trim()) {
        payload.system_prompt = systemPrompt.trim()
      }

      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const detail =
          typeof data.detail === 'string'
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d) => d.msg || JSON.stringify(d)).join(', ')
              : `Request failed with status ${res.status}`
        setErrorMsg(`Error (${res.status}): ${detail}`)
        return
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.reply,
          metadata: data.metadata,
        },
      ])
    } catch (err) {
      setErrorMsg(`Network error: ${err.message || 'Could not reach backend'}`)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleNewSession() {
    setSessionId(generateSessionId())
    setMessages([])
    setErrorMsg(null)
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <h1>AI Gateway Chat</h1>
            <span className="session-badge">Session: {sessionId}</span>
          </div>
        </div>
        <button type="button" className="new-session-btn" onClick={handleNewSession}>
          New Session
        </button>
      </header>

      <div className="system-prompt-bar">
        <label htmlFor="system-prompt">System Prompt (optional)</label>
        <input
          id="system-prompt"
          data-testid="system-prompt-input"
          type="text"
          placeholder="e.g. You are a concise technical assistant."
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
        />
      </div>

      <main className="chat-container">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>
              Powered by LiteLLM with automatic key failover, response caching, and SQLite
              multi-turn memory.
            </p>
          </div>
        )}

        {messages.map((msg, idx) =>
          msg.role === 'user' ? (
            <div key={idx} className="message-row user-row">
              <div className="message-bubble user-bubble">{msg.content}</div>
            </div>
          ) : (
            <div key={idx} className="message-row assistant-row">
              <div className="message-bubble assistant-bubble" data-testid="assistant-message">
                <div className="reply-text">{msg.content}</div>
                {msg.metadata && (
                  <div className="meta-bar">
                    <span
                      className={`meta-pill ${
                        msg.metadata.resolution_source === 'LLM_CACHE'
                          ? 'pill-cache'
                          : 'pill-live'
                      }`}
                      data-testid="meta-source"
                    >
                      {msg.metadata.resolution_source}
                    </span>
                    <span className="meta-item" data-testid="meta-model">
                      {msg.metadata.model}
                    </span>
                    <span className="meta-item" data-testid="meta-latency">
                      {msg.metadata.latency_ms} ms
                    </span>
                    <span className="meta-item">
                      in: <strong data-testid="meta-input-tokens">{msg.metadata.input_tokens}</strong>
                    </span>
                    <span className="meta-item">
                      out: <strong data-testid="meta-output-tokens">{msg.metadata.output_tokens}</strong>
                    </span>
                  </div>
                )}
              </div>
            </div>
          ),
        )}

        {loading && (
          <div className="message-row assistant-row">
            <div className="message-bubble assistant-bubble typing">Thinking…</div>
          </div>
        )}

        {errorMsg && (
          <div className="error-banner" role="alert">
            {errorMsg}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <form className="input-bar" onSubmit={handleSend}>
        <textarea
          data-testid="chat-input"
          rows={1}
          placeholder="Type your message… (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          type="submit"
          data-testid="send-button"
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </form>
    </div>
  )
}
