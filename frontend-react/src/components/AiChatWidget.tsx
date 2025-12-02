import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { ChatMessage, ChatSession } from '../api/types';

type Props = {
  scope: 'uploaded' | 'public';
  docId?: number | null;
  open?: boolean;
  onOpenChange?: (next: boolean) => void;
};

export function AiChatWidget({ scope, docId, open: controlledOpen, onOpenChange }: Props) {
  const [internalOpen, setInternalOpen] = useState(false);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);

  const open = controlledOpen ?? internalOpen;
  const setOpen = (next: boolean) => {
    if (controlledOpen === undefined) {
      setInternalOpen(next);
    }
    onOpenChange?.(next);
  };

  useEffect(() => {
    const stored = localStorage.getItem('chat_session_id');
    if (stored) {
      const sid = Number(stored);
      if (!Number.isNaN(sid)) {
        setSessionId(sid);
        api.chatHistory(sid).then((res) => setMessages(res.messages || [])).catch(() => {});
      }
    }
  }, []);

  const ensureSession = async () => {
    if (sessionId) return sessionId;
    const res = await api.chatSend({ session_only: true });
    setSessionId(res.session_id);
    localStorage.setItem('chat_session_id', String(res.session_id));
    return res.session_id;
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;
    setError('');
    setLoading(true);
    try {
      const res = await api.chatSend({
        session_id: sessionId || undefined,
        message: text,
        scope,
        doc_id: scope === 'uploaded' ? (docId || undefined) : undefined,
      });
      setSessionId(res.session_id);
      localStorage.setItem('chat_session_id', String(res.session_id));
      setMessages(res.messages || []);
      setInput('');
    } catch (e: any) {
      setError(e?.message || 'Assistant error');
    }
    setLoading(false);
  };

  const uploadFile = async (file: File) => {
    try {
      const sid = sessionId || (await ensureSession());
      const res = await api.chatUpload(sid, file);
      setSessionId(res.session_id);
      localStorage.setItem('chat_session_id', String(res.session_id));
      setError('');
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'assistant',
          content: `Attached ${file.name}. I will use it for answers.`,
        },
      ]);
    } catch (e: any) {
      setError(e?.message || 'Upload failed');
    }
  };

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const scopeLabel = scope === 'uploaded' ? 'Your docs' : 'Public papers';

  return (
    <div className={`chat-floating ${open ? 'open' : ''}`}>
      {!open && (
        <button className="chat-launch" onClick={() => setOpen(true)}>
          💬 AI Assistant
        </button>
      )}
      {open && (
        <div className="chat-widget open animate-in">
          <div className="chat-header">
            <div>
              <div className="chat-title">AI Assistant</div>
              <div className="chat-subtitle">{scopeLabel}{docId ? ` • Doc #${docId}` : ''}</div>
            </div>
            <div className="chat-controls">
              <button className="ghost" onClick={() => fileRef.current?.click()} title="Attach a file">📎</button>
              <button className="ghost" onClick={() => setOpen(false)} title="Minimize">—</button>
            </div>
          </div>
          <div className="chat-body" ref={bodyRef}>
            {messages.map((m) => (
              <div key={m.id} className={`chat-msg ${m.role}`}>
                <div className="chat-bubble">
                  <div className="chat-role">{m.role === 'assistant' ? 'Assistant' : 'You'}</div>
                  <div className="chat-text">{m.content}</div>
                </div>
              </div>
            ))}
            {loading && <div className="chat-msg assistant"><div className="chat-bubble">Thinking…</div></div>}
            {error && <div className="alert">{error}</div>}
          </div>
          <div className="chat-input">
            <button className="ghost" onClick={() => fileRef.current?.click()} title="Attach a file">📎</button>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about research..."
              onKeyDown={(e) => { if (e.key === 'Enter') sendMessage(); }}
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={loading}>{loading ? '…' : 'Send'}</button>
          </div>
          <input
            type="file"
            ref={fileRef}
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                uploadFile(e.target.files[0]);
                e.target.value = '';
              }
            }}
          />
        </div>
      )}
    </div>
  );
}
