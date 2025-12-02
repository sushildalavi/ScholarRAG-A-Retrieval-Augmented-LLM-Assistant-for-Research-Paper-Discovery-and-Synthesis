import { AnswerResponse, ChunkResult, DocumentRow } from './types';
import { ChatSession } from './types';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

async function jsonRequest<T>(path: string, opts: RequestInit = {}): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...opts,
      headers: {
        'Content-Type': 'application/json',
        ...(opts.headers || {}),
      },
    });
    if (!res.ok) {
      throw new Error((await res.text()) || res.statusText);
    }
    return res.json() as Promise<T>;
  } catch (e: any) {
    // Normalize network errors
    throw new Error(e?.message || 'Network error');
  }
}

export const api = {
  async listDocs(limit = 12): Promise<{ documents: DocumentRow[] }> {
    return jsonRequest(`/documents/latest?limit=${limit}`);
  },

  async uploadFile(file: File): Promise<{ document_id: number; pages: number; chunks: number }> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async searchChunks(q: string, k = 10, docId?: number): Promise<{ results: ChunkResult[] }> {
    return jsonRequest('/documents/search/chunks', {
      method: 'POST',
      body: JSON.stringify({ q, k, doc_id: docId }),
    });
  },

  async askAssistant(payload: { query: string; scope: 'uploaded' | 'public'; doc_id?: number; k?: number }): Promise<AnswerResponse> {
    return jsonRequest('/assistant/answer', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async deleteDoc(docId: number): Promise<{ ok: boolean }> {
    const res = await fetch(`${API_BASE}/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async chatSend(payload: { session_id?: number; message?: string; scope?: 'uploaded' | 'public'; doc_id?: number; k?: number; session_only?: boolean }): Promise<ChatSession> {
    return jsonRequest('/assistant/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async chatHistory(session_id: number): Promise<ChatSession> {
    return jsonRequest(`/assistant/chat/${session_id}`, { method: 'GET' });
  },

  async chatUpload(session_id: number, file: File): Promise<{ session_id: number; doc_id?: number }> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/assistant/chat/${session_id}/upload`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};
