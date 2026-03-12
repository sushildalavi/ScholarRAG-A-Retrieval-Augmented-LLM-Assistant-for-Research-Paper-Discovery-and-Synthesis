import {
  AnswerResponse,
  ChunkResult,
  DocumentRow,
  EvalCase,
  EvalRunResponse,
  JudgeRunPayload,
  JudgeRunResponse,
  JudgeRunSummary,
  LatestResearchResponse,
  MsaCalibrationPayload,
  MsaCalibrationResponse,
  MsaCalibrationLatest,
} from './types';
import { ChatSession } from './types';

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_BASE ||
  'http://127.0.0.1:8000';

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
    const msg = e?.message || 'Network error';
    if (/Failed to fetch/i.test(msg)) {
      throw new Error(`Backend unreachable at ${API_BASE}`);
    }
    throw new Error(msg);
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

  async askAssistant(payload: {
    query: string;
    scope: 'uploaded' | 'public';
    doc_id?: number;
    doc_ids?: number[];
    k?: number;
    sense?: string;
    compare_senses?: boolean;
    allow_general_background?: boolean;
    run_judge?: boolean;
    run_judge_llm?: boolean;
  }): Promise<AnswerResponse> {
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

  async rawGet(path: string): Promise<any> {
    return jsonRequest(path, { method: 'GET' });
  },

  async runEval(payload: { name?: string; scope?: 'uploaded' | 'public'; k?: number; cases: EvalCase[] }): Promise<EvalRunResponse> {
    return jsonRequest('/eval/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async listEvalRuns(limit = 20): Promise<{ runs: EvalRunResponse[] }> {
    return jsonRequest(`/eval/runs?limit=${limit}`);
  },

  async runJudge(payload: JudgeRunPayload): Promise<JudgeRunResponse> {
    return jsonRequest('/eval/judge', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async listJudgeRuns(limit = 20): Promise<{ runs: JudgeRunSummary[] }> {
    return jsonRequest(`/eval/judge/runs?limit=${limit}`);
  },

  async calibrateConfidence(payload: MsaCalibrationPayload): Promise<MsaCalibrationResponse> {
    return jsonRequest('/confidence/calibrate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async getLatestCalibration(): Promise<MsaCalibrationLatest> {
    return jsonRequest('/confidence/calibration');
  },

  async latestResearch(params?: { topic?: string; limit?: number; days?: number; sort?: 'latest' | 'trending' | 'top_cited' }): Promise<LatestResearchResponse> {
    const search = new URLSearchParams();
    if (params?.topic) search.set('topic', params.topic);
    if (params?.limit) search.set('limit', String(params.limit));
    if (params?.days) search.set('days', String(params.days));
    if (params?.sort) search.set('sort', params.sort);
    const qs = search.toString();
    return jsonRequest(`/research/latest${qs ? `?${qs}` : ''}`);
  },
};
