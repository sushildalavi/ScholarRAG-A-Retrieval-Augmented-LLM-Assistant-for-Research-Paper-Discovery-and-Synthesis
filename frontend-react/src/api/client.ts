const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

async function request(path: string, opts: RequestInit = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...opts,
    headers: {
      'Content-Type': 'application/json',
      ...(opts.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export const api = {
  async searchChunks(q: string, k = 10, docId?: number) {
    return request('/documents/search/chunks', {
      method: 'POST',
      body: JSON.stringify({ q, k, doc_id: docId }),
    });
  },
  async qa(q: string, k = 8, docId?: number) {
    return request('/documents/qa', {
      method: 'POST',
      body: JSON.stringify({ q, k, doc_id: docId }),
    });
  },
  async listDocs(limit = 10) {
    return request(`/documents/latest?limit=${limit}`);
  },
};
