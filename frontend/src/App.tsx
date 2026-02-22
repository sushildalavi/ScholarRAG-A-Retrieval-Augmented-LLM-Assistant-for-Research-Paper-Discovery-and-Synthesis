import { Dispatch, SetStateAction, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api/client';
import { Citation, ConfidenceObject, DocumentRow, EvalCase, EvalRunResponse, WhyTraceChunk } from './api/types';
import { SearchBar } from './components/SearchBar';
import { UploadPanel } from './components/UploadPanel';
import { LatestDocumentsList } from './components/LatestDocumentsList';

function confidenceText(c?: ConfidenceObject): string {
  if (!c) return 'No confidence metadata';
  const f = c.factors;
  return `Confidence reflects evidence strength and coverage, penalized for ambiguity/insufficiency. top_sim=${f.top_sim.toFixed(4)}, top_rerank_norm=${f.top_rerank_norm.toFixed(4)}, citation_coverage=${f.citation_coverage.toFixed(4)}, evidence_margin=${f.evidence_margin.toFixed(4)}, ambiguity_penalty=${f.ambiguity_penalty.toFixed(4)}, insufficiency_penalty=${f.insufficiency_penalty.toFixed(4)}.`;
}

function ConfidenceBadge({ confidence }: { confidence?: ConfidenceObject }) {
  if (!confidence) return null;
  if (confidence.needs_clarification) {
    return <div className="ans-confidence low" title={confidenceText(confidence)}>Needs clarification</div>;
  }
  const pct = Math.round((confidence.score || 0) * 100);
  const raw = (confidence.label || 'Low').toLowerCase();
  const cls = raw === 'high' || raw === 'med' || raw === 'low' ? raw : 'low';
  return (
    <div className={`ans-confidence ${cls}`} title={confidenceText(confidence)}>
      <span>{confidence.label}</span>
      <span>{pct}%</span>
    </div>
  );
}

function renderWithInlineCitations(text: string) {
  const normalized = (text || '')
    .replace(/\[(?:S)?(\d+)\]/g, '[$1]')
    .replace(/([.,;:!?])\s*(\[\d+\])\s*([.,;:!?]+)/g, '$1 $2')
    .replace(/(\[\d+\])\s*([.,;:!?]+)/g, '$2 $1')
    .replace(/([.,;:!?])\s*([.,;:!?])+\s*(\[\d+\])/g, '$1 $3')
    .replace(/\s+([.,;:!?])/g, '$1')
    .replace(/(\[\d+\])([A-Za-z0-9])/g, '$1 $2')
    .replace(/([.,;:!?])([A-Za-z0-9])/g, '$1 $2');

  const chunks = normalized.split(/(\[\d+\])/g);
  return chunks.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) return <span key={i}>{part}</span>;
    return (
      <sup key={i} className="inline-cite" title={`Source ${m[1]}`}>
        [{m[1]}]
      </sup>
    );
  });
}

function WhyAnswerSection({ why }: { why?: { rerank_changed_order: boolean; top_chunks: WhyTraceChunk[] } }) {
  return null;
}

function SourceModal({ chunk, onClose }: { chunk: WhyTraceChunk | null; onClose: () => void }) {
  if (!chunk) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>{chunk.title || `Document ${chunk.doc_id || '?'}`}</strong>
          <button onClick={onClose}>Close</button>
        </div>
        <div className="modal-meta">chunk {chunk.chunk_id ?? '?'} • p.{chunk.page ?? '?'}</div>
        <div className="modal-content">{chunk.snippet_preview || 'No snippet available.'}</div>
      </div>
    </div>
  );
}

function SourcesPanel({ citations, traceChunks }: { citations: Citation[]; traceChunks: WhyTraceChunk[] }) {
  const [showCitedOnly, setShowCitedOnly] = useState(true);
  const [modalChunk, setModalChunk] = useState<WhyTraceChunk | null>(null);
  const confidenceById = useMemo(() => {
    const m = new Map<number, ConfidenceObject>();
    (citations || []).forEach((c, i) => {
      const sid = c.id || i + 1;
      if (c.confidence_obj) m.set(sid, c.confidence_obj);
    });
    return m;
  }, [citations]);

  const sourceRows = useMemo(() => {
    const traceById = new Map<number, WhyTraceChunk>();
    (traceChunks || []).forEach((t, i) => {
      const id = t.id || i + 1;
      traceById.set(id, t);
    });
    const base = (citations || []).map((c, i) => {
      const id = c.id || i + 1;
      const trace = traceById.get(id);
      return {
        id,
        title: c.title || trace?.title || `Document ${c.doc_id ?? trace?.doc_id ?? '?'}`,
        url: c.url,
        source: c.source,
        page: c.page ?? trace?.page,
        cited: c.used_in_answer ?? trace?.cited,
        snippet_preview: trace?.snippet_preview || '',
      };
    });
    const filtered = showCitedOnly ? base.filter((x) => x.cited) : base;
    return filtered.length ? filtered : base;
  }, [traceChunks, citations, showCitedOnly]);

  if (!sourceRows.length) return null;

  return (
    <>
      <div className="sources-panel compact">
        <div className="sources-head">
          <div className="sources-title">Sources</div>
          <div className="scope-toggle">
            <button className={showCitedOnly ? 'active' : ''} onClick={() => setShowCitedOnly(true)}>Cited only</button>
            <button className={!showCitedOnly ? 'active' : ''} onClick={() => setShowCitedOnly(false)}>Top retrieved</button>
          </div>
        </div>
        <div className="sources-list">
          {sourceRows.map((c, idx) => (
            <button className="source-item source-item-btn" key={`${c.id || idx}-${idx}`} onClick={() => setModalChunk(c)}>
              <div className="source-row">
                <div className="source-tag-wrap">
                  <span className="source-tag">S{c.id || idx + 1}</span>
                  {c.cited ? <span className="source-scope scope-uploaded-document">Cited</span> : null}
                </div>
                <div className="source-conf-wrap">
                  {confidenceById.get((c.id || idx + 1)) ? (
                    <span
                      className="source-conf"
                      title={confidenceText(confidenceById.get((c.id || idx + 1)))}
                    >
                      {confidenceById.get((c.id || idx + 1))?.label} {Math.round((confidenceById.get((c.id || idx + 1))?.score || 0) * 100)}%
                    </span>
                  ) : <span className="source-conf">Source</span>}
                </div>
              </div>
              <div className="source-name">{c.title || `Document ${c.doc_id || '?'}`}</div>
              <div className="source-sub">
                {(c.source || 'source').toString()}
                {typeof c.page === 'number' ? ` • p.${c.page}` : ''}
                {c.url ? (
                  <>
                    {' • '}
                    <a className="source-link" href={c.url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                      Open source
                    </a>
                  </>
                ) : null}
              </div>
              {c.snippet_preview ? <div className="source-snippet">{c.snippet_preview}</div> : null}
            </button>
          ))}
        </div>
      </div>
      <SourceModal chunk={modalChunk} onClose={() => setModalChunk(null)} />
    </>
  );
}

type UiMessage = {
  role: 'you' | 'assistant';
  text: string;
  streaming?: boolean;
  citations?: Citation[];
  confidence?: ConfidenceObject;
  why_answer?: { rerank_changed_order: boolean; top_chunks: WhyTraceChunk[] };
  latency_breakdown_ms?: { retrieve: number; rerank: number; generate: number; total: number };
  needs_clarification?: boolean;
  clarification?: { question: string; options: string[]; recommended_option?: string } | null;
  answer_scope?: string;
  unsupported_claims?: number;
  query_ref?: string;
};

function isFollowUpQuery(text: string): boolean {
  const q = (text || '').trim().toLowerCase();
  if (!q) return false;
  const cues = [
    'from ieee', 'from springer', 'from elsevier', 'from arxiv', 'from openalex',
    'give from', 'give relevant', 'the one written by', 'that one', 'this one',
    'from semantic scholar',
  ];
  return q.split(/\s+/).length <= 8 || cues.some((c) => q.includes(c));
}

function enrichWithPreviousTopic(current: string, messages: UiMessage[]): string {
  const q = (current || '').trim();
  if (!isFollowUpQuery(q)) return q;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== 'you') continue;
    const prev = (m.text || '').trim();
    if (!prev) continue;
    if (prev.toLowerCase() === q.toLowerCase()) continue;
    if (!isFollowUpQuery(prev) && prev.split(/\s+/).length >= 5) {
      return `${q} about ${prev}`;
    }
  }
  return q;
}

function EvalPage() {
  const [name, setName] = useState('Local eval run');
  const [k, setK] = useState(10);
  const [rawCases, setRawCases] = useState('[\n  {"query":"DES key size", "expected_doc_id": 48}\n]');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<EvalRunResponse | null>(null);
  const [runs, setRuns] = useState<EvalRunResponse[]>([]);

  const loadRuns = async () => {
    const res = await api.listEvalRuns(20);
    setRuns(res.runs || []);
  };

  useEffect(() => {
    loadRuns().catch(() => undefined);
  }, []);

  const runEval = async () => {
    setError('');
    setRunning(true);
    try {
      const parsed = JSON.parse(rawCases) as EvalCase[];
      const res = await api.runEval({ name, scope: 'uploaded', k, cases: parsed });
      setResult(res);
      await loadRuns();
    } catch (e: any) {
      setError(e?.message || 'Failed to run eval');
    } finally {
      setRunning(false);
    }
  };

  const row = (label: string, a: number, b: number) => (
    <tr key={label}>
      <td>{label}</td>
      <td>{a.toFixed(3)}</td>
      <td>{b.toFixed(3)}</td>
    </tr>
  );

  return (
    <div className="eval-page">
      <div className="eval-header">
        <h1>Evaluation</h1>
        <button
          className="ghost"
          onClick={() => {
            window.history.pushState({}, '', '/');
            window.dispatchEvent(new PopStateEvent('popstate'));
          }}
        >
          Back to Studio
        </button>
      </div>
      <div className="eval-grid">
        <div className="eval-card">
          <label>Run name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} />
          <label>Top K</label>
          <input type="number" value={k} onChange={(e) => setK(Number(e.target.value || 10))} />
          <label>Test set JSON (query + expected_doc_id)</label>
          <textarea rows={10} value={rawCases} onChange={(e) => setRawCases(e.target.value)} />
          <button onClick={runEval} disabled={running}>{running ? 'Running...' : 'Run evaluation'}</button>
          {error ? <div className="alert">{error}</div> : null}
        </div>

        <div className="eval-card">
          <h3>Latest result</h3>
          {!result ? <p>No run yet.</p> : (
            <>
              <table className="eval-table">
                <thead>
                  <tr><th>Metric</th><th>Retrieval only</th><th>Retrieval + rerank</th></tr>
                </thead>
                <tbody>
                  {row('Recall@1', result.metrics_retrieval_only.recall_at['1'], result.metrics_retrieval_rerank.recall_at['1'])}
                  {row('Recall@3', result.metrics_retrieval_only.recall_at['3'], result.metrics_retrieval_rerank.recall_at['3'])}
                  {row('Recall@5', result.metrics_retrieval_only.recall_at['5'], result.metrics_retrieval_rerank.recall_at['5'])}
                  {row('Recall@10', result.metrics_retrieval_only.recall_at['10'], result.metrics_retrieval_rerank.recall_at['10'])}
                  {row('MRR', result.metrics_retrieval_only.mrr, result.metrics_retrieval_rerank.mrr)}
                  {row('nDCG@3', result.metrics_retrieval_only.ndcg_at['3'], result.metrics_retrieval_rerank.ndcg_at['3'])}
                  {row('nDCG@5', result.metrics_retrieval_only.ndcg_at['5'], result.metrics_retrieval_rerank.ndcg_at['5'])}
                  {row('nDCG@10', result.metrics_retrieval_only.ndcg_at['10'], result.metrics_retrieval_rerank.ndcg_at['10'])}
                </tbody>
              </table>

              <div className="latency-bars">
                <div>
                  <span>Retrieve avg {result.latency_breakdown.retrieve_ms_avg} ms</span>
                  <div className="bar"><i style={{ width: `${Math.min(100, result.latency_breakdown.retrieve_ms_avg)}%` }} /></div>
                </div>
                <div>
                  <span>Rerank avg {result.latency_breakdown.rerank_ms_avg} ms</span>
                  <div className="bar"><i style={{ width: `${Math.min(100, result.latency_breakdown.rerank_ms_avg)}%` }} /></div>
                </div>
                <div>
                  <span>Generate avg {result.latency_breakdown.generate_ms_avg} ms</span>
                  <div className="bar"><i style={{ width: `${Math.min(100, result.latency_breakdown.generate_ms_avg)}%` }} /></div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="eval-card">
        <h3>Stored runs</h3>
        <div className="run-list">
          {runs.map((r) => (
            <div key={`${r.run_id}-${r.created_at}`} className="run-item">
              <strong>{r.name}</strong>
              <span>{r.created_at}</span>
              <span>cases {r.case_count} • R@5 {r.metrics_retrieval_rerank?.recall_at?.['5'] ?? '-'}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StudioPage() {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<number | null>(null);
  const [questionDoc, setQuestionDoc] = useState('');
  const [allowGeneralBackground, setAllowGeneralBackground] = useState(false);
  const [compareSenses, setCompareSenses] = useState(false);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [errorDoc, setErrorDoc] = useState('');
  const [docMessages, setDocMessages] = useState<UiMessage[]>([]);
  const askDocRef = useRef<HTMLInputElement | null>(null);
  const answersAreaRef = useRef<HTMLDivElement | null>(null);
  const evidenceBodyRef = useRef<HTMLDivElement | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<{ citations: Citation[]; trace: WhyTraceChunk[] }>({ citations: [], trace: [] });
  const streamTimersRef = useRef<number[]>([]);
  const STUDIO_STATE_KEY = 'scholarrag_studio_state_v1';

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STUDIO_STATE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed.docMessages)) setDocMessages(parsed.docMessages);
      if (typeof parsed.questionDoc === 'string') setQuestionDoc(parsed.questionDoc);
      if (parsed.activeEvidence) setActiveEvidence(parsed.activeEvidence);
      if (typeof parsed.selectedDoc === 'number') setSelectedDoc(parsed.selectedDoc);
    } catch {
      // no-op
    }
  }, []);

  const refreshDocs = async () => {
    try {
      const res = await api.listDocs();
      const list = res.documents || [];
      setDocs(list);
      if (selectedDoc && !list.some((d) => d.id === selectedDoc)) setSelectedDoc(null);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    refreshDocs();
  }, []);

  useEffect(() => () => {
    streamTimersRef.current.forEach((id) => window.clearInterval(id));
    streamTimersRef.current = [];
  }, []);

  useEffect(() => {
    try {
      sessionStorage.setItem(
        STUDIO_STATE_KEY,
        JSON.stringify({
          docMessages,
          questionDoc,
          activeEvidence,
          selectedDoc,
        }),
      );
    } catch {
      // no-op
    }
  }, [docMessages, questionDoc, activeEvidence, selectedDoc]);

  useEffect(() => {
    if (!docs.some((d) => d.status === 'processing')) return;
    const id = setInterval(() => { refreshDocs(); }, 2500);
    return () => clearInterval(id);
  }, [docs]);

  useEffect(() => {
    if (!answersAreaRef.current) return;
    answersAreaRef.current.scrollTop = answersAreaRef.current.scrollHeight;
  }, [docMessages, loadingDoc]);

  useEffect(() => {
    if (!evidenceBodyRef.current) return;
    evidenceBodyRef.current.scrollTop = 0;
  }, [activeEvidence]);

  const dedupedDocs = useMemo(() => {
    const seen = new Set<string>();
    const out: DocumentRow[] = [];
    docs.forEach((d) => {
      const key = (d.title || '').toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      out.push(d);
    });
    return out;
  }, [docs]);

  const streamAssistantMessage = (
    setter: Dispatch<SetStateAction<UiMessage[]>>,
    fullText: string,
    payload: Pick<UiMessage, 'citations' | 'confidence' | 'why_answer' | 'latency_breakdown_ms' | 'needs_clarification' | 'clarification' | 'answer_scope' | 'unsupported_claims' | 'query_ref'>,
  ) => {
    const finalText = fullText || 'No response received. Check backend/OpenAI key.';
    setter((msgs) => [...msgs, { role: 'assistant', text: '', streaming: true, ...payload }]);

    let idx = 0;
    const step = Math.max(1, Math.floor(finalText.length / 80));
    const timer = window.setInterval(() => {
      idx += step;
      const partial = finalText.slice(0, idx);
      const done = idx >= finalText.length;
      setter((msgs) => {
        const next = [...msgs];
        for (let i = next.length - 1; i >= 0; i--) {
          const m = next[i];
          if (m.role === 'assistant' && m.streaming) {
            next[i] = { ...m, role: 'assistant', text: partial, streaming: !done };
            break;
          }
        }
        return next;
      });
      if (done) {
        window.clearInterval(timer);
        streamTimersRef.current = streamTimersRef.current.filter((t) => t !== timer);
      }
    }, 18);
    streamTimersRef.current.push(timer);
    setActiveEvidence({ citations: payload.citations || [], trace: payload.why_answer?.top_chunks || [] });
  };

  const ask = async (text: string, k = 10, sense?: string) => {
    const q = text.trim();
    if (!q) return setErrorDoc('Please enter a question.');
    setErrorDoc('');
    setLoadingDoc(true);
    setDocMessages((msgs) => [...msgs, { role: 'you', text: q }]);
    setQuestionDoc('');

    try {
      const hasUploads = Boolean(selectedDoc || docs.length);
      const effectiveScope: 'uploaded' | 'public' = allowGeneralBackground ? 'public' : (hasUploads ? 'uploaded' : 'public');
      const effectiveQuery = allowGeneralBackground ? enrichWithPreviousTopic(q, docMessages) : q;
      const res = await api.askAssistant({
        query: effectiveQuery,
        scope: effectiveScope,
        doc_id: effectiveScope === 'uploaded' && hasUploads && selectedDoc ? selectedDoc : undefined,
        k,
        sense,
        compare_senses: compareSenses,
        allow_general_background: allowGeneralBackground,
      });
      streamAssistantMessage(setDocMessages, (res.answer || res.clarification?.question || '').trim(), {
        citations: res.citations || [],
        confidence: res.confidence,
        why_answer: res.why_answer,
        latency_breakdown_ms: res.latency_breakdown_ms,
        needs_clarification: res.needs_clarification,
        clarification: res.clarification,
        answer_scope: res.answer_scope,
        unsupported_claims: res.unsupported_claims,
        query_ref: effectiveQuery,
      });
    } catch (e: any) {
      setErrorDoc(e?.message || 'Search failed');
      streamAssistantMessage(setDocMessages, 'No response received. Check backend/OpenAI key.', { citations: [] });
    } finally {
      setLoadingDoc(false);
    }
  };

  return (
    <div className="page-three">
      <div className="panel panel-left">
        <div className="brand">
          <span className="brand-icon" aria-hidden="true" />
          <span className="brand-name">Scholar Studio</span>
        </div>
        <div className="section">
          <h3>Upload & Query Docs</h3>
          <UploadPanel onUploaded={refreshDocs} />
        </div>
        <div className="section">
          <h3>Document Chat</h3>
          <LatestDocumentsList
            documents={dedupedDocs}
            selectedId={selectedDoc}
            onSelect={setSelectedDoc}
            onDelete={async (id) => {
              try {
                await api.deleteDoc(id);
                if (selectedDoc === id) setSelectedDoc(null);
                refreshDocs();
              } catch (e) {
                console.error(e);
              }
            }}
          />
        </div>
        <div className="section"><h3>My Library</h3></div>
      </div>

      <div className="panel panel-center">
        <div className="center-inner">
          <div className={`center-hero ${docMessages.length ? 'compact' : ''}`}>
            <div>
              <h1>Connect your research. Get instant answers</h1>
              <p>Upload documents on the left; get doc-grounded answers here.</p>
              <div className="scope-toggle" style={{ marginTop: 8 }}>
                <button className={!allowGeneralBackground ? 'active' : ''} onClick={() => setAllowGeneralBackground(false)}>Docs only</button>
                <button className={allowGeneralBackground ? 'active' : ''} onClick={() => setAllowGeneralBackground(true)}>Allow general background</button>
              </div>
              <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center', marginTop: 8, fontSize: 12, color: '#355a84' }}>
                <input type="checkbox" checked={compareSenses} onChange={(e) => setCompareSenses(e.target.checked)} />
                Compare senses (only when ambiguous)
              </label>
            </div>
            <div className="center-graphic" />
          </div>

          <div className="answers-area" ref={answersAreaRef}>
            {errorDoc && <div className="alert">{errorDoc}</div>}
            {docMessages.length === 0 && !loadingDoc && (
              <div className="empty-state tall">
                <div className="empty-icon" />
                <div>
                  <h3>Ask about your docs</h3>
                  <p>Upload documents and ask a question to see answers here.</p>
                </div>
              </div>
            )}

            {docMessages.map((m, i) => (
              <div key={i} className={`message-group ${m.role}`}>
                <div className={`chat-bubble-row ${m.role}`}>
                  <div
                    className="chat-bubble"
                    onClick={() => {
                      if (m.role === 'assistant') {
                        setActiveEvidence({ citations: m.citations || [], trace: m.why_answer?.top_chunks || [] });
                      }
                    }}
                  >
                    {renderWithInlineCitations(m.text)}
                    {m.streaming && <span className="stream-cursor">▋</span>}
                  </div>
                </div>
                {m.role === 'assistant' && !m.streaming && (
                  <>
                    <div className="assistant-meta-row">
                      <ConfidenceBadge confidence={m.confidence} />
                      {m.answer_scope ? <div className="latency-chip">Scope: {m.answer_scope}</div> : null}
                    </div>
                    {m.needs_clarification && m.clarification?.options?.length ? (
                      <div className="clarify-box">
                        <div className="clarify-q">{m.clarification.question}</div>
                        <div className="clarify-options">
                          {m.clarification.options.map((opt) => (
                            <button key={opt} className="clarify-chip" onClick={() => ask(m.query_ref || '', 10, opt)}>
                              {opt}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </>
                )}
              </div>
            ))}

            {loadingDoc && (
              <div className="empty-state tall">
                <div className="empty-icon" />
                <div>
                  <h3>Thinking…</h3>
                  <p>{allowGeneralBackground ? 'Retrieving from public research sources.' : 'Retrieving from your uploaded documents.'}</p>
                </div>
              </div>
            )}
            <div className="chat-spacer" />
          </div>

          <div className="composer">
            <SearchBar
              value={questionDoc}
              onChange={setQuestionDoc}
              onSubmit={() => ask(questionDoc, 10)}
              disabled={loadingDoc}
              loading={loadingDoc}
              onAdvanced={() => ask(questionDoc, 8)}
              inputRef={askDocRef}
              placeholder="Ask about my docs..."
              hideAdvanced
            />
          </div>
        </div>
      </div>

      <div className="panel panel-right">
        <div className="right-header">
          <h3>Evidence</h3>
        </div>
        <div className="assistant-body evidence-body" ref={evidenceBodyRef}>
          {activeEvidence.citations.length === 0 && activeEvidence.trace.length === 0 ? (
            <div className="empty-state tall">
              <div className="empty-icon" />
              <div>
                <h3>No sources selected</h3>
                <p>Ask a question, then click an answer bubble to inspect evidence here.</p>
              </div>
            </div>
          ) : (
            <SourcesPanel citations={activeEvidence.citations} traceChunks={activeEvidence.trace} />
          )}
        </div>
        <div className="assistant-bottom evidence-foot">
          <div className="evidence-foot-note">
            Confidence = weighted retrieval quality score. Hover confidence badges for factor details.
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return <StudioPage />;
}
