import { Dispatch, SetStateAction, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api/client';
import { Citation, ConfidenceObject, DocumentRow, EvalCase, EvalRunResponse, WhyTraceChunk } from './api/types';
import { JudgeCasePayload, JudgeRunResponse, JudgeRunSummary, MsaCalibrationPayload, MsaCalibrationResponse, MsaCalibrationLatest } from './api/types';
import { SearchBar } from './components/SearchBar';
import { UploadPanel } from './components/UploadPanel';
import { LatestDocumentsList } from './components/LatestDocumentsList';

function confidenceText(c?: ConfidenceObject): string {
  if (!c) return 'No confidence metadata';
  const f = c.factors;
  const msa = f.msa ? ` M=${f.msa.M.toFixed(2)} S=${f.msa.S.toFixed(2)} A=${f.msa.A.toFixed(2)} score=${f.msa.msa_score.toFixed(3)}.` : '';
  return `Confidence reflects evidence strength and coverage, penalized for ambiguity/insufficiency. `
    + `top_sim=${f.top_sim.toFixed(4)}, top_rerank_norm=${f.top_rerank_norm.toFixed(4)}, `
    + `citation_coverage=${f.citation_coverage.toFixed(4)}, evidence_margin=${f.evidence_margin.toFixed(4)}, `
    + `ambiguity_penalty=${f.ambiguity_penalty.toFixed(4)}, insufficiency_penalty=${f.insufficiency_penalty.toFixed(4)}${msa}`;
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

function DeleteDocumentModal({
  document,
  onCancel,
  onConfirm,
}: {
  document: DocumentRow | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  if (!document) return null;
  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-card delete-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>Delete document</strong>
          <button onClick={onCancel}>Close</button>
        </div>
        <div className="modal-content">
          <p>
            You are removing <strong>{document.title}</strong>.
          </p>
          <p>
            Its chunks and embeddings will be deleted, and ScholarRAG will stop using it for retrieval and answers.
          </p>
        </div>
        <div className="delete-modal-actions">
          <button className="ghost" onClick={onCancel}>Cancel</button>
          <button className="danger-btn" onClick={onConfirm}>Delete</button>
        </div>
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
        doc_id: c.doc_id ?? trace?.doc_id,
        msa: c.msa,
        msa_supported: c.msa_supported,
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
                    {c.msa?.msa_score != null ? (
                      <span className="source-conf">
                        MSA {Math.round((c.msa.msa_score || 0) * 100)}%
                        {c.msa_supported ? null : ' (weak)'}
                      </span>
                    ) : (
                      confidenceById.get((c.id || idx + 1)) ? (
                        <span
                          className="source-conf"
                          title={confidenceText(confidenceById.get((c.id || idx + 1)))}
                        >
                          {confidenceById.get((c.id || idx + 1))?.label} {Math.round((confidenceById.get((c.id || idx + 1))?.score || 0) * 100)}%
                        </span>
                      ) : <span className="source-conf">Source</span>
                    )}
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
  faithfulness?: { overall_score: number } | null;
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
  const [judgeScope, setJudgeScope] = useState<'uploaded' | 'public'>('uploaded');
  const [judgeK, setJudgeK] = useState(10);
  const [judgeRawCases, setJudgeRawCases] = useState('[\n  {\"query\":\"What is the main contribution?\", \"scope\":\"uploaded\", \"allow_general_background\": false}\n]');
  const [judgeRunning, setJudgeRunning] = useState(false);
  const [judgeError, setJudgeError] = useState('');
  const [judgeResult, setJudgeResult] = useState<JudgeRunResponse | null>(null);
  const [judgeRuns, setJudgeRuns] = useState<JudgeRunSummary[]>([]);
  const [calibrationRaw, setCalibrationRaw] = useState('{\n  \"records\":[\n    {\"sentence\":\"The model can answer from evidence\", \"evidence\":\"retrieved chunk contains the claim\", \"M\":0.82, \"S\":0.75, \"A\":0.70, \"label\":\"strong\"}\n  ]\n}');
  const [calibrationRunning, setCalibrationRunning] = useState(false);
  const [calibrationError, setCalibrationError] = useState('');
  const [calibration, setCalibration] = useState<MsaCalibrationResponse | null>(null);
  const [calibrationLatest, setCalibrationLatest] = useState<MsaCalibrationLatest | null>(null);

  const loadRuns = async () => {
    const res = await api.listEvalRuns(20);
    setRuns(res.runs || []);
  };
  const loadJudgeRuns = async () => {
    const res = await api.listJudgeRuns(20);
    setJudgeRuns(res.runs || []);
  };
  const loadCalibrationLatest = async () => {
    const latest = await api.getLatestCalibration();
    setCalibrationLatest(latest || null);
  };

  useEffect(() => {
    loadRuns().catch(() => undefined);
    loadJudgeRuns().catch(() => undefined);
    loadCalibrationLatest().catch(() => undefined);
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

  const runJudgeEval = async () => {
    setJudgeError('');
    setJudgeRunning(true);
    try {
      const parsed = JSON.parse(judgeRawCases) as JudgeCasePayload[];
      const res = await api.runJudge({
        scope: judgeScope,
        k: judgeK,
        run_judge_llm: true,
        cases: parsed,
      });
      setJudgeResult(res);
      await loadJudgeRuns();
    } catch (e: any) {
      setJudgeError(e?.message || 'Failed to run judge eval');
    } finally {
      setJudgeRunning(false);
    }
  };

  const runCalibration = async () => {
    setCalibrationError('');
    setCalibrationRunning(true);
    try {
      const parsedPayload = JSON.parse(calibrationRaw) as unknown;
      const records = (() => {
        if (
          parsedPayload &&
          typeof parsedPayload === 'object' &&
          Array.isArray((parsedPayload as { records?: unknown[] }).records)
        ) {
          return (parsedPayload as { records: unknown[] }).records;
        }
        if (Array.isArray(parsedPayload)) {
          return parsedPayload;
        }
        return null;
      })();
      if (!records || records.length === 0) {
        throw new Error('Calibration payload must include a non-empty records array');
      }
      const payload: MsaCalibrationPayload = {
        model_name: 'msa_manual',
        label: 'manual',
        ...(parsedPayload as MsaCalibrationPayload),
        records: records as MsaCalibrationPayload['records'],
      };
      const res = await api.calibrateConfidence(payload);
      setCalibration(res);
      await loadCalibrationLatest();
    } catch (e: any) {
      setCalibrationError(e?.message || 'Failed to run calibration');
    } finally {
      setCalibrationRunning(false);
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
          <label>Judge scope</label>
          <select value={judgeScope} onChange={(e) => setJudgeScope(e.target.value as 'uploaded' | 'public')}>
            <option value="uploaded">uploaded</option>
            <option value="public">public</option>
          </select>
          <label>Judge K</label>
          <input type="number" value={judgeK} onChange={(e) => setJudgeK(Number(e.target.value || 10))} />
          <label>Judge test set JSON (query + optional answer/citations)</label>
          <textarea rows={10} value={judgeRawCases} onChange={(e) => setJudgeRawCases(e.target.value)} />
          <button onClick={runJudgeEval} disabled={judgeRunning}>{judgeRunning ? 'Running judge...' : 'Run LLM-judge eval'}</button>
          {judgeError ? <div className="alert">{judgeError}</div> : null}
          {judgeResult ? (
            <p style={{ marginTop: 12 }}>
              Judge mean score: <strong>{Math.round((judgeResult.metrics.mean_overall_score || 0) * 100)}%</strong> •
              unsupported claims: <strong>{judgeResult.metrics.unsupported_total || 0}</strong>
            </p>
          ) : null}
        </div>

        <div className="eval-card">
          <label>M/S/A calibration records JSON (records / label / M/S/A / weights)</label>
          <textarea rows={12} value={calibrationRaw} onChange={(e) => setCalibrationRaw(e.target.value)} />
          <button onClick={runCalibration} disabled={calibrationRunning}>{calibrationRunning ? 'Calibrating...' : 'Fit MSA calibration'}</button>
          {calibrationError ? <div className="alert">{calibrationError}</div> : null}
          {calibration ? (
            <div style={{ marginTop: 12 }}>
              <p>
                Last fit ({calibration.model_name}) used {calibration.records_used} records.
              </p>
              <p>
                Weights: w1={calibration.weights.w1} w2={calibration.weights.w2} w3={calibration.weights.w3} b={calibration.weights.b}
              </p>
              <p>Brier: {calibration.metrics.brier} • Accuracy: {calibration.metrics.accuracy}</p>
            </div>
          ) : null}
          {calibrationLatest ? (
            <div style={{ marginTop: 12 }}>
              <p>Latest calibration: {calibrationLatest.model_name} ({calibrationLatest.label})</p>
              <p>dataset={calibrationLatest.dataset_size} updated={calibrationLatest.created_at || 'n/a'}</p>
            </div>
          ) : null}
        </div>

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

      <div className="eval-card">
          <h3>Judge runs</h3>
        <div className="run-list">
          {judgeRuns.length === 0 ? <p>No judge runs yet.</p> : null}
          {judgeRuns.map((run, idx) => (
            <div key={`${run.id}-${idx}`} className="run-item">
              <strong>{run.scope || judgeScope}</strong>
              <span>run {run.id}</span>
              <span>query {run.query_count || 0}</span>
              <span>mean overall {Math.round(((run.metrics && run.metrics.mean_overall_score) || 0) * 100)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StudioPage() {
  const [activeView, setActiveView] = useState<'home' | 'library' | 'agent'>('home');
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<number[]>([]);
  const [questionDoc, setQuestionDoc] = useState('');
  const [allowGeneralBackground, setAllowGeneralBackground] = useState(false);
  const [compareSenses, setCompareSenses] = useState(false);
  const [runJudge, setRunJudge] = useState(false);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [errorDoc, setErrorDoc] = useState('');
  const [docMessages, setDocMessages] = useState<UiMessage[]>([]);
  const askDocRef = useRef<HTMLInputElement | null>(null);
  const answersAreaRef = useRef<HTMLDivElement | null>(null);
  const evidenceBodyRef = useRef<HTMLDivElement | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<{ citations: Citation[]; trace: WhyTraceChunk[] }>({ citations: [], trace: [] });
  const [pendingDelete, setPendingDelete] = useState<DocumentRow | null>(null);
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
      if (Array.isArray(parsed.selectedDocs)) setSelectedDocs(parsed.selectedDocs);
    } catch {
      // no-op
    }
  }, []);

  const refreshDocs = async () => {
    try {
      const res = await api.listDocs();
      const list = res.documents || [];
      setDocs(list);
      setSelectedDocs((prev) => prev.filter((id) => list.some((d) => d.id === id)));
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
          selectedDocs,
        }),
      );
    } catch {
      // no-op
    }
  }, [docMessages, questionDoc, activeEvidence, selectedDocs]);

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

  const selectedDocumentRows = useMemo(
    () => dedupedDocs.filter((d) => selectedDocs.includes(d.id)),
    [dedupedDocs, selectedDocs],
  );
  const primarySelectedDocument = selectedDocumentRows[0] || null;
  const fallbackDocument = dedupedDocs[0] || null;
  const activeDocument = primarySelectedDocument || fallbackDocument;
  const hasMultiSelection = selectedDocumentRows.length > 1;
  const selectionSummary = hasMultiSelection
    ? `${selectedDocumentRows.length} selected documents`
    : activeDocument?.title || 'No document selected';

  const summarizePrompt = hasMultiSelection
    ? 'Summarize the selected uploaded documents. Organize the response by document, then provide combined takeaways.'
    : 'Summarize the selected uploaded document.';
  const keyPointsPrompt = hasMultiSelection
    ? 'Extract the key skills, topics, or main points from each selected uploaded document. Keep the response separated by document.'
    : 'What are the key skills or topics listed in the selected document?';
  const evidencePrompt = hasMultiSelection
    ? 'What evidence best supports the main claims in each selected uploaded document? Organize the answer by document.'
    : 'What evidence best supports the main claims in this document?';

  const streamAssistantMessage = (
    setter: Dispatch<SetStateAction<UiMessage[]>>,
    fullText: string,
    payload: Pick<UiMessage, 'citations' | 'confidence' | 'why_answer' | 'latency_breakdown_ms' | 'needs_clarification' | 'clarification' | 'answer_scope' | 'unsupported_claims' | 'query_ref' | 'faithfulness'>,
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

  const ask = async (text: string, k = 10, sense?: string, options?: { skipEnrichment?: boolean }) => {
    const q = text.trim();
    if (!q) return setErrorDoc('Please enter a question.');
    setErrorDoc('');
    setLoadingDoc(true);
    setDocMessages((msgs) => [...msgs, { role: 'you', text: q }]);
    setQuestionDoc('');

    try {
      const hasUploads = Boolean(selectedDocs.length || docs.length);
      const effectiveScope: 'uploaded' | 'public' = allowGeneralBackground ? 'public' : (hasUploads ? 'uploaded' : 'public');
      const effectiveQuery = allowGeneralBackground && !options?.skipEnrichment ? enrichWithPreviousTopic(q, docMessages) : q;
      const res = await api.askAssistant({
        query: effectiveQuery,
        scope: effectiveScope,
        doc_id: effectiveScope === 'uploaded' && selectedDocs.length === 1 ? selectedDocs[0] : undefined,
        doc_ids: effectiveScope === 'uploaded' && selectedDocs.length > 1 ? selectedDocs : undefined,
        k,
        sense,
        compare_senses: compareSenses,
        allow_general_background: allowGeneralBackground,
        run_judge: runJudge,
        run_judge_llm: true,
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
        faithfulness: res.faithfulness,
        query_ref: effectiveQuery,
      });
    } catch (e: any) {
      setErrorDoc(e?.message || 'Search failed');
      streamAssistantMessage(setDocMessages, 'No response received. Check backend/OpenAI key.', { citations: [] });
    } finally {
      setLoadingDoc(false);
    }
  };

  const toggleSelectedDoc = (id: number) => {
    setSelectedDocs((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await api.deleteDoc(pendingDelete.id);
      setSelectedDocs((prev) => prev.filter((id) => id !== pendingDelete.id));
      setActiveEvidence({ citations: [], trace: [] });
      setPendingDelete(null);
      refreshDocs();
    } catch (e) {
      console.error(e);
      setPendingDelete(null);
    }
  };

  return (
    <div className="anara-shell">
      <aside className="anara-sidebar">
        <div className="anara-sidebar-top">
          <div className="workspace-header">
            <div className="workspace-account">
              <div className="workspace-avatar">SR</div>
              <div className="workspace-meta">
                <strong>ScholarRAG</strong>
                <span>Sushil workspace</span>
              </div>
            </div>
            <div className="workspace-toolbar">
              <button className="toolbar-icon" type="button" aria-label="Search workspace">⌕</button>
              <button className="toolbar-icon" type="button" aria-label="Create new">+</button>
            </div>
          </div>

          <nav className="anara-nav">
            <button className={`anara-nav-item ${activeView === 'home' ? 'active' : ''}`} onClick={() => setActiveView('home')}><span>⌂</span><span>Home</span></button>
            <button className={`anara-nav-item ${activeView === 'library' ? 'active' : ''}`} onClick={() => setActiveView('library')}><span>▤</span><span>Library</span></button>
            <button className={`anara-nav-item ${activeView === 'agent' ? 'active' : ''}`} onClick={() => setActiveView('agent')}><span>◌</span><span>Agent</span></button>
          </nav>

          <div className="anara-sidebar-section">
            <div className="sidebar-eyebrow">Private</div>
            <div className="project-card compact">
              <div className="project-card-title">Workspace files</div>
              <div className="project-card-sub">{dedupedDocs.length} indexed</div>
            </div>
          </div>

          <div className="anara-sidebar-section">
            <div className="section-heading-row">
              <h3>Upload</h3>
              <span className="mini-note">PDF ingestion</span>
            </div>
            <UploadPanel onUploaded={refreshDocs} />
          </div>

          <div className="anara-sidebar-section sidebar-library">
            <div className="section-heading-row">
              <h3>Library</h3>
              <span className="mini-note">{selectedDocs.length ? `${selectedDocs.length} selected` : 'Select files'}</span>
            </div>
            <LatestDocumentsList
              documents={dedupedDocs}
              selectedIds={selectedDocs}
              onToggle={toggleSelectedDoc}
              onDelete={(id) => setPendingDelete(dedupedDocs.find((d) => d.id === id) || null)}
            />
          </div>
        </div>

        <div className="sidebar-footer-actions">
          <button
            className="ghost usage-link"
            onClick={() => {
              window.history.pushState({}, '', '/eval');
              window.dispatchEvent(new PopStateEvent('popstate'));
            }}
          >
            Open evaluation
          </button>
        </div>
      </aside>

      <main className="anara-main">
        <header className="anara-topbar">
          <div className="topbar-title">
            <h1>{activeView === 'home' ? 'Chat with your research' : activeView === 'library' ? 'Library' : 'Agent'}</h1>
            <p>
              {activeView === 'home'
                ? (
                  hasMultiSelection
                    ? `Working across ${selectedDocumentRows.length} selected documents`
                    : (activeDocument ? `Working inside ${activeDocument.title}` : 'Select one or more processed files and ask grounded questions.')
                )
                : activeView === 'library'
                  ? 'Browse, select, and remove uploaded documents.'
                  : 'Run quick document workflows over the current file.'}
            </p>
          </div>
          <div className="topbar-actions">
            {activeView === 'home' ? (
              <>
                <div className="scope-toggle">
                  <button className={!allowGeneralBackground ? 'active' : ''} onClick={() => setAllowGeneralBackground(false)}>Docs only</button>
                  <button className={allowGeneralBackground ? 'active' : ''} onClick={() => setAllowGeneralBackground(true)}>Allow background</button>
                </div>
                <label className="toggle-chip">
                  <input type="checkbox" checked={runJudge} onChange={(e) => setRunJudge(e.target.checked)} />
                  <span>Judge</span>
                </label>
              </>
            ) : null}
          </div>
        </header>

        <div className="anara-main-inner">
          {activeView === 'home' ? <div className="studio-grid">
            <section className="conversation-card">
              <div className="conversation-toolbar master">
                <div>
                  <div className="toolbar-title">{hasMultiSelection ? 'Chat across files' : 'Chat with file'}</div>
                  <div className="toolbar-subtitle">
                    {hasMultiSelection
                      ? `Using ${selectedDocumentRows.length} selected uploaded documents`
                      : (activeDocument ? `Focused on ${activeDocument.title}` : 'Using your latest processed uploaded document')}
                  </div>
                </div>
                <div className="conversation-controls">
                  <div className="workspace-mini-actions">
                    <button className="mini-action" type="button" onClick={() => ask(summarizePrompt, 10, undefined, { skipEnrichment: true })}>Summarize {hasMultiSelection ? 'files' : 'file'}</button>
                    <button className="mini-action" type="button" onClick={() => ask(keyPointsPrompt, 10, undefined, { skipEnrichment: true })}>Extract key points</button>
                    <button className="mini-action" type="button" onClick={() => ask(evidencePrompt, 10, undefined, { skipEnrichment: true })}>Inspect evidence</button>
                  </div>
                  <label className="toggle-chip">
                    <input type="checkbox" checked={compareSenses} onChange={(e) => setCompareSenses(e.target.checked)} />
                    <span>Compare senses</span>
                  </label>
                </div>
              </div>

              <div className="answers-area conversation-stream" ref={answersAreaRef}>
                {errorDoc && <div className="alert">{errorDoc}</div>}
                {docMessages.length === 0 && !loadingDoc && (
                  <div className="empty-state workspace-empty">
                    <div className="empty-icon" />
                    <div>
                      <h3>Start with a document question</h3>
                      <p>{hasMultiSelection ? 'Try a cross-document summary, comparison, or evidence request.' : 'Try asking for a summary, skills extraction, methods section, or key findings.'}</p>
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
                          {m.faithfulness ? (
                            <div className="latency-chip">
                              Faithfulness: {Math.round((m.faithfulness.overall_score || 0) * 100)}%
                            </div>
                          ) : null}
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
                  <div className="empty-state workspace-empty">
                    <div className="empty-icon" />
                    <div>
                      <h3>Retrieving evidence</h3>
                      <p>{allowGeneralBackground ? 'Blending public research context with your workspace.' : 'Searching your uploaded documents only.'}</p>
                    </div>
                  </div>
                )}
                <div className="chat-spacer" />
              </div>

              <div className="composer conversation-composer">
                <SearchBar
                  value={questionDoc}
                  onChange={setQuestionDoc}
                  onSubmit={() => ask(questionDoc, 10)}
                  disabled={loadingDoc}
                  loading={loadingDoc}
                  onAdvanced={() => ask(questionDoc, 8)}
                  inputRef={askDocRef}
                  placeholder="Ask about your file, paper, or uploaded evidence..."
                  hideAdvanced
                />
              </div>
            </section>

            <aside className="evidence-column">
              <div className="evidence-card">
                <div className="evidence-headline">
                  <div>
                    <div className="toolbar-title">Evidence</div>
                    <div className="toolbar-subtitle">Inspect retrieved snippets, confidence, and answer support.</div>
                  </div>
                </div>
                <div className="assistant-body evidence-body" ref={evidenceBodyRef}>
                  {activeEvidence.citations.length === 0 && activeEvidence.trace.length === 0 ? (
                    <div className="empty-state workspace-empty">
                      <div className="empty-icon" />
                      <div>
                        <h3>No evidence selected</h3>
                        <p>Click any assistant answer to pin its sources here.</p>
                      </div>
                    </div>
                  ) : (
                    <SourcesPanel citations={activeEvidence.citations} traceChunks={activeEvidence.trace} />
                  )}
                </div>
                <div className="evidence-foot-note">
                  Confidence reflects retrieval quality and support coverage. Hover badges for detailed factors.
                </div>
              </div>
            </aside>
          </div> : null}

          {activeView === 'library' ? (
            <div className="single-pane">
              <div className="single-pane-card">
                <div className="toolbar-title">Document library</div>
                <div className="toolbar-subtitle">Select one or more files to query together, or remove them from the workspace.</div>
                <LatestDocumentsList
                  documents={dedupedDocs}
                  selectedIds={selectedDocs}
                  onToggle={toggleSelectedDoc}
                  onDelete={(id) => setPendingDelete(dedupedDocs.find((d) => d.id === id) || null)}
                />
              </div>
            </div>
          ) : null}

          {activeView === 'agent' ? (
            <div className="single-pane">
              <div className="single-pane-card">
                <div className="toolbar-title">Agent actions</div>
                <div className="toolbar-subtitle">Run fast workflows over the selected document.</div>
                <div className="agent-actions">
                  <button className="agent-card" type="button" onClick={() => ask('Summarize the selected uploaded document with citations.', 10)}>
                    <strong>Summarize</strong>
                    <span>Generate a concise grounded summary.</span>
                  </button>
                  <button className="agent-card" type="button" onClick={() => ask('What are the key skills or concepts in the selected document?', 10)}>
                    <strong>Extract</strong>
                    <span>Pull out important skills, entities, or ideas.</span>
                  </button>
                  <button className="agent-card" type="button" onClick={() => ask('What evidence best supports the main claims in the selected document?', 10)}>
                    <strong>Verify</strong>
                    <span>Surface the strongest supporting evidence.</span>
                  </button>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </main>
      <DeleteDocumentModal document={pendingDelete} onCancel={() => setPendingDelete(null)} onConfirm={confirmDelete} />
    </div>
  );
}

export default function App() {
  return <StudioPage />;
}
