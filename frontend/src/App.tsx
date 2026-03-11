import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { API_BASE, api } from './api/client';
import {
  Citation, ConfidenceObject, DocumentRow, EvalCase,
  EvalRunResponse, WhyTraceChunk,
} from './api/types';
import {
  JudgeCasePayload, JudgeRunResponse, JudgeRunSummary,
  MsaCalibrationPayload, MsaCalibrationResponse, MsaCalibrationLatest,
} from './api/types';

// ── Types ─────────────────────────────────────────────────────────────────────
type Page = 'studio' | 'eval';

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

type SourceRow = {
  id: number;
  title: string;
  doc_id?: number;
  msa?: { msa_score: number };
  msa_supported?: boolean;
  url?: string;
  source?: string;
  page?: number;
  pages?: number[];
  cited?: boolean;
  citation_count?: number;
  excerpt_count?: number;
  snippet_preview?: string;
  confidence_obj?: ConfidenceObject;
};

type EvidenceState = {
  citations: Citation[];
  trace: WhyTraceChunk[];
};

type StudioSession = {
  id: string;
  title: string;
  messages: UiMessage[];
  selectedDocs: number[];
  activeEvidence: EvidenceState;
  activeEvidenceMsgIdx: number;
  allowGeneralBackground: boolean;
  createdAt: number;
  updatedAt: number;
};

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderInline(text: string): ReactNode {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[S?(\d+)\])/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] != null)      nodes.push(<strong key={m.index}>{m[2]}</strong>);
    else if (m[3] != null) nodes.push(<em key={m.index}>{m[3]}</em>);
    else if (m[4] != null) nodes.push(<code key={m.index}>{m[4]}</code>);
    else if (m[5] != null) nodes.push(<span key={m.index} className="cite-chip">{m[5]}</span>);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes.length === 1 ? nodes[0] : <>{nodes}</>;
}

function renderMarkdown(raw: string): ReactNode {
  const lines = (raw || '').split('\n');
  const out: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) { i++; continue; }

    if (/^[-*_]{3,}$/.test(trimmed)) { out.push(<hr key={i} />); i++; continue; }

    const hm = line.match(/^(#{1,6})\s+(.+)/);
    if (hm) {
      const lvl = Math.min(hm[1].length, 3);
      const Tag = `h${lvl}` as 'h1' | 'h2' | 'h3';
      out.push(<Tag key={i}>{renderInline(hm[2])}</Tag>);
      i++; continue;
    }

    if (line.startsWith('> ')) {
      out.push(<blockquote key={i}>{renderInline(line.slice(2))}</blockquote>);
      i++; continue;
    }

    if (/^[-*+]\s/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^[-*+]\s/.test(lines[i])) {
        items.push(<li key={i}>{renderInline(lines[i].replace(/^[-*+]\s+/, ''))}</li>);
        i++;
      }
      out.push(<ul key={`ul${i}`}>{items}</ul>);
      continue;
    }

    if (/^\d+[.)]\s/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
        items.push(<li key={i}>{renderInline(lines[i].replace(/^\d+[.)]\s+/, ''))}</li>);
        i++;
      }
      out.push(<ol key={`ol${i}`}>{items}</ol>);
      continue;
    }

    const pLines: string[] = [];
    while (i < lines.length) {
      const l = lines[i];
      const t = l.trim();
      if (!t || /^[-*_]{3,}$/.test(t) || /^#{1,6}\s/.test(l) ||
          l.startsWith('> ') || /^[-*+]\s/.test(l) || /^\d+[.)]\s/.test(l)) break;
      pLines.push(l);
      i++;
    }
    if (pLines.length) out.push(<p key={`p${i}`}>{renderInline(pLines.join(' '))}</p>);
  }
  return <div className="md">{out}</div>;
}

// ── Confidence badge ──────────────────────────────────────────────────────────
function confidenceTooltip(c: ConfidenceObject): string {
  const f = c.factors;
  const msa = f.msa ? ` | M=${f.msa.M.toFixed(2)} S=${f.msa.S.toFixed(2)} A=${f.msa.A.toFixed(2)}` : '';
  return `sim=${f.top_sim.toFixed(3)} cov=${f.citation_coverage.toFixed(3)} margin=${f.evidence_margin.toFixed(3)}${msa}`;
}

function ConfBadge({ confidence }: { confidence?: ConfidenceObject }) {
  if (!confidence) return null;
  if (confidence.needs_clarification) {
    return <span className="conf-badge needs-clarification">Clarify</span>;
  }
  const pct = Math.round((confidence.score || 0) * 100);
  const raw = (confidence.label || 'Low').toLowerCase();
  const cls = ['high', 'med', 'low'].includes(raw) ? raw : 'low';
  return (
    <span className={`conf-badge ${cls}`} title={confidenceTooltip(confidence)}>
      {confidence.label} {pct}%
    </span>
  );
}

function formatAnswerScope(scope?: string): string | null {
  if (!scope) return null;
  const normalized = scope.trim().toLowerCase();
  const explicit: Record<string, string> = {
    official_document_context: 'Official document',
    uploaded_document_context: 'Uploaded document',
    personal_document_context: 'Personal document',
    public_research_context: 'Public research',
    mixed_research_context: 'Mixed research',
    context_limited: 'Context limited',
  };
  if (explicit[normalized]) return explicit[normalized];
  return normalized
    .replace(/_context$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── Source card ───────────────────────────────────────────────────────────────
function SourceCard({ row, idx, onClick }: { row: SourceRow; idx: number; onClick: () => void }) {
  const msaScore = row.msa?.msa_score;
  const confPct = msaScore != null
    ? Math.round(msaScore * 100)
    : row.confidence_obj ? Math.round((row.confidence_obj.score || 0) * 100) : null;
  const rawLabel = msaScore != null
    ? (row.msa_supported ? 'high' : 'low')
    : (row.confidence_obj?.label?.toLowerCase() || 'default');
  const confCls = ['high', 'med', 'low'].includes(rawLabel) ? rawLabel : 'default';
  const pages = (row.pages || []).filter((p): p is number => typeof p === 'number');
  const pageLabel =
    pages.length > 1
      ? `pp.${pages.slice(0, 4).join(', ')}${pages.length > 4 ? '…' : ''}`
      : typeof row.page === 'number'
        ? `p.${row.page}`
        : null;

  return (
    <button
      className={`source-card${row.cited ? ' cited' : ''}`}
      onClick={onClick}
      style={{ animationDelay: `${idx * 40}ms` }}
    >
      <div className="sc-overline">
        <span className="sc-source-tag">S{row.id}</span>
        <span className="sc-meta-tag">{String(row.source || 'source')}</span>
        {pageLabel && <span className="sc-meta-tag">{pageLabel}</span>}
        {(row.excerpt_count || 0) > 1 && <span className="sc-meta-tag">{row.excerpt_count} excerpts</span>}
      </div>
      <div className="sc-head">
        <span className="sc-title">{row.title || `Document ${row.doc_id ?? '?'}`}</span>
        {confPct != null && <span className={`sc-conf ${confCls}`}>{confPct}%</span>}
      </div>
      <div className="sc-foot">
        <span className={`sc-foot-state${row.cited ? ' cited' : ''}`}>{row.cited ? 'Cited' : 'Retrieved'}</span>
        {row.url && (
          <a className="sc-link" href={row.url} target="_blank" rel="noreferrer"
             onClick={(e) => e.stopPropagation()}>Open ↗</a>
        )}
      </div>
      {row.snippet_preview && <div className="sc-snippet">{row.snippet_preview}</div>}
    </button>
  );
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="typing-row">
      <div className="msg-avatar assistant">SR</div>
      <div className="typing-bubble">
        <div className="t-dot" />
        <div className="t-dot" />
        <div className="t-dot" />
      </div>
    </div>
  );
}

// ── Source detail modal ───────────────────────────────────────────────────────
function SourceModal({ row, onClose }: { row: SourceRow | null; onClose: () => void }) {
  if (!row) return null;
  const pages = (row.pages || []).filter((p): p is number => typeof p === 'number');
  const pageLabel =
    pages.length > 1
      ? ` · pp.${pages.join(', ')}`
      : typeof row.page === 'number'
        ? ` · p.${row.page}`
        : '';
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-head-title">{row.title || `Document ${row.doc_id ?? '?'}`}</div>
            <div className="modal-head-meta">
              S{row.id} · {String(row.source || 'source')}{pageLabel}
            </div>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">{row.snippet_preview || 'No snippet available.'}</div>
        {row.url && (
          <div className="modal-actions">
            <a className="btn btn-ghost btn-sm" href={row.url} target="_blank" rel="noreferrer">
              Open source ↗
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Delete confirm modal ──────────────────────────────────────────────────────
function DeleteModal({
  doc, onCancel, onConfirm,
}: { doc: DocumentRow | null; onCancel: () => void; onConfirm: () => void }) {
  if (!doc) return null;
  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-head-title">Delete document</div>
          <button className="modal-close" onClick={onCancel}>✕</button>
        </div>
        <div className="modal-body">
          <p>Remove <strong>{doc.title}</strong> from your workspace?</p>
          <p style={{ marginTop: 10 }}>
            All chunks and embeddings will be deleted permanently.
          </p>
        </div>
        <div className="modal-actions">
          <button className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
          <button className="btn btn-danger btn-sm" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  );
}

// ── Evidence panel ────────────────────────────────────────────────────────────
function EvidencePanel({
  citations, traceChunks, loading, allowGeneralBackground,
}: {
  citations: Citation[];
  traceChunks: WhyTraceChunk[];
  loading: boolean;
  allowGeneralBackground: boolean;
}) {
  const [showCitedOnly, setShowCitedOnly] = useState(true);
  const [modalRow, setModalRow] = useState<SourceRow | null>(null);

  const rows = useMemo<SourceRow[]>(() => {
    const traceById = new Map<number, WhyTraceChunk>();
    (traceChunks || []).forEach((t, i) => traceById.set(t.id || i + 1, t));
    const mapped = (citations || []).map((c, i) => {
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
        confidence_obj: c.confidence_obj,
      };
    });
    const deduped = new Map<string, SourceRow>();
    for (const row of mapped) {
      const key = row.doc_id
        ? `uploaded|${row.doc_id}`
        : `${row.source || ''}|${row.url || ''}|${row.title || ''}`;
      const existing = deduped.get(key);
      if (!existing) {
        deduped.set(key, {
          ...row,
          pages: typeof row.page === 'number' ? [row.page] : [],
          citation_count: 1,
          excerpt_count: 1,
        });
        continue;
      }
      const mergedPages = Array.from(new Set([
        ...(existing.pages || (typeof existing.page === 'number' ? [existing.page] : [])),
        ...(typeof row.page === 'number' ? [row.page] : []),
      ])).sort((a, b) => a - b);
      deduped.set(key, {
        ...existing,
        cited: Boolean(existing.cited || row.cited),
        page: mergedPages[0] ?? existing.page ?? row.page,
        pages: mergedPages,
        citation_count: (existing.citation_count || 1) + 1,
        excerpt_count: (existing.excerpt_count || 1) + 1,
        snippet_preview: existing.snippet_preview || row.snippet_preview,
        confidence_obj:
          (row.confidence_obj?.score || 0) > (existing.confidence_obj?.score || 0)
            ? row.confidence_obj
            : existing.confidence_obj,
        msa:
          (row.msa?.msa_score || 0) > (existing.msa?.msa_score || 0)
            ? row.msa
            : existing.msa,
        msa_supported: Boolean(existing.msa_supported || row.msa_supported),
      });
    }
    return Array.from(deduped.values()).map((row, idx) => ({ ...row, id: idx + 1 }));
  }, [citations, traceChunks]);

  const visible = useMemo(() => {
    const filtered = showCitedOnly ? rows.filter((r) => r.cited) : rows;
    return filtered.length ? filtered : rows;
  }, [rows, showCitedOnly]);

  const citedCount = rows.filter((r) => r.cited).length;

  return (
    <>
      <div className="evidence-panel">
        <div className="evidence-head">
          <div className="evidence-head-row">
            <div>
              <div className="evidence-head-kicker">Inspector</div>
              <div className="evidence-head-title">Evidence</div>
              <div className="evidence-head-sub">
                {loading
                  ? (allowGeneralBackground ? 'Searching public sources…' : 'Inspecting retrieved support…')
                  : rows.length > 0
                    ? `${rows.length} sources · ${citedCount} cited`
                    : 'Inspect citations, snippets, and supporting pages for the active answer.'}
              </div>
            </div>
            {!loading && rows.length > 0 && (
              <div className="ev-scope-toggle">
                <button className={showCitedOnly ? 'active' : ''} onClick={() => setShowCitedOnly(true)}>Cited</button>
                <button className={!showCitedOnly ? 'active' : ''} onClick={() => setShowCitedOnly(false)}>All</button>
              </div>
            )}
          </div>
        </div>

        <div className="evidence-body">
          {loading ? (
            <>
              <div style={{ padding: '6px 2px 12px', color: 'var(--text-2)', fontSize: 12 }}>
                {allowGeneralBackground ? 'Searching public sources…' : 'Retrieving evidence…'}
              </div>
              <div className="ev-skeleton-list">
                {[0, 1, 2].map((n) => (
                  <div key={n} className="ev-skeleton-card" style={{ animationDelay: `${n * 80}ms` }}>
                    <div className="sk-row sk-full" />
                    <div className="sk-row sk-3q" />
                    <div className="sk-row sk-half" />
                  </div>
                ))}
              </div>
            </>
          ) : visible.length === 0 ? (
            <div className="evidence-empty">
              <div className="evidence-empty-icon">⬡</div>
              <div className="evidence-empty-text">
                Click any assistant reply to inspect its grounding sources here.
              </div>
            </div>
          ) : (
            visible.map((row, idx) => (
              <SourceCard key={`${row.id}-${idx}`} row={row} idx={idx} onClick={() => setModalRow(row)} />
            ))
          )}
        </div>

      </div>
      <SourceModal row={modalRow} onClose={() => setModalRow(null)} />
    </>
  );
}

// ── Follow-up query helpers ───────────────────────────────────────────────────
function isFollowUp(text: string): boolean {
  const q = text.trim().toLowerCase();
  if (!q) return false;
  const cues = [
    'from ieee', 'from springer', 'from elsevier', 'from arxiv',
    'that one', 'this one', 'give me papers', 'relevant papers', 'more info',
  ];
  return q.split(/\s+/).length <= 8 || cues.some((c) => q.includes(c));
}

function enrichQuery(current: string, msgs: UiMessage[]): string {
  const q = current.trim();
  if (!isFollowUp(q)) return q;
  const wantsPapers = /\b(papers?|research|studies|references?|surveys?)\b/i.test(q);
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m.role !== 'you' || !m.text.trim()) continue;
    if (m.text.toLowerCase() === q.toLowerCase()) continue;
    if (!isFollowUp(m.text) && m.text.split(/\s+/).length >= 5) {
      return wantsPapers
        ? `Find relevant research papers about ${m.text}. Include foundational, survey, and highly relevant papers.`
        : `${q} about ${m.text}`;
    }
  }
  return q;
}

function isExplicitDocumentQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  if (!q) return false;
  return (
    /\b(this|these|selected|uploaded)\s+(document|documents|file|files|pdf|pdfs)\b/.test(q) ||
    /\bin\s+(this|these)\s+(document|documents|file|files)\b/.test(q) ||
    /\b(across|from)\s+(these|selected)\s+(documents|files)\b/.test(q)
  );
}

function isGreetingQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  return /^(hi|hello|hey|yo|sup|wassup|what'?s up|howdy|good morning|good afternoon|good evening|hola)[!.?]*$/.test(q);
}

function isAssistantSetupQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  return (
    /\b(what can you do|how can you help)\b/.test(q) ||
    /\b(help me with research|research based questions|research questions)\b/.test(q) ||
    /\b(answer my research based questions)\b/.test(q)
  );
}

function isLiteratureQuery(text: string): boolean {
  const q = text.trim().toLowerCase();
  return (
    /\b(show me papers|find papers|give me papers|list papers|relevant papers|sources|citations|bibliography)\b/.test(q) ||
    /\b(in the literature|recent papers|recent research|what do papers say)\b/.test(q)
  );
}

function buildAssistantIntroReply(kind: 'greeting' | 'setup', hasUploads: boolean): string {
  const intro = kind === 'greeting'
    ? 'Hi. I can help with concept explanations, paper discovery, research synthesis, and document-grounded analysis.'
    : 'I can explain concepts, find papers, compare findings across the literature, and analyze uploaded documents.';
  const thirdPrompt = hasUploads
    ? 'Select a document and ask for key findings or supporting evidence'
    : 'Upload a paper and ask for a grounded summary';
  return `${intro}

Try one of these:
- Tell me about RNNs
- Show me papers on attention mechanisms
- ${thirdPrompt}`;
}

// ── Eval page ─────────────────────────────────────────────────────────────────
function EvalPage({ onBack }: { onBack: () => void }) {
  const [name, setName] = useState('Local eval run');
  const [k, setK] = useState(10);
  const [rawCases, setRawCases] = useState('[\n  {"query":"DES key size", "expected_doc_id": 48}\n]');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<EvalRunResponse | null>(null);
  const [runs, setRuns] = useState<EvalRunResponse[]>([]);

  const [judgeScope, setJudgeScope] = useState<'uploaded' | 'public'>('uploaded');
  const [judgeK, setJudgeK] = useState(10);
  const [judgeRaw, setJudgeRaw] = useState('[\n  {"query":"What is the main contribution?", "scope":"uploaded"}\n]');
  const [judgeRunning, setJudgeRunning] = useState(false);
  const [judgeError, setJudgeError] = useState('');
  const [judgeResult, setJudgeResult] = useState<JudgeRunResponse | null>(null);
  const [judgeRuns, setJudgeRuns] = useState<JudgeRunSummary[]>([]);

  const [calibRaw, setCalibRaw] = useState('{\n  "records": [\n    {"sentence":"The model answers from evidence","evidence":"chunk contains the claim","M":0.82,"S":0.75,"A":0.70,"label":"strong"}\n  ]\n}');
  const [calibRunning, setCalibRunning] = useState(false);
  const [calibError, setCalibError] = useState('');
  const [calibResult, setCalibResult] = useState<MsaCalibrationResponse | null>(null);
  const [calibLatest, setCalibLatest] = useState<MsaCalibrationLatest | null>(null);

  useEffect(() => {
    api.listEvalRuns(20).then((r) => setRuns(r.runs || [])).catch(() => undefined);
    api.listJudgeRuns(20).then((r) => setJudgeRuns(r.runs || [])).catch(() => undefined);
    api.getLatestCalibration().then((r) => setCalibLatest(r || null)).catch(() => undefined);
  }, []);

  const runEval = async () => {
    setError(''); setRunning(true);
    try {
      const cases = JSON.parse(rawCases) as EvalCase[];
      const res = await api.runEval({ name, scope: 'uploaded', k, cases });
      setResult(res);
      const r = await api.listEvalRuns(20);
      setRuns(r.runs || []);
    } catch (e: any) { setError(e?.message || 'Failed'); }
    finally { setRunning(false); }
  };

  const runJudge = async () => {
    setJudgeError(''); setJudgeRunning(true);
    try {
      const cases = JSON.parse(judgeRaw) as JudgeCasePayload[];
      const res = await api.runJudge({ scope: judgeScope, k: judgeK, run_judge_llm: true, cases });
      setJudgeResult(res);
      const r = await api.listJudgeRuns(20);
      setJudgeRuns(r.runs || []);
    } catch (e: any) { setJudgeError(e?.message || 'Failed'); }
    finally { setJudgeRunning(false); }
  };

  const runCalib = async () => {
    setCalibError(''); setCalibRunning(true);
    try {
      const parsed = JSON.parse(calibRaw) as any;
      const records = Array.isArray(parsed) ? parsed : (parsed?.records || null);
      if (!records?.length) throw new Error('records array required');
      const payload: MsaCalibrationPayload = { model_name: 'msa_manual', label: 'manual', ...parsed, records };
      const res = await api.calibrateConfidence(payload);
      setCalibResult(res);
      const latest = await api.getLatestCalibration();
      setCalibLatest(latest);
    } catch (e: any) { setCalibError(e?.message || 'Failed'); }
    finally { setCalibRunning(false); }
  };

  return (
    <div className="eval-page">
      <div className="eval-topbar">
        <h1>Evaluation Studio</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack}>← Back to Chat</button>
      </div>

      <div className="eval-grid">
        <div className="eval-card">
          <h3>LLM Judge Evaluation</h3>
          <label>Scope</label>
          <select value={judgeScope} onChange={(e) => setJudgeScope(e.target.value as 'uploaded' | 'public')}>
            <option value="uploaded">Uploaded</option>
            <option value="public">Public</option>
          </select>
          <label>Top K</label>
          <input type="number" value={judgeK} onChange={(e) => setJudgeK(Number(e.target.value) || 10)} />
          <label>Test cases JSON</label>
          <textarea rows={8} value={judgeRaw} onChange={(e) => setJudgeRaw(e.target.value)} />
          <button className="btn btn-primary btn-sm" onClick={runJudge} disabled={judgeRunning}>
            {judgeRunning ? 'Running…' : 'Run judge eval'}
          </button>
          {judgeError && <div className="alert">{judgeError}</div>}
          {judgeResult && (
            <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 4, lineHeight: 1.8 }}>
              Mean score: <strong style={{ color: 'var(--text)' }}>{Math.round((judgeResult.metrics.mean_overall_score || 0) * 100)}%</strong>
              {' · '}Unsupported: <strong style={{ color: 'var(--text)' }}>{judgeResult.metrics.unsupported_total || 0}</strong>
            </div>
          )}
        </div>

        <div className="eval-card">
          <h3>M/S/A Calibration</h3>
          <label>Calibration records JSON</label>
          <textarea rows={12} value={calibRaw} onChange={(e) => setCalibRaw(e.target.value)} />
          <button className="btn btn-primary btn-sm" onClick={runCalib} disabled={calibRunning}>
            {calibRunning ? 'Calibrating…' : 'Fit MSA calibration'}
          </button>
          {calibError && <div className="alert">{calibError}</div>}
          {calibResult && (
            <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.8 }}>
              <div>{calibResult.model_name} · {calibResult.records_used} records used</div>
              <div>Brier: {calibResult.metrics.brier} · Accuracy: {calibResult.metrics.accuracy}</div>
            </div>
          )}
          {calibLatest && (
            <div style={{ fontSize: 11, color: 'var(--text-3)', borderTop: '1px solid var(--border)', paddingTop: 8 }}>
              Latest: {calibLatest.model_name} ({calibLatest.label}) · {calibLatest.dataset_size} records
            </div>
          )}
        </div>

        <div className="eval-card">
          <h3>Retrieval Evaluation</h3>
          <label>Run name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} />
          <label>Top K</label>
          <input type="number" value={k} onChange={(e) => setK(Number(e.target.value) || 10)} />
          <label>Test cases JSON</label>
          <textarea rows={8} value={rawCases} onChange={(e) => setRawCases(e.target.value)} />
          <button className="btn btn-primary btn-sm" onClick={runEval} disabled={running}>
            {running ? 'Running…' : 'Run evaluation'}
          </button>
          {error && <div className="alert">{error}</div>}
        </div>

        <div className="eval-card">
          <h3>Latest result</h3>
          {!result ? (
            <p style={{ color: 'var(--text-3)', fontSize: 13 }}>No run yet.</p>
          ) : (
            <>
              <table className="eval-table">
                <thead>
                  <tr><th>Metric</th><th>Retrieval</th><th>+Rerank</th></tr>
                </thead>
                <tbody>
                  {(['1', '3', '5', '10'] as const).map((n) => (
                    <tr key={n}>
                      <td>Recall@{n}</td>
                      <td>{result.metrics_retrieval_only.recall_at[n]?.toFixed(3) ?? '–'}</td>
                      <td>{result.metrics_retrieval_rerank.recall_at[n]?.toFixed(3) ?? '–'}</td>
                    </tr>
                  ))}
                  <tr>
                    <td>MRR</td>
                    <td>{result.metrics_retrieval_only.mrr.toFixed(3)}</td>
                    <td>{result.metrics_retrieval_rerank.mrr.toFixed(3)}</td>
                  </tr>
                </tbody>
              </table>
              <div className="lat-bars">
                {([
                  ['Retrieve', result.latency_breakdown.retrieve_ms_avg],
                  ['Rerank', result.latency_breakdown.rerank_ms_avg],
                  ['Generate', result.latency_breakdown.generate_ms_avg],
                ] as [string, number][]).map(([label, val]) => (
                  <div key={label} className="lat-row">
                    <span className="lat-label">{label} {Math.round(val)} ms</span>
                    <div className="lat-bar">
                      <div className="lat-fill" style={{ width: `${Math.min(100, val / 3000 * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="eval-card" style={{ marginBottom: 16 }}>
        <h3>Stored eval runs</h3>
        <div className="run-list">
          {runs.length === 0
            ? <div style={{ color: 'var(--text-3)', fontSize: 13 }}>No runs yet.</div>
            : runs.map((r) => (
              <div key={`${r.run_id}-${r.created_at}`} className="run-item">
                <strong>{r.name}</strong>
                <span>{r.created_at}</span>
                <span>{r.case_count} cases</span>
                <span>R@5 {r.metrics_retrieval_rerank?.recall_at?.['5'] ?? '–'}</span>
              </div>
            ))}
        </div>
      </div>

      <div className="eval-card">
        <h3>Judge runs</h3>
        <div className="run-list">
          {judgeRuns.length === 0
            ? <div style={{ color: 'var(--text-3)', fontSize: 13 }}>No judge runs yet.</div>
            : judgeRuns.map((r, idx) => (
              <div key={`${r.id}-${idx}`} className="run-item">
                <strong>{r.scope || judgeScope}</strong>
                <span>Run {r.id}</span>
                <span>{r.query_count || 0} queries</span>
                <span>Mean {Math.round(((r.metrics?.mean_overall_score) || 0) * 100)}%</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

// ── Studio page ───────────────────────────────────────────────────────────────
const STORAGE_KEY = 'scholarrag_studio_v3';
const MEMORY_PREF_KEY = 'scholarrag_memory_pref_v1';
const DEFAULT_SESSION_TITLE = 'New chat';
const EMPTY_EVIDENCE: EvidenceState = { citations: [], trace: [] };

function createSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function shortTitle(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  if (!compact) return DEFAULT_SESSION_TITLE;
  return compact.length > 44 ? `${compact.slice(0, 44).trimEnd()}…` : compact;
}

function deriveSessionTitle(messages: UiMessage[]): string {
  const firstUser = messages.find((m) => m.role === 'you' && m.text.trim());
  return firstUser ? shortTitle(firstUser.text) : DEFAULT_SESSION_TITLE;
}

function createStudioSession(overrides: Partial<StudioSession> = {}): StudioSession {
  const now = Date.now();
  const messages = overrides.messages || [];
  return {
    id: overrides.id || createSessionId(),
    title: overrides.title || deriveSessionTitle(messages),
    messages,
    selectedDocs: overrides.selectedDocs || [],
    activeEvidence: overrides.activeEvidence || EMPTY_EVIDENCE,
    activeEvidenceMsgIdx: overrides.activeEvidenceMsgIdx ?? -1,
    allowGeneralBackground: overrides.allowGeneralBackground ?? false,
    createdAt: overrides.createdAt ?? now,
    updatedAt: overrides.updatedAt ?? now,
  };
}

const INITIAL_SESSION = createStudioSession();

function resolveUpdater<T>(next: T | ((prev: T) => T), prev: T): T {
  return typeof next === 'function' ? (next as (prev: T) => T)(prev) : next;
}

function StudioPage({ onNavigateEval }: { onNavigateEval: () => void }) {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [sessions, setSessions] = useState<StudioSession[]>([INITIAL_SESSION]);
  const [activeSessionId, setActiveSessionId] = useState<string>(INITIAL_SESSION.id);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [pendingDelete, setPendingDelete] = useState<DocumentRow | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ text: string; state: 'idle' | 'uploading' | 'done' | 'err' }>({ text: '', state: 'idle' });
  const [uploadPct, setUploadPct] = useState(0);
  const [preserveMemory, setPreserveMemory] = useState<boolean>(() => {
    try {
      return localStorage.getItem(MEMORY_PREF_KEY) !== 'off';
    } catch {
      return true;
    }
  });

  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) || sessions[0] || INITIAL_SESSION,
    [sessions, activeSessionId],
  );
  const messages = activeSession.messages;
  const selectedDocs = activeSession.selectedDocs;
  const allowGeneralBackground = activeSession.allowGeneralBackground;
  const activeEvidence = activeSession.activeEvidence;
  const activeEvidenceMsgIdx = activeSession.activeEvidenceMsgIdx;

  const updateActiveSession = (updater: (session: StudioSession) => StudioSession) => {
    setSessions((prev) =>
      prev.map((session) =>
        session.id === activeSessionId
          ? { ...updater(session), updatedAt: Date.now() }
          : session,
      ),
    );
  };

  const setMessages = (next: UiMessage[] | ((prev: UiMessage[]) => UiMessage[])) => {
    updateActiveSession((session) => {
      const messages = resolveUpdater(next, session.messages);
      return {
        ...session,
        messages,
        title: deriveSessionTitle(messages),
      };
    });
  };

  const setSelectedDocs = (next: number[] | ((prev: number[]) => number[])) => {
    updateActiveSession((session) => ({
      ...session,
      selectedDocs: resolveUpdater(next, session.selectedDocs),
    }));
  };

  const setAllowGeneralBackground = (next: boolean | ((prev: boolean) => boolean)) => {
    updateActiveSession((session) => ({
      ...session,
      allowGeneralBackground: resolveUpdater(next, session.allowGeneralBackground),
    }));
  };

  const setActiveEvidence = (next: EvidenceState | ((prev: EvidenceState) => EvidenceState)) => {
    updateActiveSession((session) => ({
      ...session,
      activeEvidence: resolveUpdater(next, session.activeEvidence),
    }));
  };

  const setActiveEvidenceMsgIdx = (next: number | ((prev: number) => number)) => {
    updateActiveSession((session) => ({
      ...session,
      activeEvidenceMsgIdx: resolveUpdater(next, session.activeEvidenceMsgIdx),
    }));
  };

  // Restore session
  useEffect(() => {
    if (!preserveMemory) return;
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (Array.isArray(s.sessions) && s.sessions.length) {
        const restored = s.sessions.map((session: any) => createStudioSession({
          id: session.id,
          title: session.title,
          messages: Array.isArray(session.messages) ? session.messages : [],
          selectedDocs: Array.isArray(session.selectedDocs) ? session.selectedDocs : [],
          activeEvidence: session.activeEvidence || EMPTY_EVIDENCE,
          activeEvidenceMsgIdx: typeof session.activeEvidenceMsgIdx === 'number' ? session.activeEvidenceMsgIdx : -1,
          allowGeneralBackground: Boolean(session.allowGeneralBackground),
          createdAt: session.createdAt,
          updatedAt: session.updatedAt,
        }));
        setSessions(restored);
        setActiveSessionId(
          restored.some((session: StudioSession) => session.id === s.activeSessionId)
            ? s.activeSessionId
            : restored[0].id,
        );
        return;
      }
      if (Array.isArray(s.messages)) {
        const migrated = createStudioSession({
          messages: s.messages,
          selectedDocs: Array.isArray(s.selectedDocs) ? s.selectedDocs : [],
          activeEvidence: s.activeEvidence || EMPTY_EVIDENCE,
          activeEvidenceMsgIdx: typeof s.activeEvidenceMsgIdx === 'number' ? s.activeEvidenceMsgIdx : -1,
        });
        setSessions([migrated]);
        setActiveSessionId(migrated.id);
      }
    } catch {}
  }, [preserveMemory]);

  // Persist session
  useEffect(() => {
    if (!preserveMemory) return;
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ sessions, activeSessionId }));
    } catch {}
  }, [sessions, activeSessionId, preserveMemory]);

  useEffect(() => {
    try {
      localStorage.setItem(MEMORY_PREF_KEY, preserveMemory ? 'on' : 'off');
      if (!preserveMemory) {
        sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch {}
  }, [preserveMemory]);

  const refreshDocs = async () => {
    try {
      const res = await api.listDocs();
      const list = res.documents || [];
      setDocs(list);
      setSessions((prev) =>
        prev.map((session) => ({
          ...session,
          selectedDocs: session.selectedDocs.filter((id) => list.some((d) => d.id === id)),
        })),
      );
      setError('');
    } catch (e: any) {
      setError(e?.message || `Backend unreachable at ${API_BASE}`);
    }
  };

  useEffect(() => { refreshDocs(); }, []);

  useEffect(() => {
    if (!docs.some((d) => d.status === 'processing')) return;
    const id = setInterval(refreshDocs, 2500);
    return () => clearInterval(id);
  }, [docs]);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  const dedupedDocs = useMemo(() => {
    const seen = new Set<string>();
    return docs.filter((d) => {
      const k = d.title.toLowerCase();
      if (seen.has(k)) return false;
      seen.add(k); return true;
    });
  }, [docs]);

  const selectedRows = useMemo(() => dedupedDocs.filter((d) => selectedDocs.includes(d.id)), [dedupedDocs, selectedDocs]);
  const activeDoc = selectedRows[0] || null;
  const hasMulti = selectedRows.length > 1;
  const processedCount = dedupedDocs.filter((d) => d.status === 'ready').length;
  const processingCount = dedupedDocs.filter((d) => d.status === 'processing').length;
  const orderedSessions = useMemo(
    () => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions],
  );
  const selectedPreview = selectedRows.slice(0, 3);

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploadStatus({ text: `Uploading ${files.length} file(s)…`, state: 'uploading' });
    setUploadPct(0);
    try {
      for (let i = 0; i < files.length; i++) {
        setUploadStatus({ text: `Uploading ${i + 1}/${files.length}: ${files[i].name}`, state: 'uploading' });
        setUploadPct(Math.round(((i + 0.5) / files.length) * 100));
        await api.uploadFile(files[i]);
      }
      setUploadPct(100);
      setUploadStatus({
        text: `${files.length === 1 ? files[0].name : `${files.length} files`} uploaded`,
        state: 'done',
      });
      refreshDocs();
      setTimeout(() => setUploadStatus({ text: '', state: 'idle' }), 3000);
    } catch (e: any) {
      setUploadStatus({ text: e?.message || 'Upload failed', state: 'err' });
    }
  };

  const streamMessage = (
    fullText: string,
    meta: Pick<UiMessage, 'citations' | 'confidence' | 'why_answer' | 'latency_breakdown_ms' | 'needs_clarification' | 'clarification' | 'answer_scope' | 'unsupported_claims' | 'query_ref' | 'faithfulness'>,
  ) => {
    const text = fullText || 'No response received. Check backend/OpenAI key.';
    setMessages((prev) => [...prev, { role: 'assistant', text, streaming: false, ...meta }]);
    setActiveEvidence({ citations: meta.citations || [], trace: meta.why_answer?.top_chunks || [] });
  };

  const startNewChat = () => {
    const session = createStudioSession({
      selectedDocs,
      allowGeneralBackground,
    });
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setInput('');
    setError('');
    setLoading(false);
  };

  const ask = async (text: string, skipEnrichment = false, sense?: string) => {
    const q = text.trim();
    if (!q) return;
    setError('');
    setLoading(true);
    setMessages((prev) => [...prev, { role: 'you', text: q }]);
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    try {
      const hasUploads = Boolean(selectedDocs.length || docs.length);
      if (isGreetingQuery(q)) {
        streamMessage(
          buildAssistantIntroReply('greeting', hasUploads),
          { citations: [] },
        );
        setLoading(false);
        return;
      }
      if (selectedDocs.length === 0 && isAssistantSetupQuery(q)) {
        streamMessage(
          buildAssistantIntroReply('setup', hasUploads),
          { citations: [] },
        );
        setLoading(false);
        return;
      }
      if (hasUploads && selectedDocs.length === 0 && isExplicitDocumentQuery(q)) {
        streamMessage(
          'Select one or more documents first, or ask a workspace-wide question without referring to "this document".',
          { citations: [] },
        );
        setLoading(false);
        return;
      }
      if (!allowGeneralBackground && isLiteratureQuery(q)) {
        streamMessage(
          selectedDocs.length > 0
            ? 'Docs only mode is active. Ask about the selected document(s), or switch to Public research if you want papers, citations, or literature search.'
            : 'Docs only mode is active. Select one or more documents for grounded answers, or switch to Public research if you want papers, citations, or literature search.',
          { citations: [] },
        );
        setLoading(false);
        return;
      }
      const forcePublicWithoutSelection = allowGeneralBackground && hasUploads && selectedDocs.length === 0;
      const scope: 'uploaded' | 'public' =
        forcePublicWithoutSelection ? 'public' : (hasUploads ? 'uploaded' : 'public');
      const query = allowGeneralBackground && !skipEnrichment && scope === 'public' ? enrichQuery(q, messages) : q;
      const res = await api.askAssistant({
        query, scope, sense,
        doc_id: scope === 'uploaded' && selectedDocs.length === 1 ? selectedDocs[0] : undefined,
        doc_ids: scope === 'uploaded' && selectedDocs.length > 1 ? selectedDocs : undefined,
        k: 8,
        allow_general_background: allowGeneralBackground,
      });
      streamMessage((res.answer || res.clarification?.question || '').trim(), {
        citations: res.citations || [],
        confidence: res.confidence,
        why_answer: res.why_answer,
        latency_breakdown_ms: res.latency_breakdown_ms,
        needs_clarification: res.needs_clarification,
        clarification: res.clarification,
        answer_scope: res.answer_scope,
        unsupported_claims: res.unsupported_claims,
        faithfulness: res.faithfulness,
        query_ref: query,
      });
    } catch (e: any) {
      setError(e?.message || 'Request failed');
      streamMessage(e?.message || `Backend unreachable at ${API_BASE}`, { citations: [] });
    } finally {
      setLoading(false);
    }
  };

  const toggleDoc = (id: number) =>
    setSelectedDocs((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);

  const forgetMemoryNow = () => {
    updateActiveSession((session) => ({
      ...session,
      title: DEFAULT_SESSION_TITLE,
      messages: [],
      activeEvidence: EMPTY_EVIDENCE,
      activeEvidenceMsgIdx: -1,
    }));
    setInput('');
    setError('');
    setLoading(false);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {}
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await api.deleteDoc(pendingDelete.id);
      setSelectedDocs((prev) => prev.filter((id) => id !== pendingDelete.id));
      setActiveEvidence({ citations: [], trace: [] });
    } catch {}
    setPendingDelete(null);
    refreshDocs();
  };

  const quickAsk = (prompt: string) => ask(prompt, true);
  const activeSessionTitle = activeSession.title !== DEFAULT_SESSION_TITLE
    ? activeSession.title
    : 'Chat with your research';

  const EMPTY_CARDS = activeDoc || hasMulti
    ? [
        {
          eyebrow: 'Summary',
          title: hasMulti ? 'Summarize selected documents' : 'Summarize selected document',
          text: 'Get the main findings, structure, and takeaways in a compact form.',
          prompt: hasMulti
            ? 'Summarize the selected uploaded documents. Organize by document and provide combined takeaways.'
            : 'Summarize the selected uploaded document.',
        },
        {
          eyebrow: 'Key points',
          title: 'Extract key concepts',
          text: 'Highlight the most important skills, topics, and claims.',
          prompt: hasMulti
            ? 'Extract key skills, topics, and main points from each selected document.'
            : 'What are the key skills or topics in this document?',
        },
        {
          eyebrow: 'Evidence',
          title: 'Inspect supporting evidence',
          text: 'Surface the chunks and pages that best support the main claims.',
          prompt: hasMulti
            ? 'What evidence best supports the main claims across these documents?'
            : 'What evidence best supports the main claims in this document?',
        },
        {
          eyebrow: 'Analysis',
          title: 'Find gaps and risks',
          text: 'Point out ambiguities, weak support, or missing details in the material.',
          prompt: hasMulti
            ? 'Identify weakly supported claims, inconsistencies, or missing details across these documents.'
            : 'Identify weakly supported claims, inconsistencies, or missing details in this document.',
        },
      ]
    : [
        {
          eyebrow: 'Explain',
          title: 'Understand a concept',
          text: 'Get a clear explanation grounded in relevant research sources.',
          prompt: 'Tell me about retrieval-augmented generation.',
        },
        {
          eyebrow: 'Discover',
          title: 'Find relevant papers',
          text: 'Pull strong papers, surveys, or references on a research topic.',
          prompt: 'Show me papers on transformer interpretability.',
        },
        {
          eyebrow: 'Compare',
          title: 'Compare methods',
          text: 'Synthesize tradeoffs, strengths, and limitations across approaches.',
          prompt: 'Compare LSTMs and GRUs in the literature.',
        },
        {
          eyebrow: 'Ground',
          title: 'Analyze an uploaded document',
          text: 'Upload a PDF, select it, then ask for summaries or evidence-backed answers.',
          prompt: 'What are the key findings of the selected document?',
        },
      ];

  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">SR</div>
          <div className="brand-text">
            <div className="brand-name">ScholarRAG</div>
            <div className="brand-sub">Research assistant</div>
          </div>
        </div>

        <div className="sidebar-body">
          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>Workspace</span>
              <span>
                {processedCount} ready{processingCount ? ` · ${processingCount} processing` : ''}
              </span>
            </div>
            <div className="workspace-card">
              <div className="workspace-card-head">
                <div className="workspace-card-title">Research workspace</div>
                <span className="workspace-card-chip">{orderedSessions.length} chats</span>
              </div>
              <div className="workspace-card-copy">
                Keep documents on the left, work in sessions, and inspect grounding evidence on the right.
              </div>
              <div className="workspace-card-metrics">
                <div className="workspace-metric">
                  <span className="workspace-metric-value">{processedCount}</span>
                  <span className="workspace-metric-label">Ready docs</span>
                </div>
                <div className="workspace-metric">
                  <span className="workspace-metric-value">{orderedSessions.length}</span>
                  <span className="workspace-metric-label">Chat sessions</span>
                </div>
              </div>
            </div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>Chats</span>
              <button className="sidebar-inline-action" onClick={startNewChat}>+ New</button>
            </div>
            <div className="chat-session-list">
              {orderedSessions.map((session) => {
                const isActive = session.id === activeSessionId;
                const turnCount = session.messages.filter((m) => m.role === 'you').length;
                return (
                  <button
                    key={session.id}
                    className={`chat-session-item${isActive ? ' active' : ''}`}
                    onClick={() => {
                      setActiveSessionId(session.id);
                      setInput('');
                      setError('');
                      setLoading(false);
                    }}
                  >
                    <div className="chat-session-title-row">
                      <span className="chat-session-title">{session.title}</span>
                      {isActive && <span className="chat-session-pill">Open</span>}
                    </div>
                    <div className="chat-session-meta">
                      {turnCount ? `${turnCount} prompts` : 'Empty chat'}
                      {session.selectedDocs.length ? ` · ${session.selectedDocs.length} doc${session.selectedDocs.length > 1 ? 's' : ''}` : ''}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>Session</span>
              <span>{preserveMemory ? 'saved' : 'ephemeral'}</span>
            </div>
            <div className="session-card">
              <div className="session-card-copy">
                Keep the current chat across refreshes, or turn memory off for a temporary scratch session.
              </div>
              <div className="session-actions">
                <button
                  className={`session-toggle${preserveMemory ? ' active' : ''}`}
                  onClick={() => setPreserveMemory((prev) => !prev)}
                  aria-pressed={preserveMemory}
                  type="button"
                >
                  <span className="session-toggle-track">
                    <span className="session-toggle-thumb" />
                  </span>
                  <span>{preserveMemory ? 'Memory on' : 'Memory off'}</span>
                </button>
                <button
                  className="session-clear"
                  onClick={forgetMemoryNow}
                  disabled={messages.length === 0}
                  type="button"
                >
                  Clear chat
                </button>
              </div>
            </div>
          </div>

          {/* Upload zone */}
          <div className="sidebar-section">
            <div className="sidebar-label"><span>Upload</span><span>PDF / TXT</span></div>
            <div
              className={`upload-zone${dragActive ? ' drag-active' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); }}
              onClick={() => document.getElementById('sr-upload')?.click()}
            >
              <div className="upload-icon">📄</div>
              <div className="upload-text">Drop files or <strong>browse</strong></div>
              <input
                id="sr-upload"
                type="file"
                accept=".pdf,.txt,.md"
                multiple
                style={{ display: 'none' }}
                onChange={(e) => handleFiles(e.target.files)}
              />
            </div>
            {uploadStatus.state !== 'idle' && (
              <div className={`upload-status ${uploadStatus.state}`}>
                {uploadStatus.state === 'uploading' && (
                  <span style={{
                    display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                    border: '1.5px solid var(--cyan)', borderTopColor: 'transparent',
                    animation: 'spin 0.7s linear infinite',
                  }} />
                )}
                {uploadStatus.state === 'done' && '✓'}
                {uploadStatus.state === 'err' && '✕'}
                {uploadStatus.text}
              </div>
            )}
            {uploadStatus.state === 'uploading' && (
              <div className="upload-bar">
                <div className="upload-bar-fill" style={{ width: `${uploadPct}%` }} />
              </div>
            )}
          </div>

          {/* Documents */}
          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>Documents</span>
              {selectedDocs.length > 0 && (
                <span style={{ color: 'var(--indigo)' }}>{selectedDocs.length} selected</span>
              )}
            </div>
            {dedupedDocs.length === 0 ? (
              <div className="doc-empty">
                <div className="doc-empty-icon">📂</div>
                <div>No documents yet.<br />Upload a PDF to begin.</div>
              </div>
            ) : (
              <div className="doc-list">
                {dedupedDocs.map((d) => {
                  const sel = selectedDocs.includes(d.id);
                  const badgeCls = d.status === 'ready' ? 'ready' : d.status === 'error' ? 'error' : 'processing';
                  const badgeText = d.status === 'ready' ? 'Ready' : d.status === 'error' ? 'Error' : 'Processing…';
                  return (
                    <div
                      key={d.id}
                      className={`doc-item${sel ? ' selected' : ''}`}
                      onClick={() => toggleDoc(d.id)}
                    >
                      <div className="doc-check">{sel ? '✓' : ''}</div>
                      <div className="doc-icon">📝</div>
                      <div className="doc-info">
                        <div className="doc-name" title={d.title}>{d.title}</div>
                        <div className="doc-status-row">
                          <span className={`doc-badge ${badgeCls}`}>{badgeText}</span>
                        </div>
                      </div>
                      <button
                        className="doc-del"
                        title="Delete document"
                        onClick={(e) => { e.stopPropagation(); setPendingDelete(d); }}
                      >✕</button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <button className="sidebar-footer-btn" onClick={onNavigateEval}>
            <span className="sidebar-footer-btn-icon">⚗</span>
            Evaluation Studio
          </button>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="main-content">
        <div className="topbar">
          <div className="topbar-left">
            <div className="topbar-kicker">Research workspace</div>
            <div className="topbar-title">
              {hasMulti
                ? `${selectedRows.length} documents selected`
                : (activeDoc ? activeDoc.title : activeSessionTitle)}
            </div>
            <div className="topbar-sub">
              {hasMulti
                ? 'Cross-document analysis mode'
                : (activeDoc ? 'Grounded research assistant' : 'Session-based research assistant')}
            </div>
          </div>
          <div className="topbar-right">
            <div className="seg-toggle">
              <button
                className={!allowGeneralBackground ? 'active' : ''}
                onClick={() => setAllowGeneralBackground(false)}
              >Docs only</button>
              <button
                className={allowGeneralBackground ? 'active' : ''}
                onClick={() => setAllowGeneralBackground(true)}
              >Public research</button>
            </div>
          </div>
        </div>

        <div className="context-strip">
          <div className="context-strip-label">Active context</div>
          <div className="context-chip-row">
            <span className={`context-chip mode ${allowGeneralBackground ? 'public' : 'docs'}`}>
              {allowGeneralBackground ? 'Public research enabled' : 'Docs only mode'}
            </span>
            {selectedRows.length === 0 && !allowGeneralBackground && (
              <span className="context-chip muted">No documents selected</span>
            )}
            {selectedPreview.map((doc) => (
              <button key={doc.id} className="context-chip" onClick={() => toggleDoc(doc.id)}>
                {doc.title}
                <span className="context-chip-close">×</span>
              </button>
            ))}
            {selectedRows.length > selectedPreview.length && (
              <span className="context-chip muted">+{selectedRows.length - selectedPreview.length} more</span>
            )}
          </div>
        </div>

        {/* Chat stream */}
        <div className="chat-stream">
          <div className="chat-inner">
            {messages.length === 0 && !loading && (
              <div className="chat-empty">
                <div className="chat-empty-hero">
                  <div className="chat-empty-kicker">Grounded AI research assistant</div>
                  <div className="chat-empty-logo">SR</div>
                  <div className="chat-empty-title">Ask better research questions</div>
                  <div className="chat-empty-sub">
                    {activeDoc
                      ? `You’re currently focused on "${activeDoc.title}". Ask for summaries, evidence, gaps, or key findings.`
                      : 'Search the literature, explain concepts, compare methods, or upload documents for grounded analysis.'}
                  </div>
                  <div className="hero-hints">
                    <div className="hero-hint">
                      <span className="hero-hint-label">Explain</span>
                      <strong>Tell me about RNNs</strong>
                    </div>
                    <div className="hero-hint">
                      <span className="hero-hint-label">Discover</span>
                      <strong>Show me papers on multimodal attention</strong>
                    </div>
                    <div className="hero-hint">
                      <span className="hero-hint-label">Compare</span>
                      <strong>Compare LSTMs and GRUs in the literature</strong>
                    </div>
                  </div>
                </div>
                <div className="empty-grid">
                  {EMPTY_CARDS.map((card) => (
                    <button key={card.title} className="empty-card" onClick={() => quickAsk(card.prompt)}>
                      <div className="empty-card-eyebrow">{card.eyebrow}</div>
                      <div className="empty-card-title">{card.title}</div>
                      <div className="empty-card-text">{card.text}</div>
                      <div className="empty-card-cta">Run this prompt →</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {error && <div className="alert" style={{ marginBottom: 12 }}>{error}</div>}

            {messages.map((m, i) => (
              <div key={i} className="msg-group">
                <div className={`bubble-row ${m.role}`}>
                  {m.role === 'assistant' && <div className="msg-avatar assistant">SR</div>}
                  <div
                    className={`msg-bubble${m.role === 'assistant' && activeEvidenceMsgIdx === i ? ' active-evidence' : ''}`}
                    onClick={() => {
                      if (m.role === 'assistant') {
                        setActiveEvidence({ citations: m.citations || [], trace: m.why_answer?.top_chunks || [] });
                        setActiveEvidenceMsgIdx(i);
                      }
                    }}
                  >
                    {m.role === 'assistant' ? renderMarkdown(m.text) : <div className="md"><p>{m.text}</p></div>}
                    {m.streaming && <span className="stream-cursor" />}
                  </div>
                </div>

                {m.role === 'assistant' && !m.streaming && (
                  <>
                    <div className="msg-meta">
                      <div className="assistant-card-head">
                        <span className="assistant-card-label">{formatAnswerScope(m.answer_scope) || 'Answer'}</span>
                        {m.citations?.length ? (
                          <span className="assistant-card-count">{m.citations.length} source{m.citations.length > 1 ? 's' : ''}</span>
                        ) : null}
                      </div>
                      <ConfBadge confidence={m.confidence} />
                      {m.faithfulness && (
                        <span className="meta-chip">
                          Faithfulness: {Math.round((m.faithfulness.overall_score || 0) * 100)}%
                        </span>
                      )}
                    </div>
                    {m.needs_clarification && m.clarification?.options?.length ? (
                      <div className="clarify-box">
                        <div className="clarify-q">{m.clarification.question}</div>
                        <div className="clarify-opts">
                          {m.clarification.options.map((opt) => (
                            <button
                              key={opt}
                              className="clarify-opt"
                              onClick={() => ask(m.query_ref || '', true, opt)}
                            >
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

            {loading && <TypingIndicator />}
            <div ref={chatEndRef} />
          </div>

          {/* Quick chips */}
          {messages.length > 0 && !loading && activeDoc && (
            <div className="quick-actions">
              {[
                { label: '📋 Summarize', prompt: hasMulti ? 'Summarize the selected uploaded documents.' : 'Summarize the selected uploaded document.' },
                { label: '🔑 Key points', prompt: 'What are the key points and main contributions?' },
                { label: '🔍 Evidence', prompt: 'What evidence best supports the main claims?' },
              ].map((c) => (
                <button key={c.label} className="quick-chip" onClick={() => quickAsk(c.prompt)}>
                  {c.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="composer-wrap">
          <div className="composer-inner">
            <div className="composer-box">
              <textarea
                ref={textareaRef}
                className="composer-input"
                rows={1}
                value={input}
                placeholder={
                  allowGeneralBackground
                    ? 'Search public research or ask about your documents…'
                    : selectedDocs.length > 0
                      ? 'Ask about your selected documents…'
                      : 'Select a document, or switch to Public research for general questions…'
                }
                disabled={loading}
                onChange={(e) => setInput(e.target.value)}
                onInput={(e) => {
                  const el = e.currentTarget;
                  el.style.height = 'auto';
                  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    ask(input);
                  }
                }}
              />
              <button
                className={`composer-send${loading ? ' thinking' : ''}`}
                disabled={loading || !input.trim()}
                onClick={() => ask(input)}
                title="Send (Enter)"
              >
                {!loading && '↑'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Evidence panel ── */}
      <EvidencePanel
        citations={activeEvidence.citations}
        traceChunks={activeEvidence.trace}
        loading={loading}
        allowGeneralBackground={allowGeneralBackground}
      />

      <DeleteModal
        doc={pendingDelete}
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
      />
    </div>
  );
}

// ── Root with routing ─────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState<Page>(() =>
    window.location.pathname.startsWith('/eval') ? 'eval' : 'studio',
  );

  useEffect(() => {
    const handler = () =>
      setPage(window.location.pathname.startsWith('/eval') ? 'eval' : 'studio');
    window.addEventListener('popstate', handler);
    return () => window.removeEventListener('popstate', handler);
  }, []);

  const goEval = () => { window.history.pushState({}, '', '/eval'); setPage('eval'); };
  const goStudio = () => { window.history.pushState({}, '', '/'); setPage('studio'); };

  if (page === 'eval') return <EvalPage onBack={goStudio} />;
  return <StudioPage onNavigateEval={goEval} />;
}
