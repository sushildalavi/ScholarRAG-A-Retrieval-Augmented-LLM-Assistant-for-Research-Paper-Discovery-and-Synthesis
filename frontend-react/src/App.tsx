import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api/client';
import { DocumentRow, Citation } from './api/types';
import { SearchBar } from './components/SearchBar';
import { UploadPanel } from './components/UploadPanel';
import { LatestDocumentsList } from './components/LatestDocumentsList';
import { ResultsList } from './components/ResultsList';
// Center pane now renders chat-style bubbles instead of a static answer panel.

export default function App() {
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<number | null>(null);

  // Doc-scoped QA
  const [questionDoc, setQuestionDoc] = useState('');
  const [answerDoc, setAnswerDoc] = useState('');
  const [displayAnswerDoc, setDisplayAnswerDoc] = useState('');
  const [citationsDoc, setCitationsDoc] = useState<Citation[]>([]);
  const [resultsDoc, setResultsDoc] = useState<any[]>([]);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [streamDoc, setStreamDoc] = useState(false);
  const [errorDoc, setErrorDoc] = useState('');
  const [docMessages, setDocMessages] = useState<{ role: 'you' | 'assistant'; text: string }[]>([]);
  const askDocRef = useRef<HTMLInputElement | null>(null);

  // Metrics
  const [metrics, setMetrics] = useState<any | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const metricsIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Public + optional uploads
  const [questionPub, setQuestionPub] = useState('');
  const [answerPub, setAnswerPub] = useState('');
  const [displayAnswerPub, setDisplayAnswerPub] = useState('');
  const [citationsPub, setCitationsPub] = useState<Citation[]>([]);
  const [resultsPub, setResultsPub] = useState<any[]>([]);
  const [loadingPub, setLoadingPub] = useState(false);
  const [streamPub, setStreamPub] = useState(false);
  const [errorPub, setErrorPub] = useState('');
  const includeUploadsInPub = true; // always include uploads for assistant
  const [chatMessages, setChatMessages] = useState<{ role: 'you' | 'assistant'; text: string }[]>([]);
  const askPubRef = useRef<HTMLInputElement | null>(null);
  const uploadRightRef = useRef<HTMLInputElement | null>(null);
  const uploadImageRef = useRef<HTMLInputElement | null>(null);
  const [uploadingRight, setUploadingRight] = useState(false);
  const [showUploadMenu, setShowUploadMenu] = useState(false);

  const refreshDocs = async () => {
    try {
      const res = await api.listDocs();
      const list = res.documents || [];
      setDocs(list);
      if (!selectedDoc && list.length) setSelectedDoc(list[0].id);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    refreshDocs();
  }, []);

  useEffect(() => {
    const fetchMetrics = () => {
      api
        .rawGet('/metrics')
        .then((res) => {
          setMetrics(res);
          setMetricsError(null);
        })
        .catch((err) => setMetricsError(err?.message || 'Failed to load metrics'));
    };
    fetchMetrics();
    metricsIntervalRef.current = setInterval(fetchMetrics, 60_000);
    return () => {
      if (metricsIntervalRef.current) clearInterval(metricsIntervalRef.current);
    };
  }, []);

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

  const handleDocAsk = async () => {
    const text = questionDoc.trim();
    if (!text) return setErrorDoc('Please enter a question.');
    if (!selectedDoc && !docs.length) return setErrorDoc('Upload and select a document first.');
    setErrorDoc('');
    setLoadingDoc(true);
    setStreamDoc(false);
    setAnswerDoc('');
    setDisplayAnswerDoc('');
    setCitationsDoc([]);
    setDocMessages((msgs) => [...msgs, { role: 'you', text }]);
    setQuestionDoc('');
    // auto-scroll down after pushing the user message
    setTimeout(() => {
      const pane = document.querySelector('.answers-area');
      if (pane) pane.scrollTop = pane.scrollHeight;
    }, 50);
    try {
      const res = await api.askAssistant({
        query: text,
        scope: 'uploaded',
        doc_id: selectedDoc || undefined,
        k: 10,
      });
      const ans = (res.answer || '').trim();
      if (!ans) {
        setAnswerDoc('No response received. Check backend/OpenAI key.');
        setDisplayAnswerDoc('No response received. Check backend/OpenAI key.');
        setDocMessages((msgs) => [...msgs, { role: 'assistant', text: 'No response received. Check backend/OpenAI key.' }]);
      } else {
        setAnswerDoc(ans);
        setDisplayAnswerDoc(ans);
        setCitationsDoc(res.citations || []);
        setDocMessages((msgs) => [...msgs, { role: 'assistant', text: ans }]);
      }
    } catch (e) {
      setErrorDoc((e as any)?.message || 'Doc QA failed');
      setAnswerDoc('No response received. Check backend/OpenAI key.');
      setDisplayAnswerDoc('No response received. Check backend/OpenAI key.');
      setDocMessages((msgs) => [...msgs, { role: 'assistant', text: 'No response received. Check backend/OpenAI key.' }]);
    } finally {
      setLoadingDoc(false);
      setStreamDoc(false);
      setTimeout(() => {
        const pane = document.querySelector('.answers-area');
        if (pane) pane.scrollTop = pane.scrollHeight;
      }, 50);
    }
  };

  const handleDocSearch = async (q: string, k: number) => {
    const text = q.trim();
    if (!text) return setErrorDoc('Please enter a query.');
    if (!selectedDoc && !docs.length) return setErrorDoc('Upload and select a document first.');
    setErrorDoc('');
    setLoadingDoc(true);
    setStreamDoc(false);
    setAnswerDoc('');
    setDisplayAnswerDoc('');
    setCitationsDoc([]);
    setDocMessages((msgs) => [...msgs, { role: 'you', text }]);
    setQuestionDoc('');
    try {
      const res = await api.searchChunks(text, k, selectedDoc || undefined);
      setResultsDoc(res.results || []);
    } catch (e) {
      setErrorDoc((e as any)?.message || 'Search failed');
    }
    setLoadingDoc(false);
  };

  const handlePubAsk = async () => {
    const text = questionPub.trim();
    if (!text) return setErrorPub('Please enter a question.');
    setErrorPub('');
    setLoadingPub(true);
    setStreamPub(false);
    setAnswerPub('');
    setDisplayAnswerPub('');
    setCitationsPub([]);
    try {
      const res = await api.askAssistant({
        query: text,
        scope: includeUploadsInPub && (selectedDoc || docs.length) ? 'uploaded' : 'public',
        doc_id: includeUploadsInPub && selectedDoc ? selectedDoc : undefined,
        k: 10,
      });
      const ans = (res.answer || '').trim();
      if (!ans) {
        setAnswerPub('No response received. Check backend/OpenAI key.');
        setDisplayAnswerPub('No response received. Check backend/OpenAI key.');
      } else {
        setAnswerPub(ans);
        setDisplayAnswerPub(ans);
        setCitationsPub(res.citations || []);
      }
      setChatMessages((msgs) => [...msgs, { role: 'you', text }, { role: 'assistant', text: ans || 'No response received. Check backend/OpenAI key.' }]);
      setQuestionPub('');
    } catch (e) {
      setErrorPub((e as any)?.message || 'Assistant failed');
      setAnswerPub('No response received. Check backend/OpenAI key.');
      setDisplayAnswerPub('No response received. Check backend/OpenAI key.');
    } finally {
      setLoadingPub(false);
      setStreamPub(false);
    }
  };

  const handlePubSearch = async (q: string, k: number) => {
    const text = q.trim();
    if (!text) return setErrorPub('Please enter a query.');
    setErrorPub('');
    setLoadingPub(true);
    setStreamPub(false);
    setAnswerPub('');
    setDisplayAnswerPub('');
    setCitationsPub([]);
    try {
      const res = await api.askAssistant({
        query: text,
        scope: includeUploadsInPub && (selectedDoc || docs.length) ? 'uploaded' : 'public',
        doc_id: includeUploadsInPub && selectedDoc ? selectedDoc : undefined,
        k,
      });
      setResultsPub(res.citations || []);
      const ans = (res.answer || '').trim();
      const finalAns = ans || 'No response received. Check backend/OpenAI key.';
      setAnswerPub(finalAns);
      setDisplayAnswerPub(finalAns);
      setCitationsPub(res.citations || []);
      setChatMessages((msgs) => [...msgs, { role: 'you', text }, { role: 'assistant', text: finalAns }]);
      setQuestionPub('');
    } catch (e) {
      setErrorPub((e as any)?.message || 'Search failed');
    }
    setLoadingPub(false);
  };

  const handleRightUpload = async (file?: File) => {
    const f = file || uploadRightRef.current?.files?.[0];
    if (!f) return;
    setUploadingRight(true);
    setErrorPub('');
    try {
      await api.uploadFile(f);
      await refreshDocs();
    } catch (e) {
      setErrorPub((e as any)?.message || 'Upload failed');
    } finally {
      setUploadingRight(false);
      if (uploadRightRef.current) uploadRightRef.current.value = '';
    }
  };

  // Streaming animations
  useEffect(() => {
    if (!streamDoc) return;
    setDisplayAnswerDoc('');
    if (!answerDoc) {
      setStreamDoc(false);
      return;
    }
    let idx = 0;
    const step = Math.max(1, Math.floor(answerDoc.length / 80));
    const interval = setInterval(() => {
      idx += step;
      setDisplayAnswerDoc(answerDoc.slice(0, idx));
      if (idx >= answerDoc.length) {
        clearInterval(interval);
        setStreamDoc(false);
      }
    }, 20);
    return () => clearInterval(interval);
  }, [answerDoc, streamDoc]);

  useEffect(() => {
    if (!streamPub) return;
    setDisplayAnswerPub('');
    if (!answerPub) {
      setStreamPub(false);
      return;
    }
    let idx = 0;
    const step = Math.max(1, Math.floor(answerPub.length / 80));
    const interval = setInterval(() => {
      idx += step;
      setDisplayAnswerPub(answerPub.slice(0, idx));
      if (idx >= answerPub.length) {
        clearInterval(interval);
        setStreamPub(false);
      }
    }, 20);
    return () => clearInterval(interval);
  }, [answerPub, streamPub]);

  return (
    <div className="page-three">
      <div className="panel panel-left">
        <div className="brand">
          <span className="brand-icon">📚</span>
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
        <div className="section">
          <h3>My Library</h3>
        </div>
        </div>

        <div className="panel panel-center">
          <div className="center-inner">
          {docMessages.length === 0 && (
            <div className="center-hero">
              <h1>Connect your research. Get instant answers</h1>
              <p>Upload documents on the left; get doc-grounded answers here.</p>
              <div className="center-graphic" />
            </div>
          )}

          <div className="answers-area">
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
              <div key={i} className={`chat-bubble-row ${m.role}`}>
                <div className="chat-bubble">{m.text}</div>
              </div>
            ))}
            {loadingDoc && (
              <div className="empty-state tall">
                <div className="empty-icon" />
                <div>
                  <h3>Thinking…</h3>
                  <p>Retrieving from your uploaded documents.</p>
                </div>
              </div>
            )}
            <div className="chat-spacer" />
          </div>
          <div className="composer">
            <SearchBar
              value={questionDoc}
              onChange={setQuestionDoc}
              onSubmit={() => handleDocAsk()}
              disabled={loadingDoc}
              loading={loadingDoc}
              onAdvanced={() => handleDocSearch(questionDoc, 8)}
              inputRef={askDocRef}
              placeholder="Ask about my docs..."
              hideAdvanced
            />
          </div>
        </div>
      </div>

      <div className="panel panel-right">
        <div className="right-header">
          <h3>AI Assistant</h3>
        </div>
        {metricsError && <div className="alert">{metricsError}</div>}
        {metrics && (
          <div className="metrics-card">
            <div className="metrics-header">
              <span>Metrics</span>
              <small>{metrics.updated_at ? new Date(metrics.updated_at).toLocaleTimeString() : ''}</small>
            </div>
            <div className="metrics-grid">
              <div>
                <div className="metric-label">Recall@5</div>
                <div className="metric-value">{(metrics.retrieval?.recall_at_5 ?? 0).toFixed(2)}</div>
              </div>
              <div>
                <div className="metric-label">nDCG@10</div>
                <div className="metric-value">{(metrics.retrieval?.ndcg_at_10 ?? 0).toFixed(2)}</div>
              </div>
              <div>
                <div className="metric-label">MRR</div>
                <div className="metric-value">{(metrics.retrieval?.mrr ?? 0).toFixed(2)}</div>
              </div>
            </div>
            <div className="metrics-grid">
              <div>
                <div className="metric-label">p50 latency</div>
                <div className="metric-value">{metrics.latency_ms?.p50 ?? '-'} ms</div>
              </div>
              <div>
                <div className="metric-label">p95 latency</div>
                <div className="metric-value">{metrics.latency_ms?.p95 ?? '-'} ms</div>
              </div>
              <div>
                <div className="metric-label">p99 latency</div>
                <div className="metric-value">{metrics.latency_ms?.p99 ?? '-'} ms</div>
              </div>
            </div>
            <div className="metrics-grid">
              <div>
                <div className="metric-label">Avg prompt</div>
                <div className="metric-value">{metrics.token?.avg_prompt ?? '-'} tok</div>
              </div>
              <div>
                <div className="metric-label">Avg completion</div>
                <div className="metric-value">{metrics.token?.avg_completion ?? '-'} tok</div>
              </div>
            </div>
          </div>
        )}
        {errorPub && <div className="alert">{errorPub}</div>}
        <div className="assistant-body">
          {chatMessages.length === 0 && !loadingPub && (
            <div className="empty-state tall">
              <div className="empty-icon" />
              <div>
                <h3>Ask any question…</h3>
                <p>Your questions about public data and the web will appear here.</p>
              </div>
            </div>
          )}
          {chatMessages.map((m, i) => (
            <div key={i} className={`chat-bubble-row ${m.role}`}>
              <div className="chat-bubble">{m.text}</div>
            </div>
          ))}
          {loadingPub && (
            <div className="empty-state tall">
              <div className="empty-icon" />
              <div>
                <h3>Thinking…</h3>
                <p>Processing your request.</p>
              </div>
            </div>
          )}
        </div>
        <div className="assistant-bottom">
          <input
            type="file"
            accept=".pdf,.txt,.doc,.docx,.png,.jpg,.jpeg"
            ref={uploadRightRef}
            style={{ display: 'none' }}
            onChange={(e) => handleRightUpload(e.target.files?.[0])}
          />
          <input
            type="file"
            accept="image/*"
            ref={uploadImageRef}
            style={{ display: 'none' }}
            onChange={(e) => handleRightUpload(e.target.files?.[0])}
          />
          <div className="upload-inline-wrap">
            <button
              className="upload-inline"
              title="Upload"
              disabled={uploadingRight}
              onClick={() => setShowUploadMenu((s) => !s)}
            >
              +
            </button>
            {showUploadMenu && (
              <div className="upload-menu">
                <button
                  onClick={() => {
                    setShowUploadMenu(false);
                    uploadRightRef.current?.click();
                  }}
                >
                  Upload file
                </button>
                <button
                  onClick={() => {
                    setShowUploadMenu(false);
                    uploadImageRef.current?.click();
                  }}
                >
                  Upload photo
                </button>
              </div>
            )}
          </div>
          <input
            ref={askPubRef}
            className="assistant-input"
            placeholder="Ask any question..."
            value={questionPub}
            onChange={(e) => setQuestionPub(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handlePubAsk();
            }}
            disabled={loadingPub}
          />
          <button className="assistant-send" disabled={loadingPub} onClick={handlePubAsk}>
            {loadingPub ? 'Thinking…' : 'Search'}
          </button>
        </div>
      </div>
    </div>
  );
}
