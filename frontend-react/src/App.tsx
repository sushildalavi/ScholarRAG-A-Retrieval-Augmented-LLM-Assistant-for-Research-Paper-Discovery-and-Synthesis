import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api/client';
import { DocumentRow, Citation } from './api/types';
import { SearchBar } from './components/SearchBar';
import { UploadPanel } from './components/UploadPanel';
import { LatestDocumentsList } from './components/LatestDocumentsList';
import { ResultsList } from './components/ResultsList';
import { AnswerPanel } from './components/AnswerPanel';
import { AiChatWidget } from './components/AiChatWidget';

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
  const askDocRef = useRef<HTMLInputElement | null>(null);

  // Public + optional uploads
  const [questionPub, setQuestionPub] = useState('');
  const [answerPub, setAnswerPub] = useState('');
  const [displayAnswerPub, setDisplayAnswerPub] = useState('');
  const [citationsPub, setCitationsPub] = useState<Citation[]>([]);
  const [resultsPub, setResultsPub] = useState<any[]>([]);
  const [loadingPub, setLoadingPub] = useState(false);
  const [streamPub, setStreamPub] = useState(false);
  const [errorPub, setErrorPub] = useState('');
  const [includeUploadsInPub, setIncludeUploadsInPub] = useState(true);
  const askPubRef = useRef<HTMLInputElement | null>(null);

  const [chatOpen, setChatOpen] = useState(false);

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
      } else {
        setAnswerDoc(ans);
        setDisplayAnswerDoc(ans);
        setCitationsDoc(res.citations || []);
      }
    } catch (e) {
      setErrorDoc((e as any)?.message || 'Doc QA failed');
      setAnswerDoc('No response received. Check backend/OpenAI key.');
      setDisplayAnswerDoc('No response received. Check backend/OpenAI key.');
    } finally {
      setLoadingDoc(false);
      setStreamDoc(false);
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
    } catch (e) {
      setErrorPub((e as any)?.message || 'Search failed');
    }
    setLoadingPub(false);
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
          <div className="center-hero">
            <h1>Connect your research. Get instant answers</h1>
            <p>Upload documents on the left; get doc-grounded answers here.</p>
            <div className="center-graphic" />
          </div>

          <div className="section tight full-width">
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

          <div className="answers-area">
            {errorDoc && <div className="alert">{errorDoc}</div>}
            {loadingDoc && (
              <div className="empty-state tall">
                <div className="empty-icon" />
                <div>
                  <h3>Thinking…</h3>
                  <p>Retrieving from your uploaded documents.</p>
                </div>
              </div>
            )}
            {!loadingDoc && !resultsDoc.length && !answerDoc && (
              <div className="empty-state tall">
                <div className="empty-icon" />
                <div>
                  <h3>Ask about your docs</h3>
                  <p>Upload documents and ask a question to see answers here.</p>
                </div>
              </div>
            )}
            {answerDoc && (
              <AnswerPanel answer={displayAnswerDoc} citations={citationsDoc} streaming={streamDoc} />
            )}
            {resultsDoc.length > 0 && <ResultsList scope="uploaded" results={resultsDoc} />}
          </div>
        </div>
      </div>

      <div className="panel panel-right">
        <div className="right-header">
          <h3>AI Assistant</h3>
          <label className="inline-toggle">
            <input
              type="checkbox"
              checked={includeUploadsInPub}
              onChange={(e) => setIncludeUploadsInPub(e.target.checked)}
            />
            <span>Include my docs</span>
          </label>
        </div>
        {errorPub && <div className="alert">{errorPub}</div>}
        <SearchBar
          value={questionPub}
          onChange={setQuestionPub}
          onSubmit={() => handlePubAsk()}
          disabled={loadingPub}
          loading={loadingPub}
          onAdvanced={() => handlePubSearch(questionPub, 8)}
          inputRef={askPubRef}
          placeholder="Ask any question..."
          hideAdvanced
        />
        {loadingPub && (
          <div className="empty-state tall">
            <div className="empty-icon" />
            <div>
              <h3>Thinking…</h3>
              <p>Your questions about public data and the web will appear here.</p>
            </div>
          </div>
        )}
        {!loadingPub && !resultsPub.length && !answerPub && (
          <div className="empty-state tall">
            <div className="empty-icon" />
            <div>
              <h3>Ask any question…</h3>
              <p>Your questions about public data and the web will appear here.</p>
            </div>
          </div>
        )}
        {answerPub && <AnswerPanel answer={displayAnswerPub} citations={citationsPub} streaming={streamPub} />}
        {resultsPub.length > 0 && (
          <ResultsList scope={includeUploadsInPub ? 'uploaded' : 'public'} results={resultsPub} />
        )}
        <div className="chat-launch-right">
          <button onClick={() => setChatOpen(true)}>Chat</button>
        </div>
      </div>

      <AiChatWidget
        scope={includeUploadsInPub && (selectedDoc || docs.length) ? 'uploaded' : 'public'}
        docId={includeUploadsInPub && selectedDoc ? selectedDoc : undefined}
        open={chatOpen}
        onOpenChange={setChatOpen}
      />
    </div>
  );
}
