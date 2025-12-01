import { useEffect, useState } from 'react';
import { api } from './api/client';
import { SearchBar } from './components/SearchBar';
import { ResultsList } from './components/ResultsList';
import { UploadPanel } from './components/UploadPanel';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

type Doc = {
  id: number;
  title: string;
  status: string;
  pages?: number;
  created_at?: string;
};

export default function App() {
  const [results, setResults] = useState<any[]>([]);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [answer, setAnswer] = useState<string>('');

  const loadDocs = async () => {
    try {
      const res = await api.listDocs(10);
      setDocs(res.documents || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadDocs();
  }, []);

  const onSearch = async (query: string, k: number) => {
    setAnswer('');
    try {
      const res = await api.searchChunks(query, k);
      setResults(res.results || []);
    } catch (e) {
      console.error(e);
    }
  };

  const onAsk = async () => {
    try {
      const res = await api.qa(results.length ? 'Summarize top results' : 'Explain', 8);
      setAnswer(res.answer);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="layout">
      <header>
        <div className="logo">ScholarRAG</div>
        <SearchBar onSearch={onSearch} />
      </header>
      <main>
        <section className="left">
          <UploadPanel apiBase={API_BASE} />
          <div className="card">
            <h3>Latest documents</h3>
            <ul>
              {docs.map((d) => (
                <li key={d.id}>{d.title} ({d.status})</li>
              ))}
            </ul>
          </div>
        </section>
        <section className="right">
          <div className="card actions">
            <button onClick={onAsk} disabled={!results.length}>Answer from results</button>
          </div>
          <ResultsList results={results} />
          {answer && (
            <div className="card">
              <h3>Answer</h3>
              <pre>{answer}</pre>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
