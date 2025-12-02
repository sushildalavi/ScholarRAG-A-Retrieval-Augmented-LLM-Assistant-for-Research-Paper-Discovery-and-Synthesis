import { ChunkResult } from '../api/types';

type Props = {
  scope: 'uploaded' | 'public';
  results: any[];
};

export function ResultsList({ scope, results }: Props) {
  if (!results?.length) return <div className="card">No results yet.</div>;

  return (
    <div className="results">
      {results.map((r, idx) => {
        if (scope === 'uploaded') {
          const cr = r as ChunkResult;
          return (
            <div key={cr.id || idx} className="card result">
              <div className="meta">
                <span className="pill">Doc {cr.document_id}</span>
                {cr.page_no && <span className="pill light">Page {cr.page_no}</span>}
                {cr.distance !== undefined && <span className="pill score">score {cr.distance.toFixed(4)}</span>}
              </div>
              <p>{cr.text}</p>
            </div>
          );
        }
        // public paper result
        return (
          <div key={r.paper_id || idx} className="card result">
            <div className="meta">
              <span className="pill">{r.source || 'paper'}</span>
              {r.year && <span className="pill light">{r.year}</span>}
            </div>
            <h4>{r.title}</h4>
            <p>{(r.abstract || r.summary || '').slice(0, 600)}{(r.abstract || r.summary || '').length > 600 ? '…' : ''}</p>
            {r.source_url && <a href={r.source_url} target="_blank" rel="noreferrer">Open</a>}
          </div>
        );
      })}
    </div>
  );
}
