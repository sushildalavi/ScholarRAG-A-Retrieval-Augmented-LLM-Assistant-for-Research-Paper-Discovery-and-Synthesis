type Result = {
  id: number;
  document_id: number;
  text: string;
  page_no?: number;
  chunk_index?: number;
  distance?: number;
};

type Props = {
  results: Result[];
};

export function ResultsList({ results }: Props) {
  if (!results.length) return <div className="card">No results yet</div>;
  return (
    <div className="results">
      {results.map((r) => (
        <div key={r.id} className="card">
          <div className="meta">
            <span>Doc #{r.document_id}</span>
            {r.page_no && <span>Page {r.page_no}</span>}
            {r.distance !== undefined && <span className="pill">score {r.distance.toFixed(4)}</span>}
          </div>
          <p>{r.text.slice(0, 500)}{r.text.length > 500 ? '…' : ''}</p>
        </div>
      ))}
    </div>
  );
}
