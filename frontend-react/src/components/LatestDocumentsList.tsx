import { DocumentRow } from '../api/types';

type Props = {
  documents: DocumentRow[];
  selectedId?: number | null;
  onSelect: (id: number | null) => void;
  onDelete: (id: number) => void;
};

export function LatestDocumentsList({ documents, selectedId, onSelect, onDelete }: Props) {
  if (!documents.length) {
    return <div className="library-empty">No documents yet.</div>;
  }
  return (
    <div className="library-list">
      {documents.map((d) => {
        const statusClass = d.status === 'ready' ? 'badge-success' : d.status === 'error' ? 'badge-error' : 'badge-processing';
        const statusText = d.status === 'ready' ? 'Processed' : d.status === 'error' ? 'Error' : 'Processing';
        return (
          <div
            key={d.id}
            className={`library-item ${selectedId === d.id ? 'selected' : ''}`}
          >
            <div className="lib-icon" onClick={() => onSelect(selectedId === d.id ? null : d.id)}>📄</div>
            <div className="lib-meta" onClick={() => onSelect(selectedId === d.id ? null : d.id)}>
              <div className="lib-title" title={d.title}>{d.title}</div>
              <div className="lib-status">
                <span className={`badge ${statusClass}`}>{statusText}</span>
              </div>
            </div>
            <button className="lib-delete" title="Delete" onClick={() => onDelete(d.id)}>🗑</button>
          </div>
        );
      })}
    </div>
  );
}
