import { DocumentRow } from '../api/types';

type Props = {
  documents: DocumentRow[];
  selectedIds?: number[];
  onToggle: (id: number) => void;
  onDelete: (id: number) => void;
};

export function LatestDocumentsList({ documents, selectedIds = [], onToggle, onDelete }: Props) {
  if (!documents.length) {
    return <div className="library-empty">No documents yet.</div>;
  }
  return (
    <div className="library-list">
      {documents.map((d) => {
        const statusClass = d.status === 'ready' ? 'badge-success' : d.status === 'error' ? 'badge-error' : 'badge-processing';
        const statusText = d.status === 'ready' ? 'Processed' : d.status === 'error' ? 'Error' : 'Processing';
        const selected = selectedIds.includes(d.id);
        return (
          <div
            key={d.id}
            className={`library-item ${selected ? 'selected' : ''}`}
          >
            <button type="button" className={`lib-check ${selected ? 'selected' : ''}`} onClick={() => onToggle(d.id)} aria-label={selected ? 'Deselect document' : 'Select document'} />
            <div className="lib-icon" aria-hidden="true" onClick={() => onToggle(d.id)} />
            <div className="lib-meta" onClick={() => onToggle(d.id)}>
              <div className="lib-title" title={d.title}>{d.title}</div>
              <div className="lib-status">
                <span className={`badge ${statusClass}`}>{statusText}</span>
              </div>
            </div>
            <button type="button" className="lib-delete" title="Delete" onClick={() => onDelete(d.id)}>×</button>
          </div>
        );
      })}
    </div>
  );
}
