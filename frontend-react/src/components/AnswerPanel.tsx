import { Citation } from '../api/types';

type Props = {
  answer: string;
  citations: Citation[];
  streaming?: boolean;
};

export function AnswerPanel({ answer, citations, streaming }: Props) {
  if (!answer) return null;
  return (
    <div className="card">
      <h3>Answer</h3>
      <div className={`answer-text ${streaming ? 'answer-stream' : ''}`}>{answer}</div>
      {!!citations?.length && (
        <div className="citations">
          <h4>Citations</h4>
          <ul>
            {citations.map((c, idx) => (
              <li key={idx}>
                [{idx + 1}] {c.title || 'Untitled'}
                {c.year ? ` (${c.year})` : ''}
                {c.source ? ` • ${c.source}` : ''}
                {c.url && <a href={c.url} target="_blank" rel="noreferrer"> Open</a>}
                {c.doc_id && <span> • Doc {c.doc_id}</span>}
                {c.page && <span> p.{c.page}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
