type Props = {
  scope: 'uploaded' | 'public';
  onChange: (scope: 'uploaded' | 'public') => void;
};

export function ScopeSelector({ scope, onChange }: Props) {
  return (
    <div className="card">
      <h3>Scope</h3>
      <label><input type="radio" checked={scope === 'uploaded'} onChange={() => onChange('uploaded')} /> Uploaded docs</label><br />
      <label><input type="radio" checked={scope === 'public'} onChange={() => onChange('public')} /> Public papers</label>
    </div>
  );
}
