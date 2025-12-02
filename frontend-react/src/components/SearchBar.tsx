import { RefObject } from 'react';

type Props = {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onAdvanced: () => void;
  disabled?: boolean;
  loading?: boolean;
  inputRef?: RefObject<HTMLInputElement>;
  placeholder?: string;
  hideAdvanced?: boolean;
};

export function SearchBar({
  value,
  onChange,
  onSubmit,
  onAdvanced,
  disabled,
  loading,
  inputRef,
  placeholder,
  hideAdvanced,
}: Props) {
  return (
    <div className="search-card">
      <div className="search-input-wrap">
        <span className="search-icon">🔎</span>
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder || "Ask a question about your documents..."}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              onSubmit();
            }
          }}
          disabled={disabled || loading}
        />
      </div>
      <div className="search-actions">
        <button onClick={onSubmit} disabled={disabled || loading}>
          {loading ? 'Thinking…' : 'Search'}
        </button>
        {!hideAdvanced && (
          <button className="ghost" onClick={onAdvanced} disabled={disabled || loading}>⚙</button>
        )}
      </div>
    </div>
  );
}
