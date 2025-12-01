import { useState } from 'react';

type Props = {
  onSearch: (query: string, k: number) => void;
};

export function SearchBar({ onSearch }: Props) {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(10);

  return (
    <div className="search-bar">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask or search papers..."
      />
      <div className="controls">
        <label>k={k}</label>
        <input type="range" min={1} max={30} value={k} onChange={(e) => setK(Number(e.target.value))} />
      </div>
      <button onClick={() => onSearch(query, k)}>Search</button>
    </div>
  );
}
