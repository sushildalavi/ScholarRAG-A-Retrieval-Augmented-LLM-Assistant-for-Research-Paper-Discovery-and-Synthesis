import React from "react";

export function SearchBar({ value, onChange, onSubmit, loading, k, onKChange, listening, onMicToggle }) {
  return (
    <div className="card" style={{ padding: 14, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
      <div className="pill" style={{ background: "var(--pill)", color: "var(--muted)" }}>🔎</div>
      <input
        style={{ border: "none", outline: "none", flex: 1, minWidth: 220, fontSize: 17, background: "transparent" }}
        placeholder="Ask or search papers..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSubmit();
        }}
      />
      <button className="btn" type="button" onClick={onMicToggle} style={{ background: listening ? "var(--accent-2)" : "var(--panel)", color: listening ? "#fff" : "var(--text)" }}>
        {listening ? "Stop 🎙️" : "Speak 🎤"}
      </button>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span className="pill" style={{ background: "var(--pill)" }}>k={k}</span>
        <input type="range" min={5} max={100} step={5} value={k} onChange={(e) => onKChange(Number(e.target.value))} />
      </div>
      <button className="btn" onClick={onSubmit} disabled={loading} style={{ background: "var(--accent)", color: "#fff", borderColor: "var(--accent)" }}>
        {loading ? "Searching..." : "Search"}
      </button>
    </div>
  );
}
