import React from "react";

const Chip = ({ label }) => <span className="pill">{label}</span>;

export function Filters({ onHotToggle, hot }) {
  return (
    <div className="card" style={{ padding: 10, display: "flex", gap: 8, alignItems: "center", justifyContent: "space-between" }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Chip label="Papers" />
        <Chip label="Benchmarks" />
        <Chip label="Models" />
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <button className="btn" onClick={onHotToggle} style={{ background: hot ? "var(--pill)" : "var(--panel)", borderColor: hot ? "var(--accent-2)" : "var(--border)", color: hot ? "var(--accent-2)" : "var(--text)" }}>
          🔥 Hot
        </button>
        <button className="btn">↻ Refresh</button>
      </div>
    </div>
  );
}
