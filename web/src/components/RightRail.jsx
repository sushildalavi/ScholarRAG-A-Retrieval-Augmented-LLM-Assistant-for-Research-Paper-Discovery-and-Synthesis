import React from "react";

export function RightRail({ onPersonalize, latestPapers = [], onRefreshLatest }) {
  return (
    <aside className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 16 }}>
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div style={{ fontWeight: 800 }}>Latest papers</div>
          <button className="btn" onClick={onRefreshLatest} style={{ padding: "6px 10px", fontSize: 12 }}>↻</button>
        </div>
        <div className="card" style={{ padding: 10, maxHeight: 320, overflow: "auto" }}>
          {latestPapers.length === 0 && <div style={{ color: "var(--muted)" }}>No results yet</div>}
          {latestPapers.map((p, idx) => {
            const link = p.doi ? `https://doi.org/${p.doi}` : p.url;
            return (
              <div key={p.id || idx} style={{ marginBottom: 12, paddingBottom: 10, borderBottom: "1px solid var(--border)" }}>
                <div style={{ fontWeight: 700, fontSize: 14 }}>
                  {link ? (
                    <a href={link} target="_blank" rel="noreferrer" style={{ color: "var(--text)" }}>
                      {p.title || "Untitled"}
                    </a>
                  ) : (
                    p.title || "Untitled"
                  )}
                </div>
                <div style={{ fontSize: 12, color: "var(--muted)", display: "flex", gap: 8, alignItems: "center" }}>
                  <span>{p.year || "—"}</span>
                  {p.concepts && p.concepts.slice(0, 2).map((c) => (
                    <span key={c} className="pill">{c}</span>
                  ))}
                </div>
                {p.summary && <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>{p.summary}</div>}
              </div>
            );
          })}
        </div>
      </div>

      <div className="card" style={{ padding: 16, background: "var(--accent-2)", color: "#fff", border: "none" }}>
        <div style={{ fontWeight: 800, marginBottom: 8 }}>Personalize Your Feed</div>
        <button className="btn" onClick={onPersonalize} style={{ background: "#fff", color: "var(--accent-2)", borderColor: "#fff", width: "100%", justifyContent: "center" }}>
          Start Now →
        </button>
      </div>
    </aside>
  );
}
