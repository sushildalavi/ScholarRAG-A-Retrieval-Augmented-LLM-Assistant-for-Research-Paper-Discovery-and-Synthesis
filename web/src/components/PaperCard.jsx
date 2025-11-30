import React from "react";

const Tag = ({ label }) => (
  <span className="pill" style={{ background: "#f5f0ff", color: "#6b3df5" }}>
    {label}
  </span>
);

export function PaperCard({ paper, onBookmark }) {
  const summary =
    paper.snippet ||
    paper.abstract ||
    paper.summary ||
    "";
  const authorNames =
    (paper.authors || [])
      .map((a) => a.display_name || a.name || a)
      .filter(Boolean)
      .slice(0, 3);
  const fallbackSummary = authorNames.length ? authorNames.join(", ") : paper.source || "";
  const shortSummary = summary
    ? summary.slice(0, 320) + (summary.length > 320 ? "…" : "")
    : fallbackSummary;

  const previewLabel = paper.year || "—";
  const concepts = (paper.concepts || []).slice(0, 4);
  const dateLabel = paper.date || paper.year || "";
  return (
    <div className="card" style={{ display: "flex", gap: 16, padding: 16 }}>
      <div style={{ width: 140, minWidth: 140, background: "#f5f7fb", borderRadius: 12, border: "1px solid #e6e9f1", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 12, textAlign: "center" }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: "#6b7385" }}>{previewLabel}</div>
        <div style={{ fontSize: 12, color: "#9aa3b5", marginTop: 8 }}>Preview</div>
      </div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ fontWeight: 700, color: "#7a849d", fontSize: 12 }}>{dateLabel}</span>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            {concepts.map((c) => (
              <Tag key={c} label={c} />
            ))}
          </div>
        </div>
        <h3 style={{ margin: 0, fontSize: 22, color: "#b22222" }}>
          {paper.source_url ? (
            <a href={paper.source_url} target="_blank" rel="noreferrer" style={{ color: "#b22222" }}>
              {paper.title}
            </a>
          ) : (
            paper.title
          )}
        </h3>
        <div className="card" style={{ padding: 12, background: "#fffdfc", borderColor: "#f3e1da" }}>
          <p style={{ margin: 0, color: "#3d4960", lineHeight: 1.55 }}>{shortSummary}</p>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <button className="btn" onClick={onBookmark} style={{ background: "#fff", borderColor: "#e6e9f1" }}>
            🔖 Bookmark
          </button>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {(paper.authors || []).slice(0, 3).map((a) => (
              <span key={a.display_name || a.id} className="pill">
                {a.display_name || "Author"}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
