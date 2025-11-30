import React from "react";

const NavItem = ({ icon, label, active, onClick }) => (
  <button
    className="btn"
    onClick={onClick}
    style={{
      width: "100%",
      justifyContent: "flex-start",
      background: active ? "var(--pill)" : "var(--panel)",
      borderColor: active ? "var(--accent-2)" : "var(--border)",
      color: active ? "var(--accent-2)" : "var(--text)",
    }}
  >
    <span style={{ fontSize: 18 }}>{icon}</span>
    {label}
  </button>
);

export function Sidebar({ active, onSelect, darkMode, onToggleDark, onSignIn }) {
  return (
    <aside className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12, position: "sticky", top: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <div style={{ width: 38, height: 38, borderRadius: "12px", background: "#f5e7ff", display: "grid", placeItems: "center", color: "#7b2cbf", fontWeight: 800 }}>
          α
        </div>
        <div style={{ fontWeight: 700, fontSize: 18 }}>ScholarRAG</div>
      </div>
      <NavItem icon="🔍" label="Explore" active={active === "explore"} onClick={() => onSelect("explore")} />
      <NavItem icon="🏆" label="State of the Art" active={active === "soa"} onClick={() => onSelect("soa")} />
      <NavItem icon="🔐" label="Sign In" active={false} onClick={onSignIn} />
      <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 10 }}>
        <NavItem icon="🧪" label="Labs" onClick={() => onSelect("labs")} />
        <NavItem icon="💬" label="Feedback" onClick={() => onSelect("feedback")} />
        <NavItem icon={darkMode ? "🌞" : "🌙"} label={darkMode ? "Light mode" : "Dark mode"} onClick={onToggleDark} />
      </div>
    </aside>
  );
}
