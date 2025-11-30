import React, { useEffect, useMemo, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { SearchBar } from "./components/SearchBar";
import { Filters } from "./components/Filters";
import { PaperCard } from "./components/PaperCard";
import { RightRail } from "./components/RightRail";
import { AnswerBox } from "./components/AnswerBox";
import { GoogleButton } from "./components/GoogleButton";

const BACKEND = import.meta.env.VITE_BACKEND_BASE_URL || "http://127.0.0.1:8000";
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

export default function App() {
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState([]);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState("light");
  const [activeSection, setActiveSection] = useState("explore");
  const [showPersonalize, setShowPersonalize] = useState(false);
  const [showSignIn, setShowSignIn] = useState(false);
  const [filters, setFilters] = useState({ hot: false });
  const [k, setK] = useState(10);
  const [lastQuery, setLastQuery] = useState("");
  const [listening, setListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [user, setUser] = useState(null);
  const USER_KEY = "scholarrag_user";
  const [personalizePrefs, setPersonalizePrefs] = useState({
    topics: new Set(),
    recency: "6m",
    sources: new Set(["arxiv", "openalex"]),
  });
  const [latestFeed, setLatestFeed] = useState([]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const sortedPapers = useMemo(() => {
    if (activeSection === "soa" || filters.hot) {
      return [...papers].sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
    }
    return papers;
  }, [papers, activeSection, filters]);

  const latestPapers = useMemo(() => {
    return sortedPapers.slice(0, 5);
  }, [sortedPapers]);

  useEffect(() => {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (Speech) setSpeechSupported(true);
    try {
      const cached = localStorage.getItem(USER_KEY);
      if (cached) {
        setUser(JSON.parse(cached));
      }
    } catch {
      /* ignore */
    }
    refreshLatest();
  }, []);

  const refreshLatest = async () => {
    try {
      const resp = await fetch(`${BACKEND}/feed/latest?limit=5`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setLatestFeed(data.results || []);
    } catch (err) {
      // silently ignore for now
    }
  };

  const toggleMic = () => {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) {
      setError("This browser does not support speech input.");
      return;
    }
    if (listening) {
      setListening(false);
      return;
    }
    const rec = new Speech();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      setQuery(transcript);
      setListening(false);
      runSearch({ append: false, targetK: k, forcedQuery: transcript });
    };
    rec.onerror = () => setListening(false);
    rec.onend = () => setListening(false);
    setListening(true);
    rec.start();
  };

  const mergePapers = (prev, next) => {
    const seen = new Set(prev.map((p) => p.id));
    const merged = [...prev];
    for (const p of next) {
      if (!seen.has(p.id)) {
        merged.push(p);
        seen.add(p.id);
      }
    }
    return merged;
  };

  const runSearch = async ({ append = false, targetK, forcedQuery } = {}) => {
    const q = (forcedQuery || query).trim();
    if (!q) return;
    const desiredK = targetK || k;
    setLoading(true);
    setError("");
    if (!append) {
      setAnswer("");
    }
    try {
      const resp = await fetch(`${BACKEND}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k: desiredK, multi_hop: false, user_id: user?.email || "guest" }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const incoming = (data.sources || []).map((s, idx) => ({ ...s, date: s.year, id: s.openalex_id || s.arxiv_id || `src-${idx}` }));
      setAnswer(data.answer || answer);
      setPapers((prev) => (append ? mergePapers(prev, incoming) : incoming));
      setLastQuery(q);
      setK(desiredK);
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleCredential = async (credential) => {
    try {
      const resp = await fetch(`${BACKEND}/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: credential }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const u = { name: data.name, email: data.email, picture: data.picture, token: data.token || credential };
      setUser(u);
      try {
        localStorage.setItem(USER_KEY, JSON.stringify(u));
      } catch {
        /* ignore */
      }
      setShowSignIn(false);
    } catch (err) {
      setError(err.message || "Google sign-in failed");
    }
  };

  const signOut = () => {
    setUser(null);
    try {
      localStorage.removeItem(USER_KEY);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="app-shell">
      <Sidebar
        active={activeSection}
        onSelect={(s) => setActiveSection(s)}
        darkMode={theme === "dark"}
        onToggleDark={() => setTheme(theme === "dark" ? "light" : "dark")}
        onSignIn={() => setShowSignIn(true)}
      />
      <main style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <SearchBar
          value={query}
          onChange={setQuery}
          onSubmit={() => runSearch({ append: false, targetK: k })}
          loading={loading}
          k={k}
          onKChange={setK}
          listening={listening}
          onMicToggle={toggleMic}
        />
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <div className="pill">Deep Research</div>
          <div className="pill">History</div>
          {!speechSupported && <div className="pill" style={{ background: "#ffecec", color: "#b02222" }}>No speech API</div>}
          {user ? (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div className="pill" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {user.picture && <img src={user.picture} alt="avatar" style={{ width: 20, height: 20, borderRadius: "50%" }} />}
                <span>{user.name || user.email}</span>
              </div>
              <button className="btn" onClick={signOut}>Sign out</button>
            </div>
          ) : (
            <button className="btn" onClick={() => setShowSignIn(true)}>Sign in</button>
          )}
        </div>
        <Filters hot={filters.hot} onHotToggle={() => setFilters((f) => ({ ...f, hot: !f.hot }))} />
        <AnswerBox text={answer} loading={loading && !answer} />
        {error && <div className="card" style={{ padding: 12, color: "#b02222", borderColor: "#f5c4c4" }}>Error: {error}</div>}
        {loading && !papers.length && (
          <div className="card" style={{ padding: 20 }}>
            <div className="skeleton" style={{ height: 18, width: "30%", borderRadius: 8, marginBottom: 10 }} />
            <div className="skeleton" style={{ height: 120, width: "100%", borderRadius: 12 }} />
          </div>
        )}
        {sortedPapers.map((paper) => (
          <PaperCard key={paper.id} paper={paper} onBookmark={() => {}} />
        ))}
        {sortedPapers.length > 0 && (
          <div style={{ display: "flex", justifyContent: "center", marginTop: 8 }}>
            <button
              className="btn"
              disabled={loading}
              onClick={() => runSearch({ append: true, targetK: Math.min(k + 10, 100) })}
            >
              {loading ? "Loading..." : "Load more"}
            </button>
          </div>
        )}
      </main>
      <RightRail latestPapers={latestFeed} onPersonalize={() => setShowPersonalize(true)} onRefreshLatest={refreshLatest} />

      {showPersonalize && (
        <div className="backdrop" onClick={() => setShowPersonalize(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0, marginBottom: 6 }}>Personalize Your Feed</h3>
            <p style={{ color: "var(--muted)", marginTop: 0 }}>Pick domains and filters to shape your feed.</p>
            <div style={{ margin: "12px 0" }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Topics</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                {["LLMs", "Biology", "Robotics", "NLP", "Vision", "Theory", "Healthcare", "Economics", "Medicine"].map((t) => {
                  const active = personalizePrefs.topics.has(t);
                  return (
                    <span
                      key={t}
                      className="pill"
                      onClick={() =>
                        setPersonalizePrefs((prev) => {
                          const next = new Set(prev.topics);
                          active ? next.delete(t) : next.add(t);
                          return { ...prev, topics: next };
                        })
                      }
                      style={{ cursor: "pointer", background: active ? "var(--accent)" : "var(--pill)", color: active ? "#fff" : "var(--muted)" }}
                    >
                      {t}
                    </span>
                  );
                })}
              </div>
            </div>
            <div style={{ margin: "12px 0" }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Recency</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[
                  { label: "Last month", value: "1m" },
                  { label: "6 months", value: "6m" },
                  { label: "1 year", value: "1y" },
                  { label: "All time", value: "all" },
                ].map((r) => (
                  <button
                    key={r.value}
                    className="btn"
                    onClick={() => setPersonalizePrefs((p) => ({ ...p, recency: r.value }))}
                    style={{ background: personalizePrefs.recency === r.value ? "var(--accent)" : "var(--panel)", color: personalizePrefs.recency === r.value ? "#fff" : "var(--text)" }}
                  >
                    {r.label}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ margin: "12px 0" }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Sources</div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {["arxiv", "openalex", "user"].map((s) => {
                  const active = personalizePrefs.sources.has(s);
                  return (
                    <button
                      key={s}
                      className="btn"
                      onClick={() =>
                        setPersonalizePrefs((prev) => {
                          const next = new Set(prev.sources);
                          active ? next.delete(s) : next.add(s);
                          return { ...prev, sources: next };
                        })
                      }
                      style={{ background: active ? "var(--accent)" : "var(--panel)", color: active ? "#fff" : "var(--text)" }}
                    >
                      {s}
                    </button>
                  );
                })}
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 10 }}>
              <button className="btn" onClick={() => setShowPersonalize(false)}>Cancel</button>
              <button className="btn" style={{ background: "var(--accent)", color: "#fff", borderColor: "var(--accent)" }} onClick={() => setShowPersonalize(false)}>
                Save preferences
              </button>
            </div>
          </div>
        </div>
      )}

      {showSignIn && (
        <div className="backdrop" onClick={() => setShowSignIn(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0, marginBottom: 6 }}>Sign in</h3>
            <p style={{ color: "var(--muted)", marginTop: 0 }}>Choose a method to sync bookmarks and personalization.</p>
            {GOOGLE_CLIENT_ID ? (
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 10 }}>
                <GoogleButton clientId={GOOGLE_CLIENT_ID} onCredential={handleGoogleCredential} />
              </div>
            ) : (
              <div className="card" style={{ padding: 10, borderColor: "#f5c4c4", color: "#b02222" }}>
                Set VITE_GOOGLE_CLIENT_ID to enable Google sign-in.
              </div>
            )}
            <div className="card" style={{ padding: 14, marginTop: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <a href="#" style={{ color: "#1d4ed8", fontWeight: 600 }}>Register for a free account</a>
                <a href="#" style={{ color: "#1d4ed8", fontWeight: 600 }}>Forgot your password?</a>
              </div>
              <label style={{ display: "block", fontWeight: 600, marginBottom: 4 }}>Username or Email</label>
              <input style={{ width: "100%", padding: 12, borderRadius: 10, border: "1px solid var(--border)", background: "var(--panel)", color: "var(--text)", marginBottom: 10 }} placeholder="you@example.com" />
              <label style={{ display: "block", fontWeight: 600, marginBottom: 4 }}>Password</label>
              <input type="password" style={{ width: "100%", padding: 12, borderRadius: 10, border: "1px solid var(--border)", background: "var(--panel)", color: "var(--text)" }} placeholder="********" />
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
                <input type="checkbox" defaultChecked />
                <span>Remember Me</span>
              </div>
              <button className="btn" style={{ background: "#1d4ed8", color: "#fff", borderColor: "#1d4ed8", marginTop: 12, width: "100%", justifyContent: "center" }} onClick={() => setShowSignIn(false)}>
                Login
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
