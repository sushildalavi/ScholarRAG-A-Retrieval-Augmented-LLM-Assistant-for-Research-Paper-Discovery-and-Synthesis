import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';
import type { LatestResearchPaper } from '../api/types';

type LandingPageProps = {
  authAvailable: boolean;
  signedIn: boolean;
  userLabel?: string | null;
  onOpenWorkspace: () => void;
  onSignIn: () => void;
  onSignOut: () => void;
};

const FEATURE_BLOCKS = [
  {
    eyebrow: 'Grounded answers',
    title: 'Answer first, evidence second',
    text: 'ScholarRAG keeps citations and snippets in a dedicated inspector so the chat stays clear and readable.',
  },
  {
    eyebrow: 'One workspace',
    title: 'Public literature and uploaded documents together',
    text: 'Move between paper discovery, synthesis, and document-grounded analysis without switching products.',
  },
  {
    eyebrow: 'Trust layer',
    title: 'Confidence and inspection built in',
    text: 'Every answer can be traced back to evidence, source links, and support signals instead of unsupported prose.',
  },
];

const WORKFLOW_STEPS = [
  'Search a topic and pull recent or relevant papers.',
  'Synthesize findings instead of reading raw search results.',
  'Upload documents and ask grounded questions with page-aware evidence.',
];

function LatestPaperCard({ paper }: { paper: LatestResearchPaper }) {
  return (
    <article className="landing-paper-card">
      <div className="landing-paper-meta">
        <span className="landing-paper-provider">{paper.provider}</span>
        {paper.venue && <span>{paper.venue}</span>}
        {paper.year && <span>{paper.year}</span>}
      </div>
      <h3>{paper.title}</h3>
      <p>{paper.abstract || paper.why_relevant || 'Recent research result.'}</p>
      <div className="landing-paper-foot">
        <div className="landing-paper-authors">
          {(paper.authors || []).slice(0, 3).join(', ') || 'Authors unavailable'}
        </div>
        <div className="landing-paper-links">
          {paper.url && (
            <a href={paper.url} target="_blank" rel="noreferrer">
              Open ↗
            </a>
          )}
          {paper.pdf_url && (
            <a href={paper.pdf_url} target="_blank" rel="noreferrer">
              PDF ↗
            </a>
          )}
        </div>
      </div>
    </article>
  );
}

export function LandingPage({
  authAvailable,
  signedIn,
  userLabel,
  onOpenWorkspace,
  onSignIn,
  onSignOut,
}: LandingPageProps) {
  const [topic, setTopic] = useState('');
  const [papers, setPapers] = useState<LatestResearchPaper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const headerLabel = signedIn ? (userLabel || 'Signed in') : 'Guest session';
  const activeTopic = useMemo(() => topic.trim(), [topic]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError('');
    api.latestResearch({ topic: activeTopic || undefined, limit: 6, days: 45 })
      .then((res) => {
        if (cancelled) return;
        setPapers(res.results || []);
      })
      .catch((e: any) => {
        if (cancelled) return;
        setError(e?.message || 'Failed to load latest research.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeTopic]);

  return (
    <div className="landing-shell">
      <header className="landing-topbar">
        <div className="landing-brand">
          <div className="landing-brand-mark">SR</div>
          <div>
            <div className="landing-brand-name">ScholarRAG</div>
            <div className="landing-brand-sub">Citation-grounded research workspace</div>
          </div>
        </div>
        <div className="landing-topbar-actions">
          <div className="landing-user-pill">{headerLabel}</div>
          {authAvailable && signedIn ? (
            <button className="landing-btn landing-btn-ghost" type="button" onClick={onSignOut}>
              Sign out
            </button>
          ) : authAvailable ? (
            <button className="landing-btn landing-btn-ghost" type="button" onClick={onSignIn}>
              Sign in with Google
            </button>
          ) : null}
          <button className="landing-btn landing-btn-primary" type="button" onClick={onOpenWorkspace}>
            Open workspace
          </button>
        </div>
      </header>

      <main className="landing-main">
        <section className="landing-hero">
          <div className="landing-hero-copy">
            <div className="landing-kicker">Research assistant, not a retrieval dump</div>
            <h1>Find papers, analyze documents, and inspect evidence in one workspace.</h1>
            <p>
              ScholarRAG combines public scholarly discovery with document-grounded Q&amp;A, evidence inspection,
              and confidence-aware answers. It is built for real research workflows, not generic chatbot demos.
            </p>
            <div className="landing-hero-actions">
              <button className="landing-btn landing-btn-primary" type="button" onClick={onOpenWorkspace}>
                Start in the workspace
              </button>
              {authAvailable && !signedIn && (
                <button className="landing-btn landing-btn-secondary" type="button" onClick={onSignIn}>
                  Sign in with Google
                </button>
              )}
            </div>
            <div className="landing-proof-row">
              <div className="landing-proof-card">
                <strong>Public research</strong>
                <span>Query recent papers and synthesize findings.</span>
              </div>
              <div className="landing-proof-card">
                <strong>Uploaded documents</strong>
                <span>Ask grounded questions over PDFs with page-aware evidence.</span>
              </div>
              <div className="landing-proof-card">
                <strong>Evidence inspector</strong>
                <span>Keep source details separate from the answer surface.</span>
              </div>
            </div>
          </div>
          <div className="landing-hero-panel">
            <div className="landing-panel-kicker">Why ScholarRAG is different</div>
            <div className="landing-panel-title">A serious research workspace</div>
            <ul className="landing-panel-list">
              <li>Answer-first chat, with evidence kept in a dedicated inspector</li>
              <li>One place for paper discovery, literature synthesis, and uploaded-doc analysis</li>
              <li>Confidence-aware outputs instead of opaque, unsupported prose</li>
            </ul>
          </div>
        </section>

        <section className="landing-section">
          <div className="landing-section-head">
            <div>
              <div className="landing-kicker">Product principles</div>
              <h2>Built for researchers who care about grounding.</h2>
            </div>
          </div>
          <div className="landing-feature-grid">
            {FEATURE_BLOCKS.map((item) => (
              <article key={item.title} className="landing-feature-card">
                <div className="landing-feature-eyebrow">{item.eyebrow}</div>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="landing-section">
          <div className="landing-section-head">
            <div>
              <div className="landing-kicker">How it works</div>
              <h2>Three steps, one workflow.</h2>
            </div>
          </div>
          <div className="landing-workflow">
            {WORKFLOW_STEPS.map((step, idx) => (
              <div key={step} className="landing-step-card">
                <div className="landing-step-index">0{idx + 1}</div>
                <div className="landing-step-copy">{step}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="landing-section">
          <div className="landing-section-head landing-section-head-feed">
            <div>
              <div className="landing-kicker">Latest research</div>
              <h2>Recent papers to explore right now.</h2>
            </div>
            <div className="landing-feed-controls">
              <input
                className="landing-topic-input"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Filter by topic, e.g. transformer interpretability"
              />
            </div>
          </div>
          {loading ? (
            <div className="landing-paper-grid">
              {Array.from({ length: 6 }).map((_, idx) => (
                <div key={idx} className="landing-paper-card landing-paper-skeleton">
                  <div className="landing-sk-row landing-sk-row-sm" />
                  <div className="landing-sk-row" />
                  <div className="landing-sk-row landing-sk-row-wide" />
                  <div className="landing-sk-row landing-sk-row-mid" />
                </div>
              ))}
            </div>
          ) : error ? (
            <div className="landing-feed-empty">{error}</div>
          ) : (
            <div className="landing-paper-grid">
              {papers.map((paper) => (
                <LatestPaperCard key={`${paper.provider}-${paper.id || paper.title}`} paper={paper} />
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
