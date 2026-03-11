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

type FeedSort = 'latest' | 'trending' | 'top_cited';

const FEATURE_BLOCKS = [
  {
    eyebrow: 'Grounded answers',
    title: 'Answer first, evidence second',
    text: 'ScholarRAG keeps citations, snippets, and source tracing in a dedicated inspector so the main conversation stays readable.',
  },
  {
    eyebrow: 'One workspace',
    title: 'Public literature and uploaded documents together',
    text: 'Move between paper discovery, synthesis, and PDF-grounded analysis without switching products or losing context.',
  },
  {
    eyebrow: 'Trust layer',
    title: 'Confidence and inspection built in',
    text: 'Every answer can be inspected through cited evidence, source links, and support signals instead of unsupported prose.',
  },
];

const WORKFLOW_STEPS = [
  {
    index: '01',
    title: 'Find the strongest papers fast',
    text: 'Search live scholarly sources, filter by what matters, and jump directly into the papers worth reading.',
  },
  {
    index: '02',
    title: 'Synthesize instead of skimming',
    text: 'Get answer-first summaries and literature comparisons, while detailed evidence stays in a separate inspector.',
  },
  {
    index: '03',
    title: 'Ground answers in your own files',
    text: 'Upload PDFs, select the right documents, and ask page-aware questions without losing the surrounding research context.',
  },
];

const FEED_FILTERS: Array<{ value: FeedSort; label: string }> = [
  { value: 'latest', label: 'Latest' },
  { value: 'trending', label: 'Trending' },
  { value: 'top_cited', label: 'Top cited' },
];

const SOURCE_PARTNERS = [
  'OpenAlex',
  'arXiv',
  'Semantic Scholar',
  'Crossref',
  'Springer',
  'Elsevier',
  'IEEE',
  'Google Scholar',
];

function GoogleIcon() {
  return (
    <svg aria-hidden="true" className="landing-google-icon" viewBox="0 0 24 24" fill="none">
      <path d="M21.8 12.23c0-.76-.07-1.49-.2-2.2H12v4.16h5.49a4.7 4.7 0 0 1-2.04 3.08v2.56h3.31c1.94-1.78 3.04-4.4 3.04-7.6Z" fill="#4285F4" />
      <path d="M12 22c2.75 0 5.05-.91 6.73-2.47l-3.31-2.56c-.92.62-2.09.99-3.42.99-2.63 0-4.86-1.77-5.65-4.16H2.93v2.64A10 10 0 0 0 12 22Z" fill="#34A853" />
      <path d="M6.35 13.8A5.98 5.98 0 0 1 6 12c0-.63.12-1.24.35-1.8V7.56H2.93A10 10 0 0 0 2 12c0 1.61.38 3.13 1.06 4.44l3.29-2.64Z" fill="#FBBC05" />
      <path d="M12 6.04c1.5 0 2.84.52 3.89 1.55l2.92-2.92C17.04 2.99 14.75 2 12 2a10 10 0 0 0-9.07 5.56L6.35 10.2c.79-2.39 3.02-4.16 5.65-4.16Z" fill="#EA4335" />
    </svg>
  );
}

function ScholarMark() {
  return (
    <div className="landing-brand-mark" aria-hidden="true">
      <span className="landing-brand-star" />
      <span className="landing-brand-orbit" />
    </div>
  );
}

function paperScholarHref(paper: LatestResearchPaper) {
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(paper.title)}`;
}

function LatestPaperCard({ paper, index }: { paper: LatestResearchPaper; index: number }) {
  const href = paper.url || paper.pdf_url || paperScholarHref(paper);
  const summary = (paper.abstract || paper.why_relevant || 'Recent research result.').replace(/\s+/g, ' ').trim();
  const preview = summary.length > 380 ? `${summary.slice(0, 380)}…` : summary;
  const providerLabel = (paper.provider || 'source').replace(/_/g, ' ');
  const authors = (paper.authors || []).slice(0, 5).join(', ') || 'Authors unavailable';

  return (
    <article className="landing-paper-card" style={{ animationDelay: `${index * 70}ms` }}>
      <div className="landing-paper-meta">
        <span className="landing-paper-provider">{providerLabel}</span>
        {paper.venue && <span>{paper.venue}</span>}
        {paper.year && <span>{paper.year}</span>}
        {typeof paper.citation_count === 'number' && paper.citation_count > 0 && <span>{paper.citation_count} cites</span>}
      </div>
      <h3 className="landing-paper-title">
        <a href={href} target="_blank" rel="noreferrer">
          {paper.title}
        </a>
      </h3>
      <div className="landing-paper-authors-line">{authors}</div>
      <p>{preview}</p>
      <div className="landing-paper-foot">
        <div className="landing-paper-links">
          <a href={href} target="_blank" rel="noreferrer">
            Open paper ↗
          </a>
          <a href={paperScholarHref(paper)} target="_blank" rel="noreferrer">
            Scholar ↗
          </a>
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
  const [sort, setSort] = useState<FeedSort>('latest');
  const [papers, setPapers] = useState<LatestResearchPaper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const headerLabel = signedIn ? userLabel || 'Signed in' : 'Guest session';
  const activeTopic = useMemo(() => topic.trim(), [topic]);
  const scholarTopicHref = useMemo(
    () =>
      `https://scholar.google.com/scholar?q=${encodeURIComponent(
        activeTopic || 'artificial intelligence machine learning natural language processing'
      )}`,
    [activeTopic]
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError('');
    api
      .latestResearch({ topic: activeTopic || undefined, limit: 6, days: 45, sort })
      .then((res) => {
        if (!cancelled) setPapers(res.results || []);
      })
      .catch((e: any) => {
        if (!cancelled) setError(e?.message || 'Failed to load latest research.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeTopic, sort]);

  return (
    <div className="landing-shell">
      <div className="landing-aurora landing-aurora-one" />
      <div className="landing-aurora landing-aurora-two" />
      <div className="landing-grid" />

      <header className="landing-topbar">
        <div className="landing-brand">
          <ScholarMark />
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
            <button className="landing-btn landing-btn-google" type="button" onClick={onSignIn}>
              <GoogleIcon />
              <span>Sign in with Google</span>
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
              and confidence-aware answers. It is built for research workflows that need clarity, source visibility,
              and faster iteration.
            </p>
            <div className="landing-hero-actions">
              <button className="landing-btn landing-btn-primary" type="button" onClick={onOpenWorkspace}>
                Start in the workspace
              </button>
              {authAvailable && !signedIn && (
                <button className="landing-btn landing-btn-secondary landing-btn-google" type="button" onClick={onSignIn}>
                  <GoogleIcon />
                  <span>Sign in with Google</span>
                </button>
              )}
              <a className="landing-btn landing-btn-ghost" href={scholarTopicHref} target="_blank" rel="noreferrer">
                Search Google Scholar
              </a>
            </div>
            <div className="landing-proof-row">
              <div className="landing-proof-card">
                <strong>Public research</strong>
                <span>Search recent, trending, or highly cited work across live scholarly sources.</span>
              </div>
              <div className="landing-proof-card">
                <strong>Uploaded documents</strong>
                <span>Ask grounded questions over PDFs with page-aware evidence and citations.</span>
              </div>
              <div className="landing-proof-card">
                <strong>Evidence inspector</strong>
                <span>Keep source details separate from the answer surface so the chat stays readable.</span>
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
            <div className="landing-hero-mini-grid">
              <div className="landing-hero-mini-card">
                <span>Answer-first chat</span>
                <strong>Evidence remains visible without overwhelming the response.</strong>
              </div>
              <div className="landing-hero-mini-card">
                <span>Research workflow</span>
                <strong>Move from discovery to grounded analysis in the same interface.</strong>
              </div>
            </div>
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
            {FEATURE_BLOCKS.map((item, index) => (
              <article key={item.title} className="landing-feature-card" style={{ animationDelay: `${index * 60}ms` }}>
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
            {WORKFLOW_STEPS.map((step, index) => (
              <div key={step.title} className="landing-step-card" style={{ animationDelay: `${index * 70}ms` }}>
                <div className="landing-step-index">{step.index}</div>
                <div className="landing-step-copy">
                  <strong>{step.title}</strong>
                  <span>{step.text}</span>
                </div>
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
              <div className="landing-feed-filters">
                {FEED_FILTERS.map((filter) => (
                  <button
                    key={filter.value}
                    type="button"
                    className={sort === filter.value ? 'landing-filter-chip active' : 'landing-filter-chip'}
                    onClick={() => setSort(filter.value)}
                  >
                    {filter.label}
                  </button>
                ))}
              </div>
              <input
                className="landing-topic-input"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Filter by topic, e.g. transformer interpretability"
              />
              <a className="landing-btn landing-btn-ghost landing-btn-inline" href={scholarTopicHref} target="_blank" rel="noreferrer">
                Search on Google Scholar
              </a>
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
              {papers.map((paper, index) => (
                <LatestPaperCard key={`${paper.provider}-${paper.id || paper.title}`} paper={paper} index={index} />
              ))}
            </div>
          )}
        </section>

        <section className="landing-section landing-partners-section">
          <div className="landing-section-head">
            <div>
              <div className="landing-kicker">Sources we rely on</div>
              <h2>Connected to the scholarly ecosystem.</h2>
            </div>
          </div>
          <div className="landing-partner-grid">
            {SOURCE_PARTNERS.map((partner, index) => (
              <div key={partner} className="landing-partner-card" style={{ animationDelay: `${index * 50}ms` }}>
                <span className="landing-partner-badge" />
                <span>{partner}</span>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
