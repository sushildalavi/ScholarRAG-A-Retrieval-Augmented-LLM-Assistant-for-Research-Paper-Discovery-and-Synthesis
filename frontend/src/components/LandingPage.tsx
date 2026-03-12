import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { LatestResearchPaper } from '../api/types';

type LandingPageProps = {
  authAvailable: boolean;
  signedIn: boolean;
  userLabel?: string | null;
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  onOpenWorkspace: () => void;
  onSignIn: () => void;
  onSignOut: () => void;
};

type FeedSort = 'latest' | 'trending' | 'top_cited';
type SourceKind = 'openalex' | 'arxiv' | 'semantic' | 'crossref' | 'springer' | 'elsevier' | 'ieee';

const FEATURE_BLOCKS = [
  {
    eyebrow: 'Grounded answers',
    title: 'Answer first, evidence second',
    text: 'ScholarRAG keeps citations, snippets, and support traces in a separate inspector so the main answer stays clean.',
  },
  {
    eyebrow: 'One workspace',
    title: 'Public literature and uploaded documents together',
    text: 'Move from paper discovery to grounded document analysis without changing tools or losing your working context.',
  },
  {
    eyebrow: 'Trust layer',
    title: 'Evidence and confidence built in',
    text: 'Every answer can be inspected through cited sources, links, and support signals instead of unsupported prose.',
  },
];

const WORKFLOW_STEPS = [
  {
    index: '01',
    title: 'Search and filter live literature',
    text: 'Pull recent, trending, or highly cited work from scholarly sources without manually hopping across databases.',
  },
  {
    index: '02',
    title: 'Synthesize findings quickly',
    text: 'Ask conceptual or research questions and get a structured answer instead of a raw search-results dump.',
  },
  {
    index: '03',
    title: 'Ground answers in your own PDFs',
    text: 'Upload documents, select the right context, and inspect page-aware evidence in the same workspace.',
  },
];

const FEED_FILTERS: Array<{ value: FeedSort; label: string }> = [
  { value: 'latest', label: 'Latest' },
  { value: 'trending', label: 'Trending' },
  { value: 'top_cited', label: 'Top cited' },
];

const SOURCE_PARTNERS: Array<{ name: string; kind: SourceKind; logoSlug: string; logoColor?: string; homepage: string }> = [
  { name: 'OpenAlex', kind: 'openalex', logoSlug: 'openalex', logoColor: '8B5CF6', homepage: 'https://openalex.org' },
  { name: 'arXiv', kind: 'arxiv', logoSlug: 'arxiv', logoColor: 'B31B1B', homepage: 'https://arxiv.org' },
  { name: 'Semantic Scholar', kind: 'semantic', logoSlug: 'semanticscholar', logoColor: '1857B6', homepage: 'https://www.semanticscholar.org' },
  { name: 'Crossref', kind: 'crossref', logoSlug: 'crossref', logoColor: '0057B8', homepage: 'https://www.crossref.org' },
  { name: 'Springer', kind: 'springer', logoSlug: 'springernature', logoColor: '009FE3', homepage: 'https://link.springer.com' },
  { name: 'Elsevier', kind: 'elsevier', logoSlug: 'elsevier', logoColor: 'FF6C00', homepage: 'https://www.elsevier.com' },
  { name: 'IEEE', kind: 'ieee', logoSlug: 'ieee', logoColor: '00629B', homepage: 'https://ieeexplore.ieee.org' },
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

function ThemeIcon({ theme }: { theme: 'dark' | 'light' }) {
  return (
    <svg viewBox="0 0 24 24" className="landing-inline-icon" aria-hidden="true" fill="none">
      {theme === 'dark' ? (
        <path d="M12 3.5a1 1 0 0 1 1 1V6a1 1 0 1 1-2 0V4.5a1 1 0 0 1 1-1Zm0 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm7.5-4.5a1 1 0 0 1 1 1 8.5 8.5 0 1 1-8.5-8.5 1 1 0 1 1 0 2A6.5 6.5 0 1 0 18.5 12a1 1 0 0 1 1-1Z" fill="currentColor" />
      ) : (
        <path d="M21 12.8A8.8 8.8 0 1 1 11.2 3a7 7 0 1 0 9.8 9.8Z" fill="currentColor" />
      )}
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg viewBox="0 0 24 24" className="landing-inline-icon" aria-hidden="true" fill="none">
      <path d="M10.3 3.4a1 1 0 0 1 1.4-.5l.3.2a2 2 0 0 0 2 0l.3-.2a1 1 0 0 1 1.4.5l.5 1a2 2 0 0 0 1.5 1.1l1.1.2a1 1 0 0 1 .8 1.3l-.1.3a2 2 0 0 0 .4 2l.7.9a1 1 0 0 1 0 1.4l-.7.9a2 2 0 0 0-.4 2l.1.3a1 1 0 0 1-.8 1.3l-1.1.2a2 2 0 0 0-1.5 1.1l-.5 1a1 1 0 0 1-1.4.5l-.3-.2a2 2 0 0 0-2 0l-.3.2a1 1 0 0 1-1.4-.5l-.5-1a2 2 0 0 0-1.5-1.1l-1.1-.2a1 1 0 0 1-.8-1.3l.1-.3a2 2 0 0 0-.4-2l-.7-.9a1 1 0 0 1 0-1.4l.7-.9a2 2 0 0 0 .4-2l-.1-.3a1 1 0 0 1 .8-1.3l1.1-.2A2 2 0 0 0 9.8 5l.5-1Z" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function SourceLogo({ name, logoSlug, logoColor }: { name: string; logoSlug: string; logoColor?: string }) {
  const [failed, setFailed] = useState(false);
  const abbrev = name.split(' ').map((w) => w[0]).join('').slice(0, 3).toUpperCase();
  if (failed) {
    return (
      <span className="landing-source-abbrev" style={{ '--abbrev-color': `#${logoColor || '8b5cf6'}` } as React.CSSProperties}>
        {abbrev}
      </span>
    );
  }
  return (
    <img
      className="landing-source-logo"
      src={`https://cdn.simpleicons.org/${logoSlug}/${logoColor || 'FFFFFF'}`}
      alt={name}
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
    />
  );
}

function LatestPaperCard({ paper, index }: { paper: LatestResearchPaper; index: number }) {
  const href = paper.url || paper.pdf_url || '#';
  const summary = (paper.abstract || paper.why_relevant || 'Recent research result.').replace(/\s+/g, ' ').trim();
  const preview = summary.length > 220 ? `${summary.slice(0, 220)}…` : summary;
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
  theme,
  onToggleTheme,
  onOpenWorkspace,
  onSignIn,
  onSignOut,
}: LandingPageProps) {
  const [topic, setTopic] = useState('');
  const [sort, setSort] = useState<FeedSort>('latest');
  const [papers, setPapers] = useState<LatestResearchPaper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showSettings, setShowSettings] = useState(false);

  const activeTopic = topic.trim();

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError('');
    api
      .latestResearch({ topic: activeTopic || undefined, limit: 6, days: sort === 'top_cited' ? 365 : 45, sort })
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
          <span className="landing-brand-mark" aria-hidden="true">
            <span className="landing-mark-sheet landing-mark-sheet-back" />
            <span className="landing-mark-sheet landing-mark-sheet-front" />
            <span className="landing-mark-accent" />
          </span>
          <div>
            <div className="landing-brand-name">ScholarRAG</div>
            <div className="landing-brand-sub">Citation-grounded research workspace</div>
          </div>
        </div>
        <div className="landing-topbar-actions">
          <div className="landing-settings">
            <button
              className={`landing-btn landing-btn-ghost landing-btn-icon landing-settings-trigger${showSettings ? ' active' : ''}`}
              type="button"
              onClick={() => setShowSettings((value) => !value)}
            >
              <SettingsIcon />
              <span>Settings</span>
            </button>
            {showSettings && (
              <div className="landing-settings-menu">
                <div className="landing-settings-section">
                  <div className="landing-settings-label">Appearance</div>
                  <button className="landing-settings-action" type="button" onClick={() => { setShowSettings(false); onToggleTheme(); }}>
                    <span className="landing-settings-action-icon">{theme === 'dark' ? '☀' : '🌙'}</span>
                    Switch to {theme === 'dark' ? 'light' : 'dark'} mode
                  </button>
                </div>
                <div className="landing-settings-section">
                  <div className="landing-settings-label">Research feed</div>
                  <div className="landing-settings-row-label">Default sort</div>
                  <div className="landing-settings-chips">
                    {FEED_FILTERS.map((f) => (
                      <button
                        key={f.value}
                        type="button"
                        className={`landing-settings-chip${sort === f.value ? ' active' : ''}`}
                        onClick={() => setSort(f.value)}
                      >
                        {f.label}
                      </button>
                    ))}
                  </div>
                  <div className="landing-settings-hint">
                    {sort === 'top_cited' ? 'Searches ~8 years back for highest-cited work' : sort === 'trending' ? 'Scores papers by citations per day (decay-weighted)' : 'Most recently published across all sources'}
                  </div>
                </div>
                {authAvailable && (
                  <div className="landing-settings-section">
                    <div className="landing-settings-label">Account</div>
                    {signedIn ? (
                      <>
                        {userLabel ? <div className="landing-settings-user">{userLabel}</div> : null}
                        <button className="landing-settings-action danger" type="button" onClick={() => { setShowSettings(false); onSignOut(); }}>
                          Sign out
                        </button>
                      </>
                    ) : (
                      <button className="landing-settings-action google" type="button" onClick={() => { setShowSettings(false); onSignIn(); }}>
                        <GoogleIcon />
                        <span>Sign in with Google</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
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
                Open workspace →
              </button>
              <a
                className="landing-btn landing-btn-ghost landing-btn-icon"
                href="#research-feed"
                onClick={(e) => { e.preventDefault(); document.getElementById('research-feed')?.scrollIntoView({ behavior: 'smooth' }); }}
              >
                Explore feed ↓
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

        <section className="landing-section" id="research-feed">
          <div className="landing-section-head landing-section-head-feed">
            <div>
              <div className="landing-kicker">Latest research</div>
              <h2>
                {sort === 'top_cited' ? 'Most cited papers in the field.' : sort === 'trending' ? 'Trending papers right now.' : 'Recent papers to explore right now.'}
              </h2>
              {sort === 'top_cited' && (
                <p className="landing-feed-scope-hint">Searching across ~8 years — results are from OpenAlex ranked by total citations.</p>
              )}
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
          <div className="landing-marquee" aria-label="Scholarly sources">
            <div className="landing-marquee-track">
              {[...SOURCE_PARTNERS, ...SOURCE_PARTNERS].map((partner, index) => (
                <a
                  key={`${partner.name}-${index}`}
                  href={partner.homepage}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="landing-partner-chip"
                  title={`Visit ${partner.name}`}
                >
                  <SourceLogo name={partner.name} logoSlug={partner.logoSlug} logoColor={partner.logoColor} />
                  <span>{partner.name}</span>
                </a>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
