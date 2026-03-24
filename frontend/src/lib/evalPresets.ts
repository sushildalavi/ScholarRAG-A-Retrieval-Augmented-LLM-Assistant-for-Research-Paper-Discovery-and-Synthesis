import type { DocumentRow, EvalCase, JudgeCasePayload } from '../api/types';

export const DEFAULT_EVAL_PRESET_COUNT = 120;

type PromptParts = {
  title: string;
  keywords: string;
  focus: string;
};

type RetrievalPromptBuilder = (parts: PromptParts) => string;
type JudgePromptBuilder = (parts: PromptParts) => string;
type JudgeMultiPromptBuilder = (titles: string, focus: string) => string;

const TITLE_STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'for',
  'from',
  'in',
  'of',
  'on',
  'or',
  'the',
  'to',
  'with',
]);

const RETRIEVAL_FOCI = [
  'the research problem',
  'the main contributions',
  'the proposed methodology',
  'the experimental setup',
  'the datasets or benchmarks',
  'the model architecture',
  'the evaluation protocol',
  'the quantitative results',
  'the baseline comparisons',
  'the main findings',
  'the limitations',
  'the future work',
] as const;

const RETRIEVAL_PROMPTS: readonly RetrievalPromptBuilder[] = [
  ({ title, focus }) => `In "${title}", what does the paper say about ${focus}?`,
  ({ title, focus }) => `What are the key points about ${focus} in "${title}"?`,
  ({ title, focus }) => `Summarize how "${title}" discusses ${focus}.`,
  ({ title, focus }) => `How does "${title}" describe ${focus}?`,
  ({ title, focus, keywords }) => `For ${keywords}, what is the paper's treatment of ${focus}?`,
  ({ title, focus }) => `What should I know about ${focus} from "${title}"?`,
  ({ title, focus }) => `Explain the paper's position on ${focus} in "${title}".`,
  ({ title, focus }) => `What details does "${title}" provide about ${focus}?`,
  ({ title, focus }) => `Give a concise summary of ${focus} in "${title}".`,
  ({ title, focus, keywords }) => `Within the paper ${keywords}, how is ${focus} presented?`,
];

const JUDGE_SINGLE_FOCI = [
  'the central problem',
  "the paper's contributions",
  'the technical approach',
  'the retrieval strategy',
  'the reranking strategy',
  'the evaluation setup',
  'the evidence used to support the claims',
  'the main experimental findings',
  'the limitations',
  'the practical implications',
  'the key takeaways',
  'the future directions',
] as const;

const JUDGE_SINGLE_PROMPTS: readonly JudgePromptBuilder[] = [
  ({ title, focus }) => `Synthesize ${focus} for "${title}" in a grounded summary.`,
  ({ title, focus }) => `Write a concise research synthesis of ${focus} in "${title}".`,
  ({ title, focus }) => `Using the uploaded document, explain ${focus} for "${title}".`,
  ({ title, focus }) => `Summarize ${focus} from "${title}" with the strongest supporting evidence.`,
  ({ title, focus, keywords }) => `For ${keywords}, provide a synthesis focused on ${focus}.`,
  ({ title, focus }) => `What is the document-grounded synthesis of ${focus} in "${title}"?`,
  ({ title, focus }) => `Produce a short literature-style synthesis of ${focus} in "${title}".`,
  ({ title, focus }) => `Explain ${focus} in "${title}" and keep the answer evidence-grounded.`,
  ({ title, focus }) => `From "${title}", synthesize ${focus} and cite the strongest evidence.`,
  ({ title, focus }) => `Using only "${title}", give an evidence-grounded explanation of ${focus}.`,
];

const JUDGE_MULTI_FOCI = [
  'their main contributions',
  'their methodological differences',
  'their evaluation choices',
  'their reported results',
  'their limitations',
  'their practical implications',
] as const;

const JUDGE_MULTI_PROMPTS: readonly JudgeMultiPromptBuilder[] = [
  (titles, focus) => `Compare ${focus} across ${titles}.`,
  (titles, focus) => `Provide a grounded synthesis of ${focus} for ${titles}.`,
  (titles, focus) => `What similarities and differences appear in ${focus} across ${titles}?`,
  (titles, focus) => `Write a comparative synthesis of ${focus} using ${titles}.`,
];

function cleanTitle(raw: string | undefined): string {
  const normalized = (raw || 'Untitled document')
    .replace(/\.[a-z0-9]{2,4}$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return normalized || 'Untitled document';
}

function titleKeywords(title: string): string {
  const words = cleanTitle(title)
    .split(/\s+/)
    .map((word) => word.replace(/[^a-z0-9]+/gi, ''))
    .filter((word) => word.length >= 3 && !TITLE_STOPWORDS.has(word.toLowerCase()));
  const trimmed = (words.length ? words : cleanTitle(title).split(/\s+/)).slice(0, 6);
  return trimmed.join(' ');
}

function dedupeByTitle(docs: DocumentRow[]): DocumentRow[] {
  const seen = new Set<string>();
  const out: DocumentRow[] = [];
  for (const doc of docs) {
    const key = cleanTitle(doc.title).toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(doc);
  }
  return out;
}

export function selectPresetDocuments(docs: DocumentRow[]): DocumentRow[] {
  const ready = dedupeByTitle(
    docs.filter((doc) => (doc.status || '').toLowerCase() === 'ready')
  );
  const researchOnly = ready.filter((doc) => (doc.doc_type || '').toLowerCase() === 'research_paper');
  return (researchOnly.length ? researchOnly : ready).slice(0, 12);
}

export function buildRetrievalPresetCases(docs: DocumentRow[], count = DEFAULT_EVAL_PRESET_COUNT): EvalCase[] {
  const selected = selectPresetDocuments(docs);
  if (!selected.length || count <= 0) return [];

  const cases: EvalCase[] = [];
  outer: for (const focus of RETRIEVAL_FOCI) {
    for (const promptBuilder of RETRIEVAL_PROMPTS) {
      for (const doc of selected) {
        cases.push({
          query: promptBuilder({
            title: cleanTitle(doc.title),
            keywords: titleKeywords(doc.title),
            focus,
          }),
          expected_doc_id: doc.id,
        });
        if (cases.length >= count) break outer;
      }
    }
  }

  return cases.slice(0, count);
}

function buildSingleJudgeCases(docs: DocumentRow[]): JudgeCasePayload[] {
  const cases: JudgeCasePayload[] = [];
  for (const focus of JUDGE_SINGLE_FOCI) {
    for (const promptBuilder of JUDGE_SINGLE_PROMPTS) {
      for (const doc of docs) {
        cases.push({
          query: promptBuilder({
            title: cleanTitle(doc.title),
            keywords: titleKeywords(doc.title),
            focus,
          }),
          doc_id: doc.id,
        });
      }
    }
  }
  return cases;
}

function buildMultiJudgeCases(docs: DocumentRow[]): JudgeCasePayload[] {
  if (docs.length < 2) return [];

  const pairs: DocumentRow[][] = [];
  for (let i = 0; i < docs.length; i += 1) {
    for (let j = i + 1; j < docs.length; j += 1) {
      pairs.push([docs[i], docs[j]]);
    }
  }

  const cases: JudgeCasePayload[] = [];
  for (const focus of JUDGE_MULTI_FOCI) {
    for (const promptBuilder of JUDGE_MULTI_PROMPTS) {
      for (const pair of pairs) {
        const titles = pair.map((doc) => `"${cleanTitle(doc.title)}"`).join(' and ');
        cases.push({
          query: promptBuilder(titles, focus),
          doc_ids: pair.map((doc) => doc.id),
        });
      }
    }
  }
  return cases;
}

export function buildJudgePresetCases(docs: DocumentRow[], count = DEFAULT_EVAL_PRESET_COUNT): JudgeCasePayload[] {
  const selected = selectPresetDocuments(docs);
  if (!selected.length || count <= 0) return [];

  const singles = buildSingleJudgeCases(selected);
  const multis = buildMultiJudgeCases(selected);

  if (!multis.length) return singles.slice(0, count);

  const cases: JudgeCasePayload[] = [];
  let singleIdx = 0;
  let multiIdx = 0;

  while (cases.length < count && (singleIdx < singles.length || multiIdx < multis.length)) {
    if (singleIdx < singles.length) {
      cases.push(singles[singleIdx]);
      singleIdx += 1;
      if (cases.length >= count) break;
    }
    if (singleIdx < singles.length) {
      cases.push(singles[singleIdx]);
      singleIdx += 1;
      if (cases.length >= count) break;
    }
    if (multiIdx < multis.length) {
      cases.push(multis[multiIdx]);
      multiIdx += 1;
    }
  }

  return cases.slice(0, count);
}
