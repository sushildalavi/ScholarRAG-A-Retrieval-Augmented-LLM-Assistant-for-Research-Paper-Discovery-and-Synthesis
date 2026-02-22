export type DocumentRow = {
  id: number;
  title: string;
  status: string;
  doc_type?: 'resume' | 'research_paper' | 'official_doc' | 'assignment' | 'notes' | 'other' | string;
  pages?: number;
  bytes?: number;
  created_at?: string;
};

export type ChunkResult = {
  id: number;
  document_id: number;
  text: string;
  page_no?: number;
  chunk_index?: number;
  distance?: number;
};

export type Citation = {
  id?: number;
  title?: string;
  scope?: 'personal_profile' | 'course_material' | 'uploaded_document' | 'public_reference' | string;
  authors?: string;
  year?: number;
  source?: string;
  url?: string;
  doc_id?: number;
  chunk_id?: number;
  page?: number;
  distance?: number;
  similarity?: number;
  confidence?: number;
  used_in_answer?: boolean;
  rank_before?: number;
  rank_after?: number;
  rank_delta?: number;
  rerank_score?: number;
  rerank_raw?: number;
  rerank_norm?: number;
  reranker_type?: string;
  sim_score?: number;
  sim_raw?: number;
  confidence_obj?: ConfidenceObject;
};

export type ConfidenceObject = {
  score: number;
  label: 'High' | 'Med' | 'Low' | 'Context-limited' | string;
  needs_clarification?: boolean;
  factors: {
    top_sim: number;
    top_rerank_norm: number;
    citation_coverage: number;
    evidence_margin: number;
    ambiguity_penalty: number;
    insufficiency_penalty: number;
  };
  explanation?: string;
};

export type WhyTraceChunk = {
  id?: number;
  title?: string;
  doc_id?: number;
  chunk_id?: number;
  page?: number;
  snippet_preview?: string;
  sim_score?: number;
  sim_raw?: number;
  rerank_raw?: number;
  rerank_norm?: number;
  reranker_type?: string;
  rank_before?: number;
  rank_after?: number;
  rank_delta?: number;
  cited?: boolean;
  source?: string;
  scope?: string;
};

export type AnswerResponse = {
  answer: string;
  citations: Citation[];
  confidence?: ConfidenceObject;
  why_answer?: {
    rerank_changed_order: boolean;
    top_chunks: WhyTraceChunk[];
  };
  needs_clarification?: boolean;
  clarification?: {
    question: string;
    options: string[];
    recommended_option?: string;
    rationale?: string;
    term?: string;
  } | null;
  answer_scope?: string;
  unsupported_claims?: number;
  scoring?: {
    similarity_metric: string;
    reranker_used: boolean;
    reranker_type: string;
    rerank_score_fields: string[];
  };
  latency_breakdown_ms?: {
    retrieve: number;
    rerank: number;
    generate: number;
    total: number;
  };
};

export type ChatMessage = {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  created_at?: string;
};

export type ChatSession = {
  session_id: number;
  messages: ChatMessage[];
};

export type EvalCase = {
  query: string;
  expected_doc_id?: number;
  expected_passage?: string;
};

export type EvalRunResponse = {
  run_id?: number;
  created_at?: string;
  name: string;
  scope: string;
  k: number;
  case_count: number;
  metrics_retrieval_only: {
    count: number;
    recall_at: Record<string, number>;
    mrr: number;
    ndcg_at: Record<string, number>;
  };
  metrics_retrieval_rerank: {
    count: number;
    recall_at: Record<string, number>;
    mrr: number;
    ndcg_at: Record<string, number>;
  };
  latency_breakdown: {
    retrieve_ms_avg: number;
    rerank_ms_avg: number;
    generate_ms_avg: number;
  };
  details: Array<{
    query: string;
    gold_doc_id?: number;
    retrieval_only_top: any[];
    rerank_top: any[];
    latency_ms: { retrieve_ms: number; rerank_ms: number };
  }>;
};
