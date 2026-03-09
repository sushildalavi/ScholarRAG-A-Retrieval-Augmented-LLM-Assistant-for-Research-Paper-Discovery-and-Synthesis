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
  confidence_percent?: number;
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
  evidence_id?: string;
  msa?: {
    M: number;
    S: number;
    A: number;
    msa_score: number;
    score_percent: number;
    weights?: {
      w1?: number;
      w2?: number;
      w3?: number;
      b?: number;
    };
  };
  msa_supported?: boolean;
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
    scope_penalty?: number;
    msa?: {
      M: number;
      S: number;
      A: number;
      msa_score: number;
      weights?: {
        w1?: number;
        w2?: number;
        w3?: number;
        b?: number;
      };
    };
  };
  explanation?: string;
};

export type FaithfulnessReport = {
  overall_score: number;
  citation_coverage: number;
  supported_count: number;
  unsupported_count: number;
  sentence_count: number;
  claims: Array<{
    sentence_id: number;
    sentence: string;
    supported: boolean;
    evidence_ids: string[];
    reason: string;
  }>;
  unsupported: Array<{
    sentence_id: number;
    sentence: string;
    supported: boolean;
    reason: string;
    evidence_ids?: string[];
  }>;
  method: string;
  evidence_coverage_by_id?: Record<string, boolean>;
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
  faithfulness?: FaithfulnessReport | null;
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

export type AskAssistantPayload = {
  query: string;
  scope: 'uploaded' | 'public';
  doc_id?: number;
  doc_ids?: number[];
  k?: number;
  sense?: string;
  compare_senses?: boolean;
  allow_general_background?: boolean;
  run_judge?: boolean;
  run_judge_llm?: boolean;
};

export type JudgeCasePayload = {
  query: string;
  answer?: string;
  citations?: Citation[];
  doc_id?: number;
  scope?: 'uploaded' | 'public';
  allow_general_background?: boolean;
};

export type JudgeRunPayload = {
  scope?: 'uploaded' | 'public';
  k?: number;
  run_judge_llm?: boolean;
  cases: JudgeCasePayload[];
};

export type JudgeRunResponse = {
  run_id?: number;
  created_at?: string;
  scope: 'uploaded' | 'public';
  query_count: number;
  metrics: {
    mean_overall_score: number;
    mean_coverage: number;
    unsupported_total: number;
    count: number;
  };
  details: Array<{
    query: string;
    answer: string;
    citations: Citation[];
    faithfulness: FaithfulnessReport;
    scope: 'uploaded' | 'public';
  }>;
};

export type JudgeRunSummary = {
  id: number;
  scope: 'uploaded' | 'public';
  query_count: number;
  metrics: {
    mean_overall_score?: number;
    mean_coverage?: number;
    unsupported_total?: number;
    count?: number;
  };
  created_at: string | null;
};

export type MsaCalibrationPayload = {
  records: Array<{
    sentence?: string;
    evidence?: string;
    evidence_text?: string;
    evidence_snippet?: string;
    S?: number;
    A?: number;
    M?: number;
    label?: string;
    answer_supported?: boolean;
    msa?: {
      M: number;
      S: number;
      A: number;
    };
  }>;
  model_name?: string;
  label?: string;
};

export type MsaCalibrationResponse = {
  run_id?: number;
  created_at?: string;
  model_name: string;
  label: string;
  records_used: number;
  weights: {
    w1: number;
    w2: number;
    w3: number;
    b: number;
  };
  metrics: {
    n: number;
    accuracy: number;
    brier: number;
    method: string;
  };
};

export type MsaCalibrationLatest = {
  model_name: string;
  label: string;
  weights: {
    w1: number;
    w2: number;
    w3: number;
    b: number;
  };
  metrics: unknown;
  dataset_size: number;
  created_at: string | null;
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
