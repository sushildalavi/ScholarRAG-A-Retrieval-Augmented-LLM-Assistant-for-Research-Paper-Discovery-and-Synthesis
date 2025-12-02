export type DocumentRow = {
  id: number;
  title: string;
  status: string;
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
  title?: string;
  authors?: string;
  year?: number;
  source?: string;
  url?: string;
  doc_id?: number;
  chunk_id?: number;
  page?: number;
};

export type AnswerResponse = {
  answer: string;
  citations: Citation[];
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
