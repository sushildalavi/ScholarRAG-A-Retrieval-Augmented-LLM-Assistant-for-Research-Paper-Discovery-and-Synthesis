from typing import Dict, List, Tuple

from openai import OpenAI

from utils.config import get_openai_api_key
from utils.citation_utils import short_citation, make_snippet
from utils.prompts import SYSTEM_PROMPT, build_user_prompt


def _client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def build_contexts(docs: List[Dict]) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    keys: List[str] = []
    for i, d in enumerate(docs, start=1):
        key = short_citation(d)
        text = d.get("abstract") or d.get("summary") or d.get("title") or ""
        blocks.append(f"[{i}] {key}: {make_snippet(text)}")
        keys.append(key)
    return "\n".join(blocks), keys


def synthesize_answer(query: str, docs: List[Dict]) -> str:
    contexts, _ = build_contexts(docs)
    prompt = build_user_prompt(query, contexts)
    resp = _client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content

