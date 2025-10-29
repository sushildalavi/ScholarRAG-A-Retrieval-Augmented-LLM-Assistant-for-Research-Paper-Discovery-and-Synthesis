SYSTEM_PROMPT = (
    "You are ScholarRAG, an academic assistant. Answer concisely and cite sources inline using [Surname, YEAR]. "
    "After the answer, include a short bullet list explaining why each source is relevant."
)


def build_user_prompt(query: str, contexts: str) -> str:
    return (
        f"Question: {query}\n\n"
        f"You are given numbered source snippets below. Use them to answer.\n"
        f"Snippets:\n{contexts}\n\n"
        "Provide a concise answer with inline citations like [Surname, YEAR]. Then give a short bullet list mapping numbers to sources and why relevant."
    )

