# prompts.py — all prompt templates in one place

from config import KNOWN_ABBR

# Build the abbreviation reference block once
_ABBR_BLOCK = "\n".join(f"  {k} = {v}" for k, v in KNOWN_ABBR.items())


def build_rag_prompt(question: str, context: str, history: str = "") -> str:
    history_section = f"Previous conversation:\n{history}\n\n" if history.strip() else ""

    return f"""You are a policy document assistant for NIT Calicut.
Answer using ONLY the context below. Be concise and factual.
If the answer is not in the context, say: Not found in documents.
Do not repeat the question. Do not add outside knowledge.

{history_section}Context:
{context}

Question: {question}
Answer:"""


def build_query_expansion_prompt(question: str) -> str:
    """
    Generates 2 semantically equivalent rephrasings for multi-query retrieval.
    Kept minimal so the small model follows the format reliably.
    """
    return f"""Rephrase this question 2 ways for document search. Output only the 2 rephrased questions, one per line, no numbering.

Known abbreviations: {', '.join(f'{k}={v}' for k, v in KNOWN_ABBR.items())}

Question: {question}"""
