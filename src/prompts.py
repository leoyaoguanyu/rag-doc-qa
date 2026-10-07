"""RAG prompt template. Prompt engineering is explicit here on purpose:
the template carries the grounding instruction and the "don't know"
guardrail, which is what keeps a RAG system from hallucinating when
retrieval comes back empty."""

from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using ONLY the context below.\n"
    "If the answer is not in the context, say \"I don't know based on the provided documents.\"\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Answer:"
)
