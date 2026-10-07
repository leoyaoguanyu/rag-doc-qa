from src.prompts import RAG_PROMPT


def test_prompt_has_required_placeholders():
    assert "{context}" in RAG_PROMPT.template
    assert "{question}" in RAG_PROMPT.template


def test_prompt_has_dont_know_guardrail():
    assert "don't know" in RAG_PROMPT.template.lower()


def test_prompt_renders_context_and_question():
    rendered = RAG_PROMPT.format(context="CTX", question="Q?")
    assert "CTX" in rendered and "Q?" in rendered
    assert "{context}" not in rendered
