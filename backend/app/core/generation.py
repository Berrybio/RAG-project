import logging
from collections.abc import AsyncGenerator

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a knowledgeable clinical trials assistant specializing in breast cancer \
research. You help researchers and patients find relevant clinical trials and \
understand trial details.

When answering questions:
1. Base your answers ONLY on the provided clinical trial data.
2. Reference specific trial NCT IDs when mentioning trials.
3. If the retrieved data doesn't contain enough information to answer, say so clearly.
4. Provide structured, easy-to-read responses.
5. Highlight important eligibility criteria when relevant.
6. Mention the trial phase, status, and location when helpful."""


def format_context(retrieved_docs: list[dict]) -> str:
    """Format retrieved documents into a context string for the prompt."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc["metadata"]
        context_parts.append(
            f"--- Trial {i} (Relevance: {doc['score']:.3f}) ---\n"
            f"NCT ID: {meta['nctId']}\n"
            f"Title: {meta['title']}\n"
            f"Phase: {meta['phases']}\n"
            f"Status: {meta['status']}\n"
            f"Conditions: {meta['conditions']}\n"
            f"Intervention: {meta['interventionName']}\n"
            f"Enrollment: {meta['enrollmentCont']}\n"
            f"Eligible Sex: {meta['sex']}\n"
            f"Minimum Age: {meta['minimumAge']}\n"
            f"Location: {meta['locationInfo']}\n"
            f"Contact: {meta['contactInfo']}\n"
            f"\nFull Details:\n{doc['text'][:2000]}\n"
        )
    return "\n".join(context_parts)


def _build_user_message(query: str, retrieved_docs: list[dict]) -> str:
    context = format_context(retrieved_docs)
    return (
        f"Based on the following clinical trial data, please answer the user's question.\n\n"
        f"=== RETRIEVED CLINICAL TRIALS ===\n{context}\n=== END OF DATA ===\n\n"
        f"User's Question: {query}\n\n"
        f"Please provide a comprehensive answer based on the trial data above."
    )


async def generate_answer(
    client: anthropic.AsyncAnthropic,
    query: str,
    retrieved_docs: list[dict],
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Send the query + retrieved context to Claude and return the full answer."""
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(query, retrieved_docs)}],
    )
    return response.content[0].text


async def generate_answer_stream(
    client: anthropic.AsyncAnthropic,
    query: str,
    retrieved_docs: list[dict],
    model: str = "claude-sonnet-4-20250514",
) -> AsyncGenerator[str, None]:
    """Stream answer tokens as an async generator for SSE."""
    async with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(query, retrieved_docs)}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
