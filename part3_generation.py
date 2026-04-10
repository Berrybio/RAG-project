import anthropic


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


def generate_answer(
    client: anthropic.Anthropic,
    query: str,
    retrieved_docs: list[dict],
    model: str = "claude-opus-4-6",
) -> str:
    """
    Send the query + retrieved context to Claude and get an answer.
    Uses streaming for responsive output.
    """
    context = format_context(retrieved_docs)

    system_prompt = """\
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

    user_message = f"""\
Based on the following clinical trial data, please answer the user's question.

=== RETRIEVED CLINICAL TRIALS ===
{context}
=== END OF DATA ===

User's Question: {query}

Please provide a comprehensive answer based on the trial data above."""

    collected_text = []
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            collected_text.append(text)

    print()
    return "".join(collected_text)

