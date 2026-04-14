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
6. Mention the trial phase, status, location, sponsor, and PI when helpful.
7. When discussing interventions, include drug aliases if available.
8. Reference primary outcomes and study design details when relevant."""


def format_context(retrieved_docs: list[dict]) -> str:
    """Format retrieved documents into a context string for the prompt."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc["metadata"]

        # Build a structured summary from metadata
        lines = [
            f"--- Trial {i} (Relevance: {doc['score']:.3f}) ---",
            f"NCT ID: {meta.get('nctId', '')}",
            f"Title: {meta.get('title', '')}",
        ]
        # Only include non-empty fields
        field_map = [
            ("Official Title", "officialTitle"),
            ("Acronym", "acronym"),
            ("Phase", "phases"),
            ("Status", "status"),
            ("Study Type", "studyType"),
            ("Conditions", "conditions"),
            ("Keywords", "keywords"),
            ("MeSH Terms", "meshTermsCondition"),
            ("Allocation", "allocation"),
            ("Intervention Model", "interventionModel"),
            ("Primary Purpose", "primaryPurpose"),
            ("Masking", "masking"),
            ("Intervention", "interventionName"),
            ("Intervention Type", "interventionType"),
            ("Drug Aliases", "interventionOtherNames"),
            ("Arm Groups", "armGroups"),
            ("Enrollment", "enrollmentCont"),
            ("Enrollment Type", "enrollmentType"),
            ("Eligible Sex", "sex"),
            ("Minimum Age", "minimumAge"),
            ("Maximum Age", "maximumAge"),
            ("Primary Outcomes", "primaryOutcomes"),
            ("Secondary Outcomes", "secondaryOutcomes"),
            ("Sponsor", "sponsorName"),
            ("Sponsor Class", "sponsorClass"),
            ("PI", "piName"),
            ("PI Affiliation", "piAffiliation"),
            ("Start Date", "startDate"),
            ("Completion Date", "completionETA"),
            ("Last Updated", "lastUpdateDate"),
            ("Location Countries", "locationCountries"),
            ("Number of Sites", "locationCount"),
            ("Locations", "locationInfo"),
            ("Contact", "contactInfo"),
            ("FDA Regulated Drug", "isFdaRegulatedDrug"),
        ]
        for label, key in field_map:
            val = meta.get(key, "")
            if val and str(val) not in ("", "0", "False"):
                lines.append(f"{label}: {val}")

        lines.append(f"\nFull Details:\n{doc['text'][:3000]}")
        context_parts.append("\n".join(lines))

    return "\n\n".join(context_parts)


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
