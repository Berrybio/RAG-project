import logging
from collections.abc import AsyncGenerator

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a clinical trials assistant specializing in breast cancer research.

CRITICAL GROUNDING RULES — violation of these rules is a failure:
- You will be given a <trials> section containing the ONLY trial data you may use.
- NEVER mention any NCT ID that is not explicitly listed in <trials>.
- NEVER invent, recall, or infer trial details from your training data.
- If a field is missing from a trial, say "not specified" — do NOT guess.
- If the provided trials are insufficient to fully answer, explicitly state: \
"Only [N] relevant trial(s) were found in the database for this query."
- Every fact in your answer must map to a specific field in <trials>.

When answering:
1. Summarize each relevant trial concisely: NCT ID, title, phase, status, \
intervention, key eligibility, primary outcome, and sponsor.
2. Highlight differences between trials when multiple are returned.
3. If eligibility, location, or outcome data is missing for a trial, say \
"not specified" rather than guessing.
4. Keep responses precise and structured. Do not pad with generic information."""


def format_context(retrieved_docs: list[dict]) -> str:
    """Format retrieved documents into a context string for the prompt."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc["metadata"]

        # Build a structured summary from metadata
        lines = [
            f"<trial id=\"{i}\">",
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
            ("Eligibility Criteria", "eligibilityCriteria"),
            ("FDA Regulated Drug", "isFdaRegulatedDrug"),
        ]
        for label, key in field_map:
            val = meta.get(key, "")
            if val and str(val) not in ("", "0", "False"):
                lines.append(f"{label}: {val}")

        lines.append("</trial>")
        context_parts.append("\n".join(lines))

    return "\n\n".join(context_parts)


def _build_user_message(query: str, retrieved_docs: list[dict]) -> str:
    context = format_context(retrieved_docs)
    n = len(retrieved_docs)
    nct_ids = [doc["metadata"].get("nctId", "unknown") for doc in retrieved_docs]
    id_list = ", ".join(nct_ids)
    return (
        f"Exactly {n} trial(s) were retrieved. The ONLY NCT IDs you may reference "
        f"are: {id_list}\n\n"
        f"<trials>\n{context}\n</trials>\n\n"
        f"Question: {query}\n\n"
        f"Answer using ONLY the {n} trial(s) above. Do NOT mention any NCT ID, "
        f"drug, or study not present in <trials>. If {n} trials are not enough "
        f"to fully answer the question, state that explicitly."
    )


def _assistant_prefill(retrieved_docs: list[dict]) -> str:
    """Anchor the assistant's response with the exact trial IDs available."""
    n = len(retrieved_docs)
    nct_ids = [doc["metadata"].get("nctId", "unknown") for doc in retrieved_docs]
    id_list = ", ".join(nct_ids)
    return (
        f"Based on the {n} retrieved trial(s) ({id_list}), "
        f"here is what I found:"
    )


async def generate_answer(
    client: anthropic.AsyncAnthropic,
    query: str,
    retrieved_docs: list[dict],
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Send the query + retrieved context to Claude and return the full answer."""
    prefill = _assistant_prefill(retrieved_docs)
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _build_user_message(query, retrieved_docs)},
            {"role": "assistant", "content": prefill},
        ],
    )
    return prefill + response.content[0].text


async def generate_answer_stream(
    client: anthropic.AsyncAnthropic,
    query: str,
    retrieved_docs: list[dict],
    model: str = "claude-sonnet-4-20250514",
) -> AsyncGenerator[str, None]:
    """Stream answer tokens as an async generator for SSE."""
    prefill = _assistant_prefill(retrieved_docs)
    yield prefill
    async with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _build_user_message(query, retrieved_docs)},
            {"role": "assistant", "content": prefill},
        ],
    ) as stream:
        async for text in stream.text_stream:
            yield text
