import logging
from collections.abc import AsyncGenerator

from .llm import BaseLLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical trials assistant specializing in {cancer_type} research.

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


def _get_system_prompt(cancer_type_display: str = "breast cancer") -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(cancer_type=cancer_type_display.lower())


async def generate_answer(
    llm: BaseLLMProvider,
    query: str,
    retrieved_docs: list[dict],
    cancer_type_display: str = "breast cancer",
) -> str:
    """Send the query + retrieved context to the LLM and return the full answer.

    Note: assistant prefill (a forced opening like "Looking at the retrieved
    trials...") is an Anthropic-specific pattern and isn't part of the
    OpenAI-compat shape, so we drop it from the messages array and instead
    prepend it to the returned text. Other providers will naturally pick up
    the same opening style from the system prompt.
    """
    system_prompt = _get_system_prompt(cancer_type_display)
    prefill = _assistant_prefill(retrieved_docs)
    body = await llm.complete(
        system=system_prompt,
        messages=[
            {"role": "user", "content": _build_user_message(query, retrieved_docs)},
        ],
        max_tokens=4096,
    )
    return prefill + body


async def generate_answer_stream(
    llm: BaseLLMProvider,
    query: str,
    retrieved_docs: list[dict],
    cancer_type_display: str = "breast cancer",
) -> AsyncGenerator[str, None]:
    """Stream answer tokens as an async generator for SSE."""
    system_prompt = _get_system_prompt(cancer_type_display)
    prefill = _assistant_prefill(retrieved_docs)
    yield prefill
    async for text in llm.stream(
        system=system_prompt,
        messages=[
            {"role": "user", "content": _build_user_message(query, retrieved_docs)},
        ],
        max_tokens=4096,
    ):
        yield text
