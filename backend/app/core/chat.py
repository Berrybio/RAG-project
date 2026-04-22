"""Multi-turn conversational generation over retrieved trials.

The chat flow is designed for a clinician iteratively planning a trial:
question → grounded answer → follow-up → etc. Each user turn triggers
retrieval, and the retrieved trials are attached to *that* turn's user
message as <trials>...</trials>. Older turns keep their original content
so conversation history stays compact.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import anthropic

from .generation import format_context

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """\
You are a clinical trials research assistant helping a clinician plan a breast cancer trial.

You have access to retrieved trial data from ClinicalTrials.gov. The most recent user turn \
includes a <trials> section with the trials retrieved for that question. Rely on those trials \
for trial-specific facts (NCT IDs, enrollment, eligibility, endpoints, etc).

Behaviour:
- Be concise and clinically precise; use oncology terminology fluently.
- When citing a specific trial, give its NCT ID from <trials>. Do NOT invent NCT IDs.
- When the user asks for design guidance (sample size, eligibility patterns, endpoints), \
synthesize across the provided trials and mark which conclusions come from the trials vs. \
your own reasoning.
- If the retrieved trials are insufficient, say so and suggest what to search for next.
- Keep answers focused; avoid boilerplate."""


SUMMARIZE_SYSTEM_PROMPT = """\
You are a clinical trials research assistant. Summarize the conversation between a clinician \
and the assistant into a short planning brief (3-5 sentences) that captures:
- The trial concept under discussion (population, intervention, setting).
- Key design decisions, open questions, or constraints the clinician raised.
- Any reference trials or patterns that came up.

Write in plain prose, not bullet points. Do not invent details. Finish by asking: \
"Would you like me to generate a detailed protocol report based on this summary?"
"""


def _augment_with_context(user_content: str, retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return user_content
    n = len(retrieved_docs)
    nct_ids = [doc["metadata"].get("nctId", "unknown") for doc in retrieved_docs]
    context = format_context(retrieved_docs)
    return (
        f"<trials>\n{context}\n</trials>\n\n"
        f"{n} trial(s) retrieved for this question "
        f"(NCT IDs: {', '.join(nct_ids)}).\n\n"
        f"Question: {user_content}"
    )


async def generate_chat_stream(
    client: anthropic.AsyncAnthropic,
    messages: list[dict],
    retrieved_docs: list[dict],
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream the assistant's reply for a multi-turn chat.

    `messages` is the full conversation history as [{role, content}, ...].
    Retrieved trials are attached to the latest user turn only.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Chat history must end with a user message")

    augmented = list(messages)
    augmented[-1] = {
        "role": "user",
        "content": _augment_with_context(messages[-1]["content"], retrieved_docs),
    }

    async with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=CHAT_SYSTEM_PROMPT,
        messages=augmented,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def summarize_conversation(
    client: anthropic.AsyncAnthropic,
    messages: list[dict],
    model: str,
) -> str:
    """Produce a short planning-brief summary of the conversation."""
    if not messages:
        return "No conversation to summarize yet."
    # Render the conversation as a single user message for summarization.
    transcript_lines = []
    for m in messages:
        speaker = "Clinician" if m["role"] == "user" else "Assistant"
        transcript_lines.append(f"{speaker}: {m['content'].strip()}")
    transcript = "\n\n".join(transcript_lines)

    response = await client.messages.create(
        model=model,
        max_tokens=800,
        system=SUMMARIZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Conversation transcript:\n\n{transcript}"}],
    )
    return response.content[0].text
