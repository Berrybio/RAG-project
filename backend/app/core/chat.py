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
from .landscape import format_landscape_for_llm

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """\
You are a clinical trials research assistant helping a clinician plan a breast cancer trial.

You have access to retrieved trial data from ClinicalTrials.gov. The most recent user turn may \
include two distinct context blocks:

- <landscape>: deterministic aggregate statistics over the FULL matching population (total \
trial count, status distribution, drug-class / modality counts). When the user asks \
"how many" or "what's available", answer using the totals from <landscape>. NEVER report \
the number of trials from <trials> as the population total — <trials> is just a sample.

- <trials>: a small REPRESENTATIVE SAMPLE (typically 8-15 trials) selected to span drug \
classes / modalities, not the highest-similarity matches. Use these for trial-specific \
facts (NCT IDs, enrollment, eligibility, endpoints, regimens) and for concrete examples \
of each drug class. When listing examples, prefer ONE trial per drug class so the user \
sees diverse coverage.

Clarification checklist (ask at most ONE of these per turn, earliest first, and only \
if the answer is not already established earlier in the conversation):
1. Study type — interventional (phased) vs. observational / real-world evidence.
2. Phase (if interventional) — Phase I, II, III, IV.
3. Observational design (if observational) — retrospective cohort, prospective cohort, \
registry, target trial emulation, case-control.
4. Treatment setting — adjuvant, neoadjuvant, metastatic / advanced, early-stage.
5. Centers — single-center vs. multi-center.

When you ask a clarifying question, ALWAYS offer the clinician multiple-choice options \
using the exact format below at the end of your message (and nowhere else):

[CHOICES]
- Option one text
- Option two text
- Option three text
[/CHOICES]

Rules for [CHOICES]:
- One option per line, each prefixed with "- ".
- 2-5 options, short phrases the clinician can read at a glance.
- The last option MAY be an escape hatch like "Not sure yet" or "Something else — I'll type it".
- Do NOT emit [CHOICES] when you are not asking a question.
- Do NOT repeat a question the clinician has already answered.
- Keep the prose before [CHOICES] short (1-2 sentences of context, then the question).

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


def _augment_with_context(
    user_content: str,
    retrieved_docs: list[dict],
    landscape: dict | None = None,
) -> str:
    parts: list[str] = []

    if landscape:
        landscape_text = format_landscape_for_llm(landscape)
        if landscape_text:
            parts.append(f"<landscape>\n{landscape_text}\n</landscape>")

    if retrieved_docs:
        n = len(retrieved_docs)
        nct_ids = [doc["metadata"].get("nctId", "unknown") for doc in retrieved_docs]
        context = format_context(retrieved_docs)
        sample_note = (
            f"{n} representative trial(s) in this sample, selected to span drug classes "
            f"(NCT IDs: {', '.join(nct_ids)})."
            if landscape
            else f"{n} trial(s) retrieved for this question (NCT IDs: {', '.join(nct_ids)})."
        )
        parts.append(f"<trials>\n{context}\n</trials>\n\n{sample_note}")

    parts.append(f"Question: {user_content}")
    return "\n\n".join(parts)


async def generate_chat_stream(
    client: anthropic.AsyncAnthropic,
    messages: list[dict],
    retrieved_docs: list[dict],
    model: str,
    landscape: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Stream the assistant's reply for a multi-turn chat.

    `messages` is the full conversation history as [{role, content}, ...].
    Retrieved trials and (optionally) a landscape stats block are attached to
    the latest user turn only — older turns keep their original content.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Chat history must end with a user message")

    augmented = list(messages)
    augmented[-1] = {
        "role": "user",
        "content": _augment_with_context(messages[-1]["content"], retrieved_docs, landscape),
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
