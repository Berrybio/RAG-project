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

from .feedback import aliases_prompt_block
from .feedback_examples import FeedbackExampleStore, format_examples_block
from .generation import format_context
from .landscape import format_landscape_for_llm
from .llm import BaseLLMProvider

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """\
You are a clinical trials research assistant helping a clinician plan a breast cancer trial.

You have access to retrieved trial data from ClinicalTrials.gov. The most recent user turn may \
include three distinct context blocks:

- <previous_protocols>: when present, the user has prior protocols stored and is asking to \
work on one. The block lists each protocol by title, version, phase, and population. Your \
reply MUST acknowledge the listed protocol(s) by name — never say "I don't see any \
previous protocol" or "no protocol history" when this block is present. Tell the user \
their protocols are loaded as clickable chips below your reply and that clicking one \
opens it for refinement. Do NOT regenerate the protocol contents inline; the chip is the \
loading mechanism. Keep this acknowledgement brief (2-3 sentences) and DO NOT emit \
[CHOICES] — the chips already serve as the picker.

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
- Keep answers focused; avoid boilerplate.

Protocol / report requests — REDIRECT, do not draft inline:
If the clinician asks you to generate, write, draft, produce, or create a "protocol", \
"report", "study protocol", "trial protocol", "planning report", or similar full document \
(including phrases like "I've learned enough, write the protocol"), DO NOT write protocol \
prose in the chat. Instead, reply with a short message (2-3 sentences) telling them to \
click the **"Summarize & plan report"** button below the chat. Briefly explain that the \
button runs a structured pipeline that produces a downloadable Word document with audit-ready \
sections (objectives, eligibility, endpoints, schedule), with admin fields left blank for \
them to fill in — whereas an inline chat draft would be free-form prose without that \
structure. Do not emit [CHOICES] for this redirect. Continue answering follow-up questions \
about specific trials or design choices in chat as usual; only redirect when they ask for \
the full protocol/report itself.

Trial-list formatting (MANDATORY whenever you list two or more trials from <trials>):

Render EXACTLY in this shape, with no variation. Each trial uses one labeled \
field per line:

```
1. **NCT12345678 — Short trial name (acronym if any)**
   Phase: Phase III
   Status: Recruiting
   Number of patients: n=500
   Sponsor: Sponsor Name
   Arms: <experimental arm> vs <control / comparator arm>
   - Optional one-line distinguishing detail (≤20 words).
   - Optional second sub-bullet (≤20 words).

2. **NCT...**
   ...
```

Hard rules — these are the most common mistakes, do NOT make them:
- ALWAYS a single numbered list (1., 2., 3., …), ordered as the trials appear in \
<trials>. NEVER regroup into sections like "Completed trials" / "Active trials" / \
"Landmark trials". One list, one ordering, no headings between trials.
- Each labeled metadata line appears on its OWN line, in this exact order: \
`Phase:`, `Status:`, `Number of patients:`, `Sponsor:`, `Arms:`. Do not collapse \
multiple fields onto one line and do not reorder. Use "—" when a field is missing \
from <trials>; never invent a value.
- `Number of patients:` value is `n=<count>` where <count> is the Enrollment field \
from <trials> (patient count, not number of arms or sites).
- `Sponsor:` value comes from the Sponsor field in <trials>.
- `Arms:` is `<experimental> vs <comparator>` pulled from the Arm Groups field. \
If the trial is single-arm or has no comparator, write `Single-arm — <intervention>`. \
If the comparator is unclear, write `<experimental> vs —`. Never invent a control arm.
- Keep prose before the list to one short sentence (e.g. the population total from \
<landscape>). After the list, at most one short closing sentence."""


SUMMARIZE_SYSTEM_PROMPT = """\
You are a clinical trials research assistant. Summarize the conversation between a clinician \
and the assistant into a short planning brief (3-5 sentences) that captures:
- The trial concept under discussion (population, intervention, setting).
- Key design decisions, open questions, or constraints the clinician raised.
- Any reference trials or patterns that came up.

Write in plain prose, not bullet points. Do not invent details. Finish by asking: \
"Would you like me to generate a detailed protocol report based on this summary?"
"""


def _format_previous_protocols_block(entries: list[dict]) -> str:
    """Render a hint block about previously generated protocols for the LLM.

    The actual protocol JSON isn't inlined — too many tokens, and the user
    will pick one via the load chip the frontend renders alongside this
    reply. The LLM just needs to know the protocols *exist* and what their
    titles are, so it doesn't gaslight the user with "I don't see any."
    """
    if not entries:
        return ""
    lines = [
        f"The user has {len(entries)} previously generated protocol"
        f"{'s' if len(entries) != 1 else ''} stored. "
        "The frontend is rendering load chips alongside your reply so the user "
        "can click to open one for refinement.",
    ]
    for i, e in enumerate(entries, 1):
        title = e.get("title") or "Untitled"
        phase = e.get("phase") or ""
        conds = e.get("conditions") or ""
        bits = [f"v{e.get('version_count', 1)}"]
        if phase:
            bits.append(phase)
        if conds:
            bits.append(conds[:80])
        lines.append(f"{i}. {title} ({', '.join(bits)})")
    lines.append(
        "Your reply: acknowledge the available protocol(s) by name, briefly say "
        "what you'll help refine, and tell the user to click the matching chip "
        "below to load it. Do NOT claim the protocol does not exist or that "
        "there is no protocol history — they are listed above. Do NOT regenerate "
        "the protocol contents inline."
    )
    return "\n".join(lines)


def _augment_with_context(
    user_content: str,
    retrieved_docs: list[dict],
    landscape: dict | None = None,
    previous_protocols: list[dict] | None = None,
) -> str:
    parts: list[str] = []

    if previous_protocols:
        # Place the previous-protocols hint FIRST — when the user explicitly
        # asked about a prior protocol, that intent should anchor the reply,
        # not the trial list (which the planner will retrieve anyway).
        block = _format_previous_protocols_block(previous_protocols)
        if block:
            parts.append(f"<previous_protocols>\n{block}\n</previous_protocols>")

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
    llm: BaseLLMProvider,
    messages: list[dict],
    retrieved_docs: list[dict],
    landscape: dict | None = None,
    aliases: dict[str, str] | None = None,
    examples: FeedbackExampleStore | None = None,
    previous_protocols: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream the assistant's reply for a multi-turn chat.

    `messages` is the full conversation history as [{role, content}, ...].
    Retrieved trials and (optionally) a landscape stats block are attached to
    the latest user turn only — older turns keep their original content.
    `aliases` (when provided) is appended to the system prompt as a "Known
    drug aliases" block so the model uses canonical names in its prose.
    `examples` is the up-voted few-shot store; the most similar past
    (question, answer) pair is appended to the system prompt to anchor the
    reply structure on what the user has already endorsed.
    `previous_protocols` (when provided) is the list of stored protocols
    matched to the user's NL hint ("based on the previous protocol", etc.).
    Their meta is injected into the user message as a hint block so the LLM
    knows to acknowledge them rather than gaslight the user with "no
    protocol exists."
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Chat history must end with a user message")

    augmented = list(messages)
    augmented[-1] = {
        "role": "user",
        "content": _augment_with_context(
            messages[-1]["content"],
            retrieved_docs,
            landscape,
            previous_protocols,
        ),
    }

    system_prompt = CHAT_SYSTEM_PROMPT
    aliases_block = aliases_prompt_block(aliases or {})
    if aliases_block:
        system_prompt = f"{system_prompt}\n\n{aliases_block}"
    if examples is not None:
        examples_block = format_examples_block(examples.pick(messages[-1]["content"]))
        if examples_block:
            system_prompt = f"{system_prompt}\n\n{examples_block}"

    async for text in llm.stream(
        system=system_prompt,
        messages=augmented,
        max_tokens=4096,
    ):
        yield text


async def summarize_conversation(
    llm: BaseLLMProvider,
    messages: list[dict],
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

    return await llm.complete(
        system=SUMMARIZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Conversation transcript:\n\n{transcript}"}],
        max_tokens=800,
    )
