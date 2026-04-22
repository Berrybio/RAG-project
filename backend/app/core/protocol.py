import json
import logging
from io import BytesIO

import anthropic
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from .generation import format_context

logger = logging.getLogger(__name__)

PROTOCOL_SYSTEM_PROMPT = """\
You are an expert clinical research physician who designs clinical trial protocols.
Given a set of reference trials retrieved from ClinicalTrials.gov, draft a NEW
Phase II breast cancer clinical trial protocol.

Return your answer as a single JSON object (no markdown fences) with exactly
these keys:

{
  "title": "Short descriptive trial title",
  "official_title": "Full formal title including phase and design",
  "protocol_id": "",
  "sponsor": "",
  "phase": "Phase II",
  "status": "PLANNED",
  "conditions": "Specific breast cancer subtype(s)",
  "summary": "2-3 sentence plain-language summary",
  "description": "Detailed scientific background and rationale (2-3 paragraphs covering unmet need, mechanism of action, preclinical/Phase I evidence)",
  "primary_objectives": ["objective 1", "..."],
  "secondary_objectives": ["objective 1", "..."],
  "exploratory_objectives": ["objective 1", "..."],
  "study_design": "Randomized, Open-label, ...",
  "study_schema": "Description of study arms, randomization ratio, stratification factors, and overall study flow",
  "intervention_name": "Drug(s) / therapy name(s)",
  "intervention_description": "Detailed dosing, schedule, route of administration, dose modifications, and treatment arms",
  "comparator": "Comparator arm description (or 'Single-arm, no comparator')",
  "treatment_duration": "Duration of treatment and follow-up periods",
  "inclusion_criteria": ["criterion 1", "criterion 2", "..."],
  "exclusion_criteria": ["criterion 1", "criterion 2", "..."],
  "primary_endpoints": ["endpoint 1 with definition and assessment timepoint", "..."],
  "secondary_endpoints": ["endpoint 1 with definition and assessment timepoint", "..."],
  "estimated_enrollment": 100,
  "sample_size_justification": "Statistical rationale for sample size including assumptions, power, alpha, expected effect size",
  "statistical_analysis": "Primary analysis method, interim analyses, multiplicity adjustments, populations (ITT, per-protocol)",
  "safety_monitoring": "Description of Data Safety Monitoring Board (DSMB), stopping rules, dose-limiting toxicity definitions",
  "adverse_event_reporting": "AE/SAE grading criteria (e.g., CTCAE v5.0), reporting timelines, and expedited reporting requirements",
  "dose_modification": "Dose reduction levels, dose delay criteria, and discontinuation rules",
  "study_assessments": "Schedule of assessments: screening, on-treatment, end-of-treatment, and follow-up visit procedures (imaging, labs, PROs)",
  "study_schedule_table": [
    {"visit": "Screening", "timepoint": "Day -28 to -1", "procedures": "Informed consent, medical history, physical exam, labs, imaging, ECG, tumor biopsy"},
    {"visit": "Cycle 1 Day 1", "timepoint": "Day 1", "procedures": "Treatment administration, vitals, AE assessment"},
    {"visit": "Every 2 Cycles", "timepoint": "Every 8 weeks", "procedures": "Tumor assessment per RECIST 1.1, labs, AE assessment"},
    {"visit": "End of Treatment", "timepoint": "Within 30 days of last dose", "procedures": "Physical exam, labs, imaging, AE assessment"},
    {"visit": "Follow-up", "timepoint": "Every 3 months for 2 years", "procedures": "Survival status, subsequent therapies"}
  ],
  "ethical_considerations": "IRB/IEC approval requirements, Declaration of Helsinki compliance, informed consent process, vulnerable populations protections",
  "data_management": "EDC system, data quality assurance, source data verification, database lock procedures",
  "regulatory_considerations": "IND/CTA requirements, regulatory authority notifications, GCP compliance",
  "sex": "ALL or FEMALE",
  "minimum_age": "18 Years",
  "locations": [],
  "contact_info": "",
  "references": ["Key reference 1 (Author et al., Journal, Year)", "..."]
}

Base the protocol on patterns you see in the reference trials but make it a
coherent NEW study. Be specific, realistic, and scientifically rigorous.
Include specific drug names, dosing regimens, and measurable endpoints.
Follow ICH-GCP E6(R2) and FDA/EMA guidance for Phase II oncology trials.

IMPORTANT — administrative fields the user will fill in themselves:
- "protocol_id": return an empty string "". Do NOT invent a protocol ID.
- "sponsor": return an empty string "". Do NOT invent a sponsor.
- "locations": return an empty list []. Do NOT invent sites, cities, or countries.
- "contact_info": return an empty string "". Do NOT invent a PI or institution.
All other scientific/design fields should be fully drafted.

STUDY CENTER CONVENTION:
- Phase I and Phase II trials are typically SINGLE-CENTER (one academic/clinical
  site). Default to single-center for these unless the user explicitly asks for
  multi-center.
- Phase III trials and real-world evidence (RWE) studies are typically
  MULTI-CENTER.
Reflect this in "study_design" and "study_schema" (e.g., "Single-center,
randomized, open-label Phase II trial..."). The "locations" array still stays
empty — the user will supply the site."""


async def generate_protocol_json(
    client: anthropic.AsyncAnthropic,
    query: str,
    retrieved_docs: list[dict],
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Ask Claude to draft a protocol based on RAG-retrieved trials; return parsed JSON dict."""
    context = format_context(retrieved_docs)

    user_msg = (
        f"The user wants to plan a clinical trial with this focus:\n"
        f'"{query}"\n\n'
        f"=== REFERENCE TRIALS FROM DATABASE ===\n{context}\n"
        f"=== END ===\n\n"
        f"Draft a new Phase II breast cancer trial protocol as JSON."
    )

    # Use streaming for long generations (max_tokens=8192). Non-streaming
    # requests with high max_tokens can exceed the Anthropic API gateway
    # timeout and fail with APIConnectionError. Streaming keeps the
    # connection warm token-by-token; we accumulate the full response.
    chunks: list[str] = []
    async with client.messages.stream(
        model=model,
        max_tokens=8192,
        system=PROTOCOL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        async for text in stream.text_stream:
            chunks.append(text)

    raw = "".join(chunks).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    protocol = json.loads(raw)
    logger.info("Protocol JSON generated with %d keys", len(protocol))
    return protocol


def build_protocol_docx(protocol: dict) -> BytesIO:
    """Build a Word document from a protocol JSON dict. Returns a BytesIO buffer."""
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Title page
    title = doc.add_heading(protocol.get("title", "Clinical Trial Protocol"), level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if protocol.get("official_title"):
        sub = doc.add_paragraph(protocol["official_title"])
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    meta_fields = [
        ("Protocol ID", "protocol_id"),
        ("Sponsor", "sponsor"),
        ("Phase", "phase"),
        ("Status", "status"),
        ("Conditions", "conditions"),
    ]
    for label, key in meta_fields:
        if protocol.get(key):
            doc.add_paragraph(f"{label}: {protocol[key]}")

    doc.add_page_break()

    # Summary & Description
    _add_section(doc, "Summary", protocol.get("summary"))
    _add_section(doc, "Description", protocol.get("description"))

    # Objectives
    _add_list_section(doc, "Primary Objectives", protocol.get("primary_objectives"))
    _add_list_section(doc, "Secondary Objectives", protocol.get("secondary_objectives"))
    _add_list_section(doc, "Exploratory Objectives", protocol.get("exploratory_objectives"))

    # Study Design
    _add_section(doc, "Study Design", protocol.get("study_design"))
    _add_section(doc, "Study Schema", protocol.get("study_schema"))

    # Intervention
    _add_section(doc, "Intervention", protocol.get("intervention_name"))
    _add_section(doc, "Intervention Description", protocol.get("intervention_description"))
    _add_section(doc, "Comparator", protocol.get("comparator"))
    _add_section(doc, "Treatment Duration", protocol.get("treatment_duration"))

    # Eligibility
    _add_list_section(doc, "Inclusion Criteria", protocol.get("inclusion_criteria"))
    _add_list_section(doc, "Exclusion Criteria", protocol.get("exclusion_criteria"))

    # Endpoints
    _add_list_section(doc, "Primary Endpoints", protocol.get("primary_endpoints"))
    _add_list_section(doc, "Secondary Endpoints", protocol.get("secondary_endpoints"))

    # Statistics
    enrollment = protocol.get("estimated_enrollment")
    if enrollment:
        _add_section(doc, "Estimated Enrollment", str(enrollment))
    _add_section(doc, "Sample Size Justification", protocol.get("sample_size_justification"))
    _add_section(doc, "Statistical Analysis", protocol.get("statistical_analysis"))

    # Safety
    _add_section(doc, "Safety Monitoring", protocol.get("safety_monitoring"))
    _add_section(doc, "Adverse Event Reporting", protocol.get("adverse_event_reporting"))
    _add_section(doc, "Dose Modification", protocol.get("dose_modification"))

    # Assessments
    _add_section(doc, "Study Assessments", protocol.get("study_assessments"))

    # Schedule table
    schedule = protocol.get("study_schedule_table")
    if schedule and isinstance(schedule, list):
        doc.add_heading("Study Schedule", level=1)
        table = doc.add_table(rows=1, cols=3)
        table.style = "Light Grid Accent 1"
        for i, header in enumerate(["Visit", "Timepoint", "Procedures"]):
            table.rows[0].cells[i].text = header
        for row_data in schedule:
            row = table.add_row()
            row.cells[0].text = row_data.get("visit", "")
            row.cells[1].text = row_data.get("timepoint", "")
            row.cells[2].text = row_data.get("procedures", "")

    # Regulatory & Ethics
    _add_section(doc, "Ethical Considerations", protocol.get("ethical_considerations"))
    _add_section(doc, "Data Management", protocol.get("data_management"))
    _add_section(doc, "Regulatory Considerations", protocol.get("regulatory_considerations"))

    # Demographics
    if protocol.get("sex"):
        _add_section(doc, "Eligible Sex", protocol["sex"])
    if protocol.get("minimum_age"):
        _add_section(doc, "Minimum Age", protocol["minimum_age"])

    # Locations
    _add_list_section(doc, "Locations", protocol.get("locations"))

    # Contact
    _add_section(doc, "Contact Information", protocol.get("contact_info"))

    # References
    _add_list_section(doc, "References", protocol.get("references"))

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def _add_section(doc: Document, title: str, content: str | None):
    if not content:
        return
    doc.add_heading(title, level=1)
    doc.add_paragraph(content)


def _add_list_section(doc: Document, title: str, items: list | None):
    if not items:
        return
    doc.add_heading(title, level=1)
    for item in items:
        doc.add_paragraph(str(item), style="List Bullet")
