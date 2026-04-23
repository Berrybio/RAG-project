# Clinical Trial RAG — Breast Cancer Trial Planner

A retrieval-augmented assistant for breast cancer clinical trial design. Clinicians chat with a planner that retrieves relevant trials from ClinicalTrials.gov, answers design questions with grounded citations, and — when the conversation is ready — drafts a full study protocol as a downloadable DOCX or PDF.

## What it does

- **Grounded Q&A** over a curated slice of ClinicalTrials.gov (breast cancer, recruiting). Answers cite the trials they used.
- **Inline clarifying choices**: when the planner needs more info (e.g. "interventional vs. observational?"), it offers multi-choice chips in the chat rather than free-text prompts.
- **Conversation → protocol**: one click summarizes the chat into a planning brief, then another generates a structured protocol (objectives, eligibility, endpoints, statistics, safety, etc.) exported as DOCX or PDF.
- **Phase-aware conventions**: single-center default for Phase I/II, multi-center for Phase III and real-world evidence studies. Admin fields (Protocol ID, Sponsor, Location, Contact) are left as placeholders — never hallucinated.

## Stack

| Layer     | Tech                                                                |
| --------- | ------------------------------------------------------------------- |
| Backend   | FastAPI, Anthropic (Claude) streaming via SSE, Voyage AI embeddings |
| Retrieval | Dense embeddings (`voyage-3`, 1024-dim) with on-disk cache          |
| Docs      | python-docx (DOCX) + reportlab/platypus (PDF)                       |
| Frontend  | Vanilla JS + CSS, SSE over fetch                                    |
| Data      | `Breast_Cancer-RECRUITING-phase2-625.csv` (ClinicalTrials.gov slice)|
| Infra     | Docker Compose (backend :8000, frontend :80)                        |

## API surface

```
GET  /api/health
POST /api/search            # one-shot retrieval
GET  /api/search/stream     # SSE answer stream
POST /api/chat/stream       # planner turn (SSE)
POST /api/chat/summarize    # conversation → planning brief
POST /api/protocol/json     # structured protocol JSON
POST /api/protocol/docx     # render protocol as Word
POST /api/protocol/pdf      # render protocol as PDF
```

## Quickstart

```bash
# 1. Create .env in repo root with:
#    ANTHROPIC_API_KEY=...
#    VOYAGE_API_KEY=...
#    ANTHROPIC_MODEL=claude-...        # optional
#    CSV_PATH=/app/data/Breast_Cancer-RECRUITING-phase2-625.csv

# 2. Put the CSV in ./data/ (or adjust CSV_PATH).

# 3. Bring it up.
docker compose up --build
# Frontend:  http://localhost
# Backend:   http://localhost:8000/api/health
```

### Local dev (without Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Serve frontend/ however you like (e.g. `python -m http.server 8080`),
# then point it at the backend by editing the API base in frontend/app.js.
```

## Repo layout

```
backend/
  app/
    api/        # FastAPI routers: health, search, chat, protocol
    core/       # pipeline, chat prompts, protocol generation (DOCX + PDF)
    models/     # pydantic schemas
frontend/       # index.html, app.js, styles.css, nginx.conf
data/           # CSV(s) mounted read-only into the backend container
```

## Tests

```bash
cd backend && pytest
```

## Notes for collaborators

- Trial setting (interventional vs. observational, phase, line) is inferred only from `title`, `conditions`, and `keywords` — never from the study description or inclusion criteria, which describe patient history rather than the trial itself.
- The protocol generator treats RWE studies as non-phased: no "Phase II" language leaks into observational protocols.
- Embedding cache lives next to the CSV; delete it if you change the embedding model or the underlying data.
