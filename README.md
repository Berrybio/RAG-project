# Clinical Trial RAG — Breast Cancer Trial Planner

A retrieval-augmented assistant for breast cancer clinical trial design. Clinicians chat with a planner that retrieves relevant trials from ClinicalTrials.gov, answers design questions with grounded citations, and — when the conversation is ready — drafts a full study protocol as a downloadable DOCX or PDF.

## What it does

- **Grounded Q&A** over 16,000+ recruiting breast cancer trials from ClinicalTrials.gov. Answers cite the trials they used.
- **Inline clarifying choices**: when the planner needs more info (e.g. "interventional vs. observational?"), it offers multi-choice chips in the chat rather than free-text prompts.
- **Conversation → protocol**: one click summarizes the chat into a planning brief, then another generates a structured protocol (objectives, eligibility, endpoints, statistics, safety, etc.) exported as DOCX or PDF.
- **Iterative protocol refinement**: refine a generated protocol through follow-up chat. The planner discusses feedback first and applies changes only on explicit confirmation.
- **Protocol history**: generated protocols are persisted per-user. Resume, reload, or compare versions across sessions.
- **Landscape brief**: aggregate statistics, history-aware filters, sponsor display, and drug-class diversification for competitive landscape analysis.
- **Feedback loop**: up/down-vote answers to improve retrieval quality. Down-vote-aware reranker and few-shot examples from up-voted answers feed back into future responses.
- **Phase-aware conventions**: single-center default for Phase I/II, multi-center for Phase III and real-world evidence studies. Admin fields (Protocol ID, Sponsor, Location, Contact) are left as placeholders — never hallucinated.
- **Pluggable LLM providers**: Anthropic (default), OpenAI, DeepSeek, and Kimi. Non-Anthropic providers are experimental.

## Stack

| Layer     | Tech                                                                          |
| --------- | ----------------------------------------------------------------------------- |
| Backend   | FastAPI, pluggable LLM providers (Anthropic/OpenAI/DeepSeek/Kimi), SSE       |
| Retrieval | Voyage AI dense embeddings (`voyage-3`, 1024-dim) with on-disk cache, TF-IDF fallback |
| Docs      | python-docx (DOCX) + reportlab/platypus (PDF)                                |
| Frontend  | Vanilla JS + CSS, SSE over fetch, nginx reverse proxy                        |
| Data      | 16,000+ breast cancer trials (ClinicalTrials.gov, 55-column schema)          |
| Infra     | Docker Compose (local), Google Cloud Run (production)                        |

## API surface

### Health

```
GET  /api/health                        # readiness check (trial count, model)
```

### Search

```
POST /api/search                        # one-shot retrieval (JSON)
GET  /api/search/stream                 # SSE answer stream
```

### Chat

```
POST /api/chat/stream                   # planner turn (SSE streaming)
POST /api/chat/summarize                # conversation → planning brief
```

### Protocol

```
POST /api/protocol/json                 # generate structured protocol JSON
POST /api/protocol/refine               # iteratively refine a protocol
POST /api/protocol/docx                 # export protocol as Word document
POST /api/protocol/pdf                  # export protocol as PDF
GET  /api/protocols                     # list user's saved protocols
GET  /api/protocols/{protocol_id}       # load a specific protocol version
```

### Feedback

```
POST /api/feedback                      # submit feedback (thumbs up/down)
GET  /api/feedback                      # list all feedback entries
POST /api/feedback/{feedback_id}/status # update feedback status (reviewed/dismissed)
POST /api/feedback/promote              # promote feedback to drug alias
GET  /api/aliases                       # get drug alias dictionary
```

## Quickstart

```bash
# 1. Create .env in repo root (see .env.example for all options):
#    ANTHROPIC_API_KEY=...
#    VOYAGE_API_KEY=...          # omit to fall back to TF-IDF
#    LLM_MODEL=claude-sonnet-4-20250514

# 2. Place the trial CSV in backend/data/ (or adjust CSV_PATH).

# 3. Bring it up.
docker compose up --build
# Frontend:  http://localhost
# Backend:   http://localhost:8000/api/health
```

### Cloud Run deployment

The backend and frontend each have a Dockerfile ready for Cloud Run. The backend bakes the trial CSV and precomputed embeddings cache into the image; the frontend's nginx proxies `/api/` requests to the backend.

```bash
# Backend
gcloud run deploy berrybio-backend \
  --source=backend \
  --region=us-central1 \
  --port=8000 \
  --memory=2Gi \
  --allow-unauthenticated \
  --set-env-vars="ANTHROPIC_API_KEY=...,VOYAGE_API_KEY=..."

# Frontend (set BACKEND_URL to the backend's Cloud Run URL)
gcloud run deploy berrybio-frontend \
  --source=frontend \
  --region=us-central1 \
  --port=80 \
  --allow-unauthenticated \
  --set-env-vars="BACKEND_URL=https://berrybio-backend-XXXXXX.us-central1.run.app"
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
    api/            # FastAPI routers: health, search, chat, protocol, feedback
    core/           # pipeline, chat prompts, protocol generation, feedback, landscape
      llm/          # pluggable LLM providers (Anthropic, OpenAI, DeepSeek, Kimi)
    models/         # pydantic schemas
  data/             # trial CSV, embeddings cache, feedback log, drug aliases
frontend/           # index.html, app.js, styles.css, nginx.conf, Dockerfile
reports/            # weekly report generator (analytics)
docker-compose.yml
.env.example
```

## Tests

```bash
cd backend && pytest
```

## Notes for collaborators

- Trial setting (interventional vs. observational, phase, line) is inferred only from `title`, `conditions`, and `keywords` — never from the study description or inclusion criteria, which describe patient history rather than the trial itself.
- The protocol generator treats RWE studies as non-phased: no "Phase II" language leaks into observational protocols.
- Embedding cache lives next to the CSV; delete it if you change the embedding model or the underlying data.
- Non-Anthropic LLM providers are wired up but unvalidated — protocol JSON parse rate, the chat planner's markup, and grounding-rule adherence may regress. Leave `LLM_PROVIDER=anthropic` unless running an explicit A/B evaluation.
