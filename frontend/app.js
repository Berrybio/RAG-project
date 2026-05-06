const API_BASE = "/api";

// --- Anonymous identity for analytics ---
// user_id persists across sessions in localStorage (per-browser, per-device).
// session_id is fresh on every page load (sessionStorage clears on tab close).
// Both ride along as request headers; the backend logs them with each event,
// which lets BigQuery answer DAU / cohort retention / funnel questions later.
// No PII collected — these are random UUIDs, not tied to any real identity.
const ANALYTICS_USER_ID = (() => {
  let id = localStorage.getItem("berrybio_user_id");
  if (!id) {
    id = (crypto.randomUUID && crypto.randomUUID()) ||
         Date.now().toString(36) + Math.random().toString(36).slice(2);
    localStorage.setItem("berrybio_user_id", id);
  }
  return id;
})();

const ANALYTICS_SESSION_ID = (() => {
  let id = sessionStorage.getItem("berrybio_session_id");
  if (!id) {
    id = (crypto.randomUUID && crypto.randomUUID()) ||
         Date.now().toString(36) + Math.random().toString(36).slice(2);
    sessionStorage.setItem("berrybio_session_id", id);
  }
  return id;
})();

// Centralized fetch wrapper so every API call carries the analytics headers
// without each call site having to remember. Merges with caller-supplied
// headers (e.g. Content-Type) instead of overwriting.
async function apiFetch(path, opts = {}) {
  const baseHeaders = {
    "X-User-Id": ANALYTICS_USER_ID,
    "X-Session-Id": ANALYTICS_SESSION_ID,
  };
  return fetch(path, {
    ...opts,
    headers: { ...baseHeaders, ...(opts.headers || {}) },
  });
}

// --- Source card rendering ---
function renderSourceCard(source) {
  return `
    <div class="source-card">
      <div class="title">${escapeHtml(source.title)}</div>
      <div class="nct-id">
        <a href="https://clinicaltrials.gov/study/${encodeURIComponent(source.nct_id)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.nct_id)}</a>
        <span class="score">${(source.score * 100).toFixed(1)}%</span>
        <a class="source-link" href="https://clinicaltrials.gov/study/${encodeURIComponent(source.nct_id)}" target="_blank" rel="noopener noreferrer">View on ClinicalTrials.gov &#8599;</a>
      </div>
      <div class="meta">
        <span>Phase: ${escapeHtml(source.phases)}</span>
        <span>Status: ${escapeHtml(source.status)}</span>
        <span>Sponsor: ${escapeHtml(source.sponsor || "N/A")}</span>
        <span>Intervention: ${escapeHtml(source.intervention || "N/A")}</span>
        <span>Enrollment: ${escapeHtml(String(source.enrollment))}</span>
      </div>
    </div>
  `;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML;
}

// Build a span whose children alternate between plain text nodes and <a>
// elements wrapping NCT IDs (e.g. "NCT06841354"). Using DOM nodes rather than
// innerHTML keeps the LLM-emitted text safe from injection — only the matched
// NCT pattern is ever placed inside an anchor's href.
function linkifyNctIds(text) {
  const span = document.createElement("span");
  const re = /\bNCT\d{8}\b/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      span.appendChild(document.createTextNode(text.slice(last, m.index)));
    }
    const a = document.createElement("a");
    a.href = `https://clinicaltrials.gov/study/${m[0]}`;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = m[0];
    span.appendChild(a);
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    span.appendChild(document.createTextNode(text.slice(last)));
  }
  return span;
}

// --- Protocol preview rendering ---
function renderProtocolPreview(p, targetId = "protocol-preview") {
  const preview = document.getElementById(targetId);
  let html = "";

  html += `<h2>${escapeHtml(p.title)}</h2>`;
  if (p.official_title) html += `<p><em>${escapeHtml(p.official_title)}</em></p>`;

  // Protocol ID and Sponsor always show, even when empty — render a visible
  // placeholder so the clinician knows to fill them in.
  const blank = '<span class="protocol-placeholder">________________</span>';
  const metaFields = [
    ["Protocol ID", p.protocol_id, true],
    ["Sponsor", p.sponsor, true],
    ["Phase", p.phase, false],
    ["Conditions", p.conditions, false],
  ];
  html += '<div class="meta" style="margin:12px 0">';
  metaFields.forEach(([label, val, always]) => {
    if (val) html += `<span><strong>${label}:</strong> ${escapeHtml(val)}</span>  `;
    else if (always) html += `<span><strong>${label}:</strong> ${blank}</span>  `;
  });
  html += "</div>";

  if (p.summary) html += `<h3>Summary</h3><p>${escapeHtml(p.summary)}</p>`;
  if (p.description) html += `<h3>Description</h3><p>${escapeHtml(p.description)}</p>`;

  html += renderList("Primary Objectives", p.primary_objectives);
  html += renderList("Secondary Objectives", p.secondary_objectives);

  if (p.study_design) html += `<h3>Study Design</h3><p>${escapeHtml(p.study_design)}</p>`;
  if (p.intervention_name)
    html += `<h3>Intervention</h3><p><strong>${escapeHtml(p.intervention_name)}</strong></p>`;
  if (p.intervention_description) html += `<p>${escapeHtml(p.intervention_description)}</p>`;

  html += renderList("Inclusion Criteria", p.inclusion_criteria);
  html += renderList("Exclusion Criteria", p.exclusion_criteria);
  html += renderList("Primary Endpoints", p.primary_endpoints);
  html += renderList("Secondary Endpoints", p.secondary_endpoints);

  if (p.estimated_enrollment)
    html += `<h3>Enrollment</h3><p>${p.estimated_enrollment} patients</p>`;
  if (p.sample_size_justification) html += `<p>${escapeHtml(p.sample_size_justification)}</p>`;

  // Schedule table
  if (p.study_schedule_table && p.study_schedule_table.length > 0) {
    html += "<h3>Study Schedule</h3>";
    html += '<table><tr><th>Visit</th><th>Timepoint</th><th>Procedures</th></tr>';
    p.study_schedule_table.forEach((row) => {
      html += `<tr><td>${escapeHtml(row.visit)}</td><td>${escapeHtml(row.timepoint)}</td><td>${escapeHtml(row.procedures)}</td></tr>`;
    });
    html += "</table>";
  }

  if (p.safety_monitoring)
    html += `<h3>Safety Monitoring</h3><p>${escapeHtml(p.safety_monitoring)}</p>`;

  // Locations always render with a placeholder when empty.
  if (p.locations && p.locations.length > 0) {
    html += renderList("Locations", p.locations);
  } else {
    html += `<h3>Locations</h3>`;
    html += `<p class="protocol-placeholder">________________________________________</p>`;
    html += `<p class="protocol-placeholder"><em>(to be completed: site name, city, state, country)</em></p>`;
  }

  // Contact Information always renders with a placeholder block when empty.
  html += `<h3>Contact Information</h3>`;
  if (p.contact_info) {
    html += `<p>${escapeHtml(p.contact_info)}</p>`;
  } else {
    html += `<p class="protocol-placeholder"><strong>Principal Investigator:</strong> ________________</p>`;
    html += `<p class="protocol-placeholder"><strong>Institution:</strong> ________________</p>`;
    html += `<p class="protocol-placeholder"><strong>Email:</strong> ________________</p>`;
    html += `<p class="protocol-placeholder"><strong>Phone:</strong> ________________</p>`;
  }

  html += renderList("References", p.references);

  preview.innerHTML = html;
}

function renderList(title, items) {
  if (!items || items.length === 0) return "";
  let html = `<h3>${escapeHtml(title)}</h3><ul>`;
  items.forEach((item) => {
    html += `<li>${escapeHtml(String(item))}</li>`;
  });
  html += "</ul>";
  return html;
}

// ===================== PLANNER (multi-turn chat) =====================

const plannerState = {
  messages: [],            // [{role:'user'|'assistant', content:string}]
  lastSummary: "",         // last summarization output, used as protocol query
  currentProtocol: null,
  // Version tracking for the generated protocol. v1 is the initial generation;
  // each refinement appends a new version. activeVersion is the index in
  // protocolVersions whose protocol is currently shown in the preview and
  // used for downloads.
  protocolVersions: [],    // [{label, protocol, userRequest, assistantNote, changedFields, timestamp}]
  activeVersion: -1,
  refineMessages: [],      // [{role:'user'|'assistant', content:string}] for /protocol/refine
};

const plannerEls = () => ({
  messagesBox: document.getElementById("planner-messages"),
  input: document.getElementById("planner-input"),
  sendBtn: document.getElementById("planner-send-btn"),
  summarizeBtn: document.getElementById("planner-summarize-btn"),
  resetBtn: document.getElementById("planner-reset-btn"),
  topk: document.getElementById("planner-topk"),
  topkVal: document.getElementById("planner-topk-val"),
  showLandscape: document.getElementById("planner-show-landscape"),
  summaryBox: document.getElementById("planner-summary-box"),
  summaryContent: document.getElementById("planner-summary-content"),
  genProtocolBtn: document.getElementById("planner-generate-protocol-btn"),
  skipProtocolBtn: document.getElementById("planner-skip-protocol-btn"),
  protocolLoading: document.getElementById("planner-protocol-loading"),
  protocolResult: document.getElementById("planner-protocol-result"),
  protocolPreview: document.getElementById("planner-protocol-preview"),
  downloadBtn: document.getElementById("planner-download-docx-btn"),
  downloadPdfBtn: document.getElementById("planner-download-pdf-btn"),
  versionList: document.getElementById("planner-version-list"),
  compareBtn: document.getElementById("planner-compare-btn"),
  refineThread: document.getElementById("planner-refine-thread"),
  refineInput: document.getElementById("planner-refine-input"),
  refineSendBtn: document.getElementById("planner-refine-send-btn"),
  diffModal: document.getElementById("planner-diff-modal"),
  diffFrom: document.getElementById("planner-diff-from"),
  diffTo: document.getElementById("planner-diff-to"),
  diffBody: document.getElementById("planner-diff-body"),
  diffCloseBtn: document.getElementById("planner-diff-close-btn"),
});

function addChatMessage(role, content) {
  const el = plannerEls().messagesBox;
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.textContent = content;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return div;
}

// Map raw enum-style filter values to clinician-friendly labels.
const LANDSCAPE_PHASE_LABEL = {
  EARLY_PHASE1: "Early Phase 1",
  PHASE1: "Phase 1",
  PHASE2: "Phase 2",
  PHASE3: "Phase 3",
  PHASE4: "Phase 4",
};
const LANDSCAPE_STATUS_LABEL = {
  RECRUITING: "Recruiting",
  NOT_YET_RECRUITING: "Not yet recruiting",
  ACTIVE_NOT_RECRUITING: "Active, not recruiting",
  COMPLETED: "Completed",
  TERMINATED: "Terminated",
  WITHDRAWN: "Withdrawn",
  SUSPENDED: "Suspended",
  UNKNOWN: "Unknown",
  ENROLLING_BY_INVITATION: "Enrolling by invitation",
};

function landscapePhaseLabel(p) {
  return LANDSCAPE_PHASE_LABEL[p] || p || "Unspecified";
}
function landscapeStatusLabel(s) {
  return LANDSCAPE_STATUS_LABEL[s] || s || "Unspecified";
}

function landscapeBar(label, count, total) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return `
    <div class="landscape-row">
      <div class="landscape-row-label">${escapeHtml(label)}</div>
      <div class="landscape-row-bar"><div class="landscape-row-fill" style="width:${pct}%"></div></div>
      <div class="landscape-row-count">${count}<span class="landscape-row-pct"> · ${pct}%</span></div>
    </div>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderLandscapeCard(L) {
  const div = document.createElement("div");
  div.className = "landscape-card";
  const filters = L.filters_applied || {};
  const filterChips = [];
  if (filters.phase) filterChips.push(landscapePhaseLabel(filters.phase));
  if (filters.status) filterChips.push(landscapeStatusLabel(filters.status));
  const filterLine = filterChips.length
    ? `<span class="landscape-filter-chips">${filterChips
        .map((c) => `<span class="landscape-filter-chip">${escapeHtml(c)}</span>`)
        .join("")}</span>`
    : `<span class="landscape-filter-chips landscape-filter-chips-empty">no filters — showing whole population</span>`;

  // If filtered out everything, render an explanatory empty-state instead of empty bars.
  if (!L.total) {
    div.innerHTML = `
      <div class="landscape-header">
        <div class="landscape-title">Landscape · ${escapeHtml(L.population)}</div>
        ${filterLine}
      </div>
      <div class="landscape-empty">
        No trials match these filters strictly.
        ${L.disease_total ? `(${L.disease_total} ${escapeHtml(L.population)} trials in the corpus overall.)` : ""}
      </div>`;
    return div;
  }

  const phases = (L.phase_distribution || []).map((r) =>
    landscapeBar(landscapePhaseLabel(r.phase), r.count, L.total)
  ).join("");
  const statuses = (L.status_distribution || []).slice(0, 8).map((r) =>
    landscapeBar(landscapeStatusLabel(r.status), r.count, L.total)
  ).join("");
  const drugClasses = (L.drug_classes || []).slice(0, 12).map((r) =>
    landscapeBar(r.class, r.count, L.total)
  ).join("");
  const geo = (L.geography_top || []).slice(0, 8).map((r) =>
    landscapeBar(r.country, r.count, L.total)
  ).join("");
  const sponsors = (L.sponsor_class || []).map((r) =>
    landscapeBar(r.class, r.count, L.total)
  ).join("");

  const totalNote =
    L.disease_total && L.disease_total !== L.total
      ? ` <span class="landscape-total-note">of ${L.disease_total} total in corpus</span>`
      : "";

  div.innerHTML = `
    <div class="landscape-header">
      <div class="landscape-title">Landscape · ${escapeHtml(L.population)}</div>
      ${filterLine}
    </div>
    <div class="landscape-total"><strong>${L.total}</strong> matching trials${totalNote}</div>
    <div class="landscape-grid">
      <section>
        <h4>Phase</h4>
        ${phases || '<div class="landscape-empty">none</div>'}
      </section>
      <section>
        <h4>Status</h4>
        ${statuses || '<div class="landscape-empty">none</div>'}
      </section>
      <section class="landscape-grid-wide">
        <h4>Drug class / modality</h4>
        ${drugClasses || '<div class="landscape-empty">no recognized drug classes</div>'}
      </section>
      <section>
        <h4>Top countries</h4>
        ${geo || '<div class="landscape-empty">none</div>'}
      </section>
      <section>
        <h4>Sponsor class</h4>
        ${sponsors || '<div class="landscape-empty">none</div>'}
      </section>
    </div>`;
  return div;
}

// ===================== FEEDBACK (thumbs + suggest-correction) =====================

function showToast(message, kind) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast" + (kind === "error" ? " error" : "");
  toast.textContent = message;
  container.appendChild(toast);
  // Trigger transition.
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 250);
  }, 2400);
}

async function postFeedback(payload) {
  const resp = await apiFetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
  return await resp.json();
}

// Snapshot the latest user query + this assistant reply so admins reviewing
// the feedback queue can see what was being reacted to without correlating
// timestamps across logs. The source NCT ids ride along too — the backend
// reranker uses them to learn from 👍/👎 per trial.
function feedbackContextFor(assistantText, sources) {
  const lastUser = [...plannerState.messages].reverse().find((m) => m.role === "user");
  const sourceIds = (sources || [])
    .map((s) => (s && s.nct_id) || "")
    .filter(Boolean);
  return {
    query: lastUser ? lastUser.content : "",
    assistant_message: assistantText || "",
    source_nct_ids: sourceIds,
  };
}

function attachFeedbackBar(messageDiv, assistantText, sources) {
  const bar = document.createElement("div");
  bar.className = "feedback-bar";

  const label = document.createElement("span");
  label.className = "feedback-bar-label";
  label.textContent = "Was this helpful?";
  bar.appendChild(label);

  const upBtn = document.createElement("button");
  upBtn.type = "button";
  upBtn.className = "feedback-thumb up";
  upBtn.textContent = "👍";
  upBtn.title = "Helpful";

  const downBtn = document.createElement("button");
  downBtn.type = "button";
  downBtn.className = "feedback-thumb down";
  downBtn.textContent = "👎";
  downBtn.title = "Not helpful";

  bar.appendChild(upBtn);
  bar.appendChild(downBtn);

  let rated = false;
  function lockThumbs() {
    rated = true;
    upBtn.disabled = true;
    downBtn.disabled = true;
  }

  upBtn.addEventListener("click", async () => {
    if (rated) return;
    lockThumbs();
    upBtn.classList.add("selected");
    try {
      await postFeedback({
        type: "rating_up",
        context: feedbackContextFor(assistantText, sources),
      });
      const thanks = document.createElement("span");
      thanks.className = "feedback-thanks";
      thanks.textContent = "Thanks!";
      bar.appendChild(thanks);
    } catch (err) {
      showToast(`Failed to send feedback: ${err.message}`, "error");
      rated = false;
      upBtn.disabled = false;
      downBtn.disabled = false;
      upBtn.classList.remove("selected");
    }
  });

  downBtn.addEventListener("click", async () => {
    if (rated) return;
    lockThumbs();
    downBtn.classList.add("selected");
    // Reveal an optional reason input below the bar. We send the rating
    // immediately on click; if the user types a reason later we send a
    // follow-up update (simplest: send as a separate "correction" entry).
    let initialId = null;
    try {
      const resp = await postFeedback({
        type: "rating_down",
        context: feedbackContextFor(assistantText, sources),
      });
      initialId = resp.id;
    } catch (err) {
      showToast(`Failed to send feedback: ${err.message}`, "error");
      rated = false;
      upBtn.disabled = false;
      downBtn.disabled = false;
      downBtn.classList.remove("selected");
      return;
    }

    const reasonWrap = document.createElement("div");
    reasonWrap.className = "feedback-down-reason";
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Optional: what was wrong? (press Enter to send)";
    reasonWrap.appendChild(input);
    bar.appendChild(reasonWrap);
    input.focus();

    input.addEventListener("keydown", async (evt) => {
      if (evt.key !== "Enter") return;
      const text = input.value.trim();
      if (!text) {
        reasonWrap.remove();
        return;
      }
      input.disabled = true;
      try {
        await postFeedback({
          type: "rating_down",
          context: feedbackContextFor(assistantText, sources),
          reason: text,
        });
        reasonWrap.remove();
        const thanks = document.createElement("span");
        thanks.className = "feedback-thanks";
        thanks.textContent = "Thanks for the detail!";
        bar.appendChild(thanks);
      } catch (err) {
        showToast(`Failed to send reason: ${err.message}`, "error");
        input.disabled = false;
      }
    });
  });

  messageDiv.appendChild(bar);
}

function attachSourcesToMessage(messageDiv, sources) {
  if (!sources || sources.length === 0) return;
  const details = document.createElement("details");
  details.className = "message-sources";
  const summary = document.createElement("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);
  const wrap = document.createElement("div");
  wrap.innerHTML = sources.map(renderSourceCard).join("");
  details.appendChild(wrap);
  messageDiv.appendChild(details);
}

// Parse a [CHOICES] ... [/CHOICES] block out of assistant text and return
// { cleanText, choices: string[] }. Tolerates leading whitespace and
// preserves the prose before the block as the visible message.
function extractChoices(text) {
  const re = /\[CHOICES\]\s*([\s\S]*?)\s*\[\/CHOICES\]/i;
  const m = text.match(re);
  if (!m) return { cleanText: text, choices: [] };
  const block = m[1];
  const choices = block
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.startsWith("- "))
    .map((l) => l.slice(2).trim())
    .filter(Boolean);
  const cleanText = text.replace(re, "").trim();
  return { cleanText, choices };
}

function attachChoicesToMessage(messageDiv, choices) {
  if (!choices || choices.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "choice-chips";
  for (const choice of choices) {
    const btn = document.createElement("button");
    btn.className = "choice-chip";
    btn.type = "button";
    btn.textContent = choice;
    btn.addEventListener("click", () => {
      // Disable every chip in this group so the clinician can't double-click.
      wrap.querySelectorAll(".choice-chip").forEach((b) => (b.disabled = true));
      btn.classList.add("selected");
      const input = plannerEls().input;
      input.value = choice;
      plannerSend();
    });
    wrap.appendChild(btn);
  }
  messageDiv.appendChild(wrap);
}

function revealPlannerActions() {
  plannerEls().summarizeBtn.classList.remove("hidden");
}

function resetPlanner() {
  plannerState.messages = [];
  plannerState.lastSummary = "";
  plannerState.currentProtocol = null;
  plannerState.protocolVersions = [];
  plannerState.activeVersion = -1;
  plannerState.refineMessages = [];
  const e = plannerEls();
  e.messagesBox.innerHTML = "";
  e.summarizeBtn.classList.add("hidden");
  e.summaryBox.classList.add("hidden");
  e.protocolLoading.classList.add("hidden");
  e.protocolResult.classList.add("hidden");
  e.input.value = "";
  if (e.refineThread) e.refineThread.innerHTML = "";
  if (e.refineInput) e.refineInput.value = "";
  if (e.versionList) e.versionList.innerHTML = "";
  if (e.compareBtn) e.compareBtn.disabled = true;
}

async function plannerSend() {
  const e = plannerEls();
  const query = e.input.value.trim();
  if (!query) return;

  plannerState.messages.push({ role: "user", content: query });
  addChatMessage("user", query);
  e.input.value = "";
  e.sendBtn.disabled = true;
  e.sendBtn.textContent = "Thinking...";

  // Create assistant bubble to stream into
  const assistantDiv = addChatMessage("assistant", "");
  const contentNode = document.createTextNode("");
  assistantDiv.appendChild(contentNode);
  let assistantText = "";
  let finished = false;
  let streamSources = [];

  try {
    const resp = await apiFetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: plannerState.messages,
        top_k: parseInt(e.topk.value),
        include_landscape: e.showLandscape.checked,
      }),
    });
    if (!resp.ok || !resp.body) throw new Error(`Server error: ${resp.status}`);

    // Parse SSE manually from streamed fetch body.
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const lines = rawEvent.split("\n");
        let evt = "message";
        let data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) evt = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;

        if (evt === "landscape") {
          try {
            const landscape = JSON.parse(data);
            // Insert the landscape card before the assistant bubble so the
            // clinician sees population context first, then the streamed reply.
            const card = renderLandscapeCard(landscape);
            e.messagesBox.insertBefore(card, assistantDiv);
            e.messagesBox.scrollTop = e.messagesBox.scrollHeight;
          } catch {}
        } else if (evt === "sources") {
          try { streamSources = JSON.parse(data); } catch { streamSources = []; }
        } else if (evt === "token") {
          try {
            assistantText += JSON.parse(data);
            contentNode.nodeValue = assistantText;
            e.messagesBox.scrollTop = e.messagesBox.scrollHeight;
          } catch {}
        } else if (evt === "done") {
          finished = true;
        } else if (evt === "error") {
          let msg = "stream error";
          try { msg = JSON.parse(data).message || msg; } catch {}
          throw new Error(msg);
        }
      }
    }

    if (!finished && !assistantText) {
      assistantText = "No response received.";
      contentNode.nodeValue = assistantText;
    }

    // Strip any [CHOICES] block out of the bubble text and render the
    // options as clickable chips below the message. The full (unstripped)
    // text goes into conversation history so the model can see what it
    // offered, but the UI only shows the prose and the chips.
    const { cleanText, choices } = extractChoices(assistantText);
    if (choices.length > 0) {
      contentNode.nodeValue = cleanText;
    }

    // Replace the streaming text node with a span that linkifies any NCT IDs
    // (NCT followed by 8 digits) to clinicaltrials.gov so the clinician can
    // click through to the source. Done after streaming completes — easier
    // than diffing partial tokens that may split an NCT ID across chunks.
    const finalText = choices.length > 0 ? cleanText : assistantText;
    const linkified = linkifyNctIds(finalText);
    assistantDiv.replaceChild(linkified, contentNode);

    // Persist assistant turn (keep original text in history for LLM context).
    plannerState.messages.push({ role: "assistant", content: assistantText });
    attachSourcesToMessage(assistantDiv, streamSources);
    attachChoicesToMessage(assistantDiv, choices);
    attachFeedbackBar(assistantDiv, assistantText, streamSources);
    revealPlannerActions();
  } catch (err) {
    assistantText = assistantText || `Error: ${err.message}`;
    contentNode.nodeValue = assistantText;
    plannerState.messages.push({ role: "assistant", content: assistantText });
  } finally {
    e.sendBtn.disabled = false;
    e.sendBtn.textContent = "Send";
  }
}

async function plannerSummarize() {
  const e = plannerEls();
  if (plannerState.messages.length === 0) return;

  e.summarizeBtn.disabled = true;
  e.summarizeBtn.textContent = "Summarizing...";
  e.summaryBox.classList.remove("hidden");
  e.summaryContent.textContent = "Generating summary...";

  try {
    const resp = await apiFetch(`${API_BASE}/chat/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: plannerState.messages }),
    });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    const data = await resp.json();
    plannerState.lastSummary = data.summary;
    e.summaryContent.textContent = data.summary;
  } catch (err) {
    e.summaryContent.textContent = `Failed to summarize: ${err.message}`;
  } finally {
    e.summarizeBtn.disabled = false;
    e.summarizeBtn.textContent = "Summarize & plan report";
  }
}

async function plannerGenerateProtocol() {
  const e = plannerEls();
  if (!plannerState.lastSummary) return;

  e.protocolLoading.classList.remove("hidden");
  e.protocolResult.classList.add("hidden");
  e.genProtocolBtn.disabled = true;

  try {
    const resp = await apiFetch(`${API_BASE}/protocol/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: plannerState.lastSummary,
        top_k: parseInt(e.topk.value),
      }),
    });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    const data = await resp.json();
    plannerState.currentProtocol = data.protocol;
    plannerState.protocolVersions = [{
      label: "v1 (original)",
      protocol: data.protocol,
      userRequest: null,
      assistantNote: "Initial generation from the planning brief.",
      changedFields: [],
      timestamp: new Date().toISOString(),
    }];
    plannerState.activeVersion = 0;
    plannerState.refineMessages = [];
    renderVersionBar();
    if (e.refineThread) e.refineThread.innerHTML = "";
    renderProtocolPreview(data.protocol, "planner-protocol-preview");
    e.protocolResult.classList.remove("hidden");
  } catch (err) {
    alert(`Failed to generate protocol: ${err.message}`);
  } finally {
    e.protocolLoading.classList.add("hidden");
    e.genProtocolBtn.disabled = false;
  }
}

// ===================== PROTOCOL REFINEMENT =====================

function renderVersionBar() {
  const e = plannerEls();
  if (!e.versionList) return;
  e.versionList.innerHTML = "";
  plannerState.protocolVersions.forEach((v, idx) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "version-chip" + (idx === plannerState.activeVersion ? " active" : "");
    const label = document.createElement("span");
    label.className = "version-chip-label";
    label.textContent = v.label;
    chip.appendChild(label);
    if (v.userRequest) {
      const note = document.createElement("span");
      note.className = "version-chip-note";
      const trimmed = v.userRequest.length > 40
        ? v.userRequest.slice(0, 40) + "…"
        : v.userRequest;
      note.textContent = `· ${trimmed}`;
      chip.appendChild(note);
    }
    chip.title = v.assistantNote || v.userRequest || v.label;
    chip.addEventListener("click", () => switchToVersion(idx));
    e.versionList.appendChild(chip);
  });
  e.compareBtn.disabled = plannerState.protocolVersions.length < 2;
}

function switchToVersion(idx) {
  const versions = plannerState.protocolVersions;
  if (idx < 0 || idx >= versions.length) return;
  plannerState.activeVersion = idx;
  plannerState.currentProtocol = versions[idx].protocol;
  renderVersionBar();
  renderProtocolPreview(versions[idx].protocol, "planner-protocol-preview");
}

function appendRefineMessage(role, content, changedFields) {
  const e = plannerEls();
  const div = document.createElement("div");
  div.className = `refine-msg ${role}`;
  div.textContent = content;
  if (role === "assistant" && changedFields && changedFields.length) {
    const note = document.createElement("div");
    note.className = "changed-fields";
    note.textContent = `Changed: ${changedFields.join(", ")}`;
    div.appendChild(note);
  }
  e.refineThread.appendChild(div);
  e.refineThread.scrollTop = e.refineThread.scrollHeight;
}

async function plannerSendRefinement() {
  const e = plannerEls();
  const text = e.refineInput.value.trim();
  if (!text || !plannerState.currentProtocol) return;

  appendRefineMessage("user", text);
  plannerState.refineMessages.push({ role: "user", content: text });
  e.refineInput.value = "";
  e.refineSendBtn.disabled = true;
  e.refineSendBtn.textContent = "Updating...";

  // Show a placeholder assistant bubble that we'll replace once the response lands.
  const pendingDiv = document.createElement("div");
  pendingDiv.className = "refine-msg assistant";
  pendingDiv.textContent = "Updating the protocol...";
  e.refineThread.appendChild(pendingDiv);
  e.refineThread.scrollTop = e.refineThread.scrollHeight;

  try {
    const resp = await apiFetch(`${API_BASE}/protocol/refine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        protocol: plannerState.currentProtocol,
        refinement_messages: plannerState.refineMessages,
        original_summary: plannerState.lastSummary,
      }),
    });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    const data = await resp.json();

    // Replace pending bubble with the real assistant note + changed-fields tag.
    pendingDiv.remove();
    const note = data.assistant_message
      || (data.changed_fields && data.changed_fields.length
            ? `Updated ${data.changed_fields.join(", ")}.`
            : "Protocol updated.");
    appendRefineMessage("assistant", note, data.changed_fields || []);
    plannerState.refineMessages.push({ role: "assistant", content: note });

    // Append a new version and switch to it.
    const nextNum = plannerState.protocolVersions.length + 1;
    plannerState.protocolVersions.push({
      label: `v${nextNum}`,
      protocol: data.protocol,
      userRequest: text,
      assistantNote: note,
      changedFields: data.changed_fields || [],
      timestamp: new Date().toISOString(),
    });
    plannerState.activeVersion = plannerState.protocolVersions.length - 1;
    plannerState.currentProtocol = data.protocol;
    renderVersionBar();
    renderProtocolPreview(data.protocol, "planner-protocol-preview");
  } catch (err) {
    pendingDiv.remove();
    appendRefineMessage("assistant", `Failed to apply correction: ${err.message}`, []);
    // Pop the user message off the refine history so retrying doesn't re-send it.
    plannerState.refineMessages.pop();
  } finally {
    e.refineSendBtn.disabled = false;
    e.refineSendBtn.textContent = "Send correction";
  }
}

// ===================== VERSION DIFF =====================

// Fields to consider when computing a diff. We omit administrative placeholders
// (which the clinician fills in themselves) and keep the order stable so the
// diff reads top-down like the rendered protocol.
const DIFF_FIELDS = [
  "title", "official_title", "phase", "status", "conditions",
  "summary", "description",
  "primary_objectives", "secondary_objectives", "exploratory_objectives",
  "study_design", "study_schema",
  "intervention_name", "intervention_description", "comparator", "treatment_duration",
  "inclusion_criteria", "exclusion_criteria",
  "primary_endpoints", "secondary_endpoints",
  "estimated_enrollment", "sample_size_justification", "statistical_analysis",
  "safety_monitoring", "adverse_event_reporting", "dose_modification",
  "study_assessments", "study_schedule_table",
  "ethical_considerations", "data_management", "regulatory_considerations",
  "sex", "minimum_age", "references",
];

function fieldToString(value) {
  if (value === undefined || value === null || value === "") return "";
  if (Array.isArray(value)) {
    return value.map((v) => {
      if (v && typeof v === "object") {
        // Schedule rows: { visit, timepoint, procedures }
        return `• ${v.visit || ""} — ${v.timepoint || ""}: ${v.procedures || ""}`;
      }
      return `• ${v}`;
    }).join("\n");
  }
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function fieldLabel(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function openDiffModal() {
  const e = plannerEls();
  if (plannerState.protocolVersions.length < 2) return;
  // Populate selectors. Default: from = previous version, to = active.
  const fromIdx = Math.max(0, plannerState.activeVersion - 1);
  const toIdx = plannerState.activeVersion >= 0
    ? plannerState.activeVersion
    : plannerState.protocolVersions.length - 1;
  populateDiffSelector(e.diffFrom, fromIdx);
  populateDiffSelector(e.diffTo, toIdx);
  renderDiff();
  e.diffModal.classList.remove("hidden");
}

function closeDiffModal() {
  plannerEls().diffModal.classList.add("hidden");
}

function populateDiffSelector(selectEl, selectedIdx) {
  selectEl.innerHTML = "";
  plannerState.protocolVersions.forEach((v, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = v.userRequest
      ? `${v.label} — ${v.userRequest.slice(0, 50)}`
      : v.label;
    if (idx === selectedIdx) opt.selected = true;
    selectEl.appendChild(opt);
  });
}

function renderDiff() {
  const e = plannerEls();
  const fromIdx = parseInt(e.diffFrom.value, 10);
  const toIdx = parseInt(e.diffTo.value, 10);
  const fromV = plannerState.protocolVersions[fromIdx];
  const toV = plannerState.protocolVersions[toIdx];
  e.diffBody.innerHTML = "";

  if (!fromV || !toV) return;
  if (fromIdx === toIdx) {
    e.diffBody.innerHTML = `<div class="diff-empty-state">Same version selected on both sides — pick two different versions to compare.</div>`;
    return;
  }

  const diffs = [];
  DIFF_FIELDS.forEach((key) => {
    const oldStr = fieldToString(fromV.protocol[key]);
    const newStr = fieldToString(toV.protocol[key]);
    if (oldStr !== newStr) diffs.push({ key, oldStr, newStr });
  });

  if (diffs.length === 0) {
    e.diffBody.innerHTML = `<div class="diff-empty-state">No differences detected between ${escapeHtml(fromV.label)} and ${escapeHtml(toV.label)}.</div>`;
    return;
  }

  const header = document.createElement("div");
  header.style.marginBottom = "12px";
  header.style.fontSize = "12px";
  header.style.color = "var(--text-secondary)";
  header.textContent = `${diffs.length} field${diffs.length === 1 ? "" : "s"} changed between ${fromV.label} and ${toV.label}.`;
  e.diffBody.appendChild(header);

  diffs.forEach((d) => {
    const section = document.createElement("div");
    section.className = "diff-section";

    const title = document.createElement("div");
    title.className = "diff-section-title";
    title.textContent = fieldLabel(d.key);
    section.appendChild(title);

    const cols = document.createElement("div");
    cols.className = "diff-cols";

    const oldCol = document.createElement("div");
    oldCol.className = "diff-col old" + (d.oldStr ? "" : " empty");
    const oldHeader = document.createElement("div");
    oldHeader.className = "diff-col-header";
    oldHeader.textContent = `${fromV.label} (before)`;
    oldCol.appendChild(oldHeader);
    const oldBody = document.createElement("div");
    oldBody.textContent = d.oldStr || "(empty)";
    oldCol.appendChild(oldBody);

    const newCol = document.createElement("div");
    newCol.className = "diff-col new" + (d.newStr ? "" : " empty");
    const newHeader = document.createElement("div");
    newHeader.className = "diff-col-header";
    newHeader.textContent = `${toV.label} (after)`;
    newCol.appendChild(newHeader);
    const newBody = document.createElement("div");
    newBody.textContent = d.newStr || "(empty)";
    newCol.appendChild(newBody);

    cols.appendChild(oldCol);
    cols.appendChild(newCol);
    section.appendChild(cols);
    e.diffBody.appendChild(section);
  });
}

// Download the current protocol in the given format ("docx" or "pdf").
async function plannerDownloadProtocolAs(format) {
  const e = plannerEls();
  if (!plannerState.currentProtocol) return;
  const btn = format === "pdf" ? e.downloadPdfBtn : e.downloadBtn;
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Downloading...";
  try {
    const resp = await apiFetch(`${API_BASE}/protocol/${format}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ protocol: plannerState.currentProtocol }),
    });
    if (!resp.ok) throw new Error("Download failed");
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const stem = (plannerState.currentProtocol.protocol_id || "").trim() || "protocol";
    a.download = `${stem}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Failed to download: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

// ===================== ADMIN FEEDBACK REVIEW =====================

const adminEls = () => ({
  panel: document.getElementById("admin-panel"),
  stats: document.getElementById("admin-panel-stats"),
  refreshBtn: document.getElementById("admin-refresh-btn"),
  closeBtn: document.getElementById("admin-close-btn"),
  aliasesBox: document.getElementById("admin-aliases"),
  feedbackList: document.getElementById("admin-feedback-list"),
});

function adminRenderStats(stats) {
  const e = adminEls();
  e.stats.innerHTML = `
    <span class="admin-stat">👍 ${stats.rating_up || 0}</span>
    <span class="admin-stat">👎 ${stats.rating_down || 0}</span>
    <span class="admin-stat">Open corrections: ${stats.open_corrections || 0}</span>
    <span class="admin-stat">Total entries: ${stats.total || 0}</span>
  `;
}

function adminRenderAliases(aliases) {
  const e = adminEls();
  e.aliasesBox.innerHTML = "";
  const entries = Object.entries(aliases || {});
  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "admin-alias-empty";
    empty.textContent = "No aliases promoted yet. Promote a 'Missing drug alias' correction below to populate this dictionary.";
    e.aliasesBox.appendChild(empty);
    return;
  }
  entries.sort((a, b) => a[0].localeCompare(b[0]));
  entries.forEach(([alias, canonical]) => {
    const row = document.createElement("div");
    row.className = "admin-alias-row";
    const aliasEl = document.createElement("strong");
    aliasEl.textContent = alias;
    row.appendChild(aliasEl);
    row.appendChild(document.createTextNode(` → ${canonical}`));
    e.aliasesBox.appendChild(row);
  });
}

function adminRenderFeedbackRow(entry) {
  const row = document.createElement("div");
  row.className = `admin-feedback-row ${entry.status || "open"}`;

  const meta = document.createElement("div");
  meta.className = "admin-feedback-meta";
  const typeChip = document.createElement("span");
  typeChip.className = `admin-feedback-type ${(entry.type || "").replace(/_/g, "-")}`;
  typeChip.textContent = (entry.type || "unknown").replace(/_/g, " ");
  meta.appendChild(typeChip);
  const statusChip = document.createElement("span");
  statusChip.className = "admin-feedback-status";
  statusChip.textContent = entry.status || "open";
  meta.appendChild(statusChip);
  const ts = document.createElement("span");
  const d = entry.created_at ? new Date(entry.created_at) : null;
  ts.textContent = d && !isNaN(d) ? d.toLocaleString() : (entry.created_at || "");
  meta.appendChild(ts);
  const id = document.createElement("span");
  id.textContent = `id: ${(entry.id || "").slice(0, 8)}…`;
  id.style.fontFamily = "monospace";
  meta.appendChild(id);
  row.appendChild(meta);

  const body = document.createElement("div");
  body.className = "admin-feedback-body";
  if (entry.type === "correction") {
    if (entry.correction_kind === "missing_alias") {
      body.innerHTML = `
        <div><span class="field-label">Alias:</span> <strong>${escapeHtml(entry.alias || "")}</strong></div>
        <div><span class="field-label">Canonical:</span> <strong>${escapeHtml(entry.canonical || "")}</strong></div>
      `;
    } else {
      body.innerHTML = `<div><span class="field-label">Kind:</span> ${escapeHtml(entry.correction_kind || "")}</div>`;
    }
    if (entry.notes) {
      const notes = document.createElement("div");
      notes.innerHTML = `<span class="field-label">Notes:</span> ${escapeHtml(entry.notes)}`;
      body.appendChild(notes);
    }
  } else if (entry.type === "rating_down" && entry.reason) {
    body.innerHTML = `<span class="field-label">Reason:</span> ${escapeHtml(entry.reason)}`;
  } else if (entry.type === "rating_up") {
    body.innerHTML = `<em>Thumbs up — no additional context.</em>`;
  } else if (entry.type === "rating_down") {
    body.innerHTML = `<em>Thumbs down — no reason provided.</em>`;
  }
  row.appendChild(body);

  if (entry.context && (entry.context.query || entry.context.assistant_message)) {
    const ctx = document.createElement("div");
    ctx.className = "admin-feedback-context";
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "Conversation context";
    det.appendChild(sum);
    if (entry.context.query) {
      const q = document.createElement("div");
      q.innerHTML = `<span class="field-label">Query:</span> ${escapeHtml(entry.context.query)}`;
      det.appendChild(q);
    }
    if (entry.context.assistant_message) {
      const a = document.createElement("div");
      a.style.marginTop = "4px";
      a.innerHTML = `<span class="field-label">Reply:</span> ${escapeHtml(entry.context.assistant_message)}`;
      det.appendChild(a);
    }
    ctx.appendChild(det);
    row.appendChild(ctx);
  }

  if ((entry.status || "open") === "open") {
    const actions = document.createElement("div");
    actions.className = "admin-feedback-actions";

    if (entry.type === "correction" && entry.correction_kind === "missing_alias") {
      const aliasInput = document.createElement("input");
      aliasInput.type = "text";
      aliasInput.value = entry.alias || "";
      aliasInput.placeholder = "alias";
      const canonicalInput = document.createElement("input");
      canonicalInput.type = "text";
      canonicalInput.value = entry.canonical || "";
      canonicalInput.placeholder = "canonical name";
      const promoteBtn = document.createElement("button");
      promoteBtn.className = "primary-btn";
      promoteBtn.textContent = "Promote to dictionary";
      promoteBtn.addEventListener("click", async () => {
        const alias = aliasInput.value.trim();
        const canonical = canonicalInput.value.trim();
        if (!alias || !canonical) {
          showToast("Both alias and canonical name are required.", "error");
          return;
        }
        promoteBtn.disabled = true;
        promoteBtn.textContent = "Promoting...";
        try {
          const resp = await apiFetch(`${API_BASE}/feedback/promote`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ feedback_id: entry.id, alias, canonical }),
          });
          if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
          showToast(`Promoted: ${alias} → ${canonical}`);
          adminLoad();
        } catch (err) {
          showToast(`Failed: ${err.message}`, "error");
          promoteBtn.disabled = false;
          promoteBtn.textContent = "Promote to dictionary";
        }
      });
      actions.appendChild(aliasInput);
      actions.appendChild(canonicalInput);
      actions.appendChild(promoteBtn);
    }

    const resolveBtn = document.createElement("button");
    resolveBtn.className = "secondary-btn";
    resolveBtn.textContent = "Mark resolved";
    resolveBtn.addEventListener("click", () => adminUpdateStatus(entry.id, "resolved"));
    const dismissBtn = document.createElement("button");
    dismissBtn.className = "secondary-btn";
    dismissBtn.textContent = "Dismiss";
    dismissBtn.addEventListener("click", () => adminUpdateStatus(entry.id, "dismissed"));
    actions.appendChild(resolveBtn);
    actions.appendChild(dismissBtn);

    row.appendChild(actions);
  }

  return row;
}

async function adminUpdateStatus(id, status) {
  try {
    const resp = await apiFetch(`${API_BASE}/feedback/${id}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    showToast(`Marked ${status}.`);
    adminLoad();
  } catch (err) {
    showToast(`Failed: ${err.message}`, "error");
  }
}

async function adminLoad() {
  const e = adminEls();
  e.feedbackList.textContent = "Loading...";
  try {
    const [feedbackResp, aliasesResp] = await Promise.all([
      apiFetch(`${API_BASE}/feedback`),
      apiFetch(`${API_BASE}/aliases`),
    ]);
    if (!feedbackResp.ok) throw new Error(`/feedback ${feedbackResp.status}`);
    if (!aliasesResp.ok) throw new Error(`/aliases ${aliasesResp.status}`);
    const fb = await feedbackResp.json();
    const al = await aliasesResp.json();

    adminRenderStats(fb.stats || {});
    adminRenderAliases(al.aliases || {});

    e.feedbackList.innerHTML = "";
    const entries = fb.entries || [];
    if (entries.length === 0) {
      const empty = document.createElement("div");
      empty.className = "admin-feedback-empty";
      empty.textContent = "No feedback yet. Once users rate replies or submit corrections, they'll appear here.";
      e.feedbackList.appendChild(empty);
      return;
    }
    entries.forEach((entry) => {
      e.feedbackList.appendChild(adminRenderFeedbackRow(entry));
    });
  } catch (err) {
    e.feedbackList.innerHTML = `<div class="admin-feedback-empty">Failed to load: ${escapeHtml(err.message)}</div>`;
  }
}

function showAdminPanel() {
  const e = adminEls();
  if (!e.panel) return;
  e.panel.classList.remove("hidden");
  e.panel.scrollIntoView({ behavior: "smooth", block: "start" });
  adminLoad();
}

function hideAdminPanel() {
  const e = adminEls();
  if (!e.panel) return;
  e.panel.classList.add("hidden");
}

function adminInit() {
  const e = adminEls();
  if (!e.panel) return;

  // Wire close + refresh once. The panel itself is shown either by a click
  // on the footer link or by an ?admin=1 URL flag at load time.
  e.refreshBtn.addEventListener("click", adminLoad);
  e.closeBtn.addEventListener("click", hideAdminPanel);

  // Footer "Admin" link: toggle in-place rather than navigating, so it works
  // under file://, http://, or any other scheme.
  const adminLink = document.getElementById("admin-link");
  if (adminLink) {
    adminLink.addEventListener("click", (evt) => {
      evt.preventDefault();
      if (e.panel.classList.contains("hidden")) {
        showAdminPanel();
      } else {
        hideAdminPanel();
      }
    });
  }

  // Optional: still honor ?admin=1 in the URL on first load so a bookmarked
  // link drops the user straight into the admin view.
  const params = new URLSearchParams(window.location.search);
  if (params.get("admin") === "1") {
    showAdminPanel();
  }
}

// Wire up planner event listeners
(function initPlanner() {
  const e = plannerEls();
  if (!e.sendBtn) return;  // tab not present
  e.sendBtn.addEventListener("click", plannerSend);
  e.summarizeBtn.addEventListener("click", plannerSummarize);
  e.resetBtn.addEventListener("click", resetPlanner);
  e.genProtocolBtn.addEventListener("click", plannerGenerateProtocol);
  e.skipProtocolBtn.addEventListener("click", () => e.summaryBox.classList.add("hidden"));
  e.downloadBtn.addEventListener("click", () => plannerDownloadProtocolAs("docx"));
  if (e.downloadPdfBtn) {
    e.downloadPdfBtn.addEventListener("click", () => plannerDownloadProtocolAs("pdf"));
  }
  e.topk.addEventListener("input", (evt) => { e.topkVal.textContent = evt.target.value; });
  e.input.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      plannerSend();
    }
  });

  // Refinement panel + version controls
  if (e.refineSendBtn) {
    e.refineSendBtn.addEventListener("click", plannerSendRefinement);
  }
  if (e.refineInput) {
    e.refineInput.addEventListener("keydown", (evt) => {
      if (evt.key === "Enter" && !evt.shiftKey) {
        evt.preventDefault();
        plannerSendRefinement();
      }
    });
  }
  if (e.compareBtn) e.compareBtn.addEventListener("click", openDiffModal);
  if (e.diffCloseBtn) e.diffCloseBtn.addEventListener("click", closeDiffModal);
  if (e.diffModal) {
    e.diffModal.querySelector(".diff-modal-backdrop")
      ?.addEventListener("click", closeDiffModal);
  }
  if (e.diffFrom) e.diffFrom.addEventListener("change", renderDiff);
  if (e.diffTo) e.diffTo.addEventListener("change", renderDiff);

  // Initialize the admin panel if ?admin=1 is in the URL.
  adminInit();
})();
