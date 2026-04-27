const API_BASE = "http://localhost:8000/api";

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
};

const plannerEls = () => ({
  messagesBox: document.getElementById("planner-messages"),
  input: document.getElementById("planner-input"),
  sendBtn: document.getElementById("planner-send-btn"),
  summarizeBtn: document.getElementById("planner-summarize-btn"),
  resetBtn: document.getElementById("planner-reset-btn"),
  topk: document.getElementById("planner-topk"),
  topkVal: document.getElementById("planner-topk-val"),
  summaryBox: document.getElementById("planner-summary-box"),
  summaryContent: document.getElementById("planner-summary-content"),
  genProtocolBtn: document.getElementById("planner-generate-protocol-btn"),
  skipProtocolBtn: document.getElementById("planner-skip-protocol-btn"),
  protocolLoading: document.getElementById("planner-protocol-loading"),
  protocolResult: document.getElementById("planner-protocol-result"),
  protocolPreview: document.getElementById("planner-protocol-preview"),
  downloadBtn: document.getElementById("planner-download-docx-btn"),
  downloadPdfBtn: document.getElementById("planner-download-pdf-btn"),
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
  const e = plannerEls();
  e.messagesBox.innerHTML = "";
  e.summarizeBtn.classList.add("hidden");
  e.summaryBox.classList.add("hidden");
  e.protocolLoading.classList.add("hidden");
  e.protocolResult.classList.add("hidden");
  e.input.value = "";
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
    const resp = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: plannerState.messages,
        top_k: parseInt(e.topk.value),
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

    // Persist assistant turn (keep original text in history for LLM context).
    plannerState.messages.push({ role: "assistant", content: assistantText });
    attachSourcesToMessage(assistantDiv, streamSources);
    attachChoicesToMessage(assistantDiv, choices);
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
    const resp = await fetch(`${API_BASE}/chat/summarize`, {
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
    const resp = await fetch(`${API_BASE}/protocol/json`, {
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
    renderProtocolPreview(data.protocol, "planner-protocol-preview");
    e.protocolResult.classList.remove("hidden");
  } catch (err) {
    alert(`Failed to generate protocol: ${err.message}`);
  } finally {
    e.protocolLoading.classList.add("hidden");
    e.genProtocolBtn.disabled = false;
  }
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
    const resp = await fetch(`${API_BASE}/protocol/${format}`, {
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
})();
