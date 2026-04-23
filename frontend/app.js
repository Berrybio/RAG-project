const API_BASE = "http://localhost:8000/api";

// --- Tab switching ---
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`${tab.dataset.tab}-tab`).classList.add("active");
  });
});

// --- Range slider labels ---
document.getElementById("search-topk").addEventListener("input", (e) => {
  document.getElementById("search-topk-val").textContent = e.target.value;
});
document.getElementById("protocol-topk").addEventListener("input", (e) => {
  document.getElementById("protocol-topk-val").textContent = e.target.value;
});

// --- Example buttons ---
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("search-query").value = btn.dataset.query;
    document.getElementById("search-btn").click();
  });
});

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

// ===================== SEARCH =====================

document.getElementById("search-btn").addEventListener("click", async () => {
  const query = document.getElementById("search-query").value.trim();
  if (!query) return;

  const topK = parseInt(document.getElementById("search-topk").value);
  const btn = document.getElementById("search-btn");
  const answerBox = document.getElementById("search-answer");
  const answerContent = document.getElementById("answer-content");
  const sourcesBox = document.getElementById("search-sources");
  const sourcesList = document.getElementById("sources-list");

  btn.disabled = true;
  btn.textContent = "Searching...";
  answerBox.classList.remove("hidden");
  answerContent.textContent = "";
  sourcesBox.classList.add("hidden");

  try {
    const evtSource = new EventSource(
      `${API_BASE}/search/stream?query=${encodeURIComponent(query)}&top_k=${topK}`
    );
    let finished = false;

    evtSource.addEventListener("sources", (e) => {
      const sources = JSON.parse(e.data);
      if (sources.length > 0) {
        sourcesBox.classList.remove("hidden");
        document.getElementById("source-count").textContent = `(${sources.length})`;
        sourcesList.innerHTML = sources.map(renderSourceCard).join("");
      }
    });

    evtSource.addEventListener("token", (e) => {
      const token = JSON.parse(e.data);
      answerContent.textContent += token;
    });

    evtSource.addEventListener("done", () => {
      finished = true;
      evtSource.close();
      btn.disabled = false;
      btn.textContent = "Search";
    });

    evtSource.addEventListener("error", () => {
      // EventSource fires "error" on normal connection close too. Only surface
      // an error message if the stream never reached "done".
      if (finished) return;
      evtSource.close();
      btn.disabled = false;
      btn.textContent = "Search";
      if (!answerContent.textContent) {
        answerContent.textContent = "An error occurred while searching. Please try again.";
      }
    });
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Search";
    answerContent.textContent = "Failed to connect to the server.";
  }
});

// Enter key to search
document.getElementById("search-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("search-btn").click();
  }
});

// ===================== PROTOCOL =====================

let currentProtocol = null;

document.getElementById("protocol-btn").addEventListener("click", async () => {
  const query = document.getElementById("protocol-query").value.trim();
  if (!query) return;

  const topK = parseInt(document.getElementById("protocol-topk").value);
  const btn = document.getElementById("protocol-btn");
  const loading = document.getElementById("protocol-loading");
  const result = document.getElementById("protocol-result");

  btn.disabled = true;
  btn.textContent = "Generating...";
  loading.classList.remove("hidden");
  result.classList.add("hidden");

  try {
    const response = await fetch(`${API_BASE}/protocol/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    currentProtocol = data.protocol;

    renderProtocolPreview(data.protocol);
    document.getElementById("protocol-json-content").textContent = JSON.stringify(
      data.protocol,
      null,
      2
    );

    // Reference trials
    if (data.reference_trials && data.reference_trials.length > 0) {
      const refsBox = document.getElementById("protocol-refs");
      refsBox.classList.remove("hidden");
      document.getElementById("protocol-refs-list").innerHTML = data.reference_trials
        .map(renderSourceCard)
        .join("");
    }

    loading.classList.add("hidden");
    result.classList.remove("hidden");
  } catch (err) {
    loading.classList.add("hidden");
    alert("Failed to generate protocol: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Protocol";
  }
});

// Toggle JSON view
document.getElementById("toggle-json-btn").addEventListener("click", () => {
  const jsonDiv = document.getElementById("protocol-json");
  const btn = document.getElementById("toggle-json-btn");
  jsonDiv.classList.toggle("hidden");
  btn.textContent = jsonDiv.classList.contains("hidden") ? "Show JSON" : "Hide JSON";
});

// Download .docx
document.getElementById("download-docx-btn").addEventListener("click", async () => {
  if (!currentProtocol) return;

  const btn = document.getElementById("download-docx-btn");
  btn.disabled = true;
  btn.textContent = "Downloading...";

  try {
    const response = await fetch(`${API_BASE}/protocol/docx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ protocol: currentProtocol }),
    });

    if (!response.ok) throw new Error("Download failed");

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (currentProtocol.protocol_id || "protocol") + ".docx";
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Failed to download: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Download as Word Document";
  }
});

// Enter key for protocol
document.getElementById("protocol-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("protocol-btn").click();
  }
});

// --- Protocol preview rendering ---
function renderProtocolPreview(p, targetId = "protocol-preview") {
  const preview = document.getElementById(targetId);
  let html = "";

  html += `<h2>${escapeHtml(p.title)}</h2>`;
  if (p.official_title) html += `<p><em>${escapeHtml(p.official_title)}</em></p>`;

  const metaFields = [
    ["Protocol ID", p.protocol_id],
    ["Sponsor", p.sponsor],
    ["Phase", p.phase],
    ["Conditions", p.conditions],
  ];
  html += '<div class="meta" style="margin:12px 0">';
  metaFields.forEach(([label, val]) => {
    if (val) html += `<span><strong>${label}:</strong> ${escapeHtml(val)}</span>  `;
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

  html += renderList("Locations", p.locations);
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

        if (evt === "sources") {
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

async function plannerDownloadProtocol() {
  const e = plannerEls();
  if (!plannerState.currentProtocol) return;
  e.downloadBtn.disabled = true;
  e.downloadBtn.textContent = "Downloading...";
  try {
    const resp = await fetch(`${API_BASE}/protocol/docx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ protocol: plannerState.currentProtocol }),
    });
    if (!resp.ok) throw new Error("Download failed");
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (plannerState.currentProtocol.protocol_id || "protocol") + ".docx";
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Failed to download: ${err.message}`);
  } finally {
    e.downloadBtn.disabled = false;
    e.downloadBtn.textContent = "Download as Word Document";
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
  e.downloadBtn.addEventListener("click", plannerDownloadProtocol);
  e.topk.addEventListener("input", (evt) => { e.topkVal.textContent = evt.target.value; });
  e.input.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      plannerSend();
    }
  });
})();
