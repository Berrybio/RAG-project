const API_BASE = "/api";

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
      <div class="nct-id">${escapeHtml(source.nct_id)} <span class="score">${(source.score * 100).toFixed(1)}%</span></div>
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
      evtSource.close();
      btn.disabled = false;
      btn.textContent = "Search";
    });

    evtSource.addEventListener("error", (e) => {
      evtSource.close();
      btn.disabled = false;
      btn.textContent = "Search";
      answerContent.textContent = "An error occurred while searching. Please try again.";
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
function renderProtocolPreview(p) {
  const preview = document.getElementById("protocol-preview");
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
