// Hybrid RAG web UI - dependency-free ES module.

const $ = (sel) => document.querySelector(sel);

const SUGGESTIONS = [
  "How many PTO days can I carry over into next year?",
  "How do I roll back a bad deployment?",
  "What is the hotel limit in London?",
  "How quickly must access be revoked after an involuntary termination?",
  "Who updates the status page during a SEV1?",
  "What is the company's stock ticker?",
];

const STATUS_LABEL = {
  supported: "Supported",
  partially_supported: "Partial",
  unsupported: "Unsupported",
  uncited: "Uncited",
};

const state = { citations: [], verification: null, citeStatus: new Map() };

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

// Source documents are hard-wrapped; join soft line breaks but keep lists, tables and code.
function unwrap(text) {
  const out = [];
  let fence = false;
  for (const line of String(text).split("\n")) {
    const t = line.trim();
    if (/^(```|~~~)/.test(t)) { fence = !fence; out.push(line); continue; }
    const prev = out.length ? out[out.length - 1] : "";
    const blockStart = fence || !t || /^([-*+]\s|\d+[.)]\s|\||#|>)/.test(t);
    if (!blockStart && prev.trim() && !/^(\||#|```|~~~)/.test(prev.trim())) {
      out[out.length - 1] = `${prev} ${t}`;
    } else {
      out.push(line);
    }
  }
  return out.join("\n");
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// ---------------------------------------------------------------------------------------------
// Health, config & documents
// ---------------------------------------------------------------------------------------------
async function refreshHealth() {
  const el = $("#status");
  try {
    const h = await api("/health");
    el.className = `status ${h.status === "ok" ? "ok" : "bad"}`;
    $("#status-text").textContent = `${h.providers.llm} · ${h.providers.vector_store} · ${h.providers.reranker}`;
  } catch {
    el.className = "status bad";
    $("#status-text").textContent = "offline";
  }
}

async function refreshConfig() {
  try {
    const cfg = await api("/api/config");
    const sel = $("#department");
    const current = sel.value;
    sel.innerHTML = '<option value="">All</option>' +
      cfg.facets.department.map((d) => `<option>${escapeHtml(d)}</option>`).join("");
    sel.value = cfg.facets.department.includes(current) ? current : "";
    $("#rerank").checked = cfg.reranker !== "none";
    $("#rerank").disabled = cfg.reranker === "none";
  } catch { /* non-fatal */ }
}

async function refreshDocuments() {
  const list = $("#doc-list");
  try {
    const docs = await api("/api/documents");
    const chunks = docs.reduce((n, d) => n + d.num_chunks, 0);
    $("#doc-count").textContent = `${docs.length} document${docs.length === 1 ? "" : "s"}`;
    $("#chunk-count").textContent = `${chunks} chunks`;
    if (!docs.length) {
      list.innerHTML = '<li class="empty">No documents yet. Upload some, or run <code>make seed</code>.</li>';
      return;
    }
    list.innerHTML = docs.map((d) => `
      <li>
        <span class="type">${escapeHtml(d.doc_type === "markdown" ? "md" : d.doc_type)}</span>
        <div class="info">
          <div class="title" title="${escapeHtml(d.title)}">${escapeHtml(d.title)}</div>
          <div class="sub">${escapeHtml(d.metadata.department || d.source)} · ${d.num_chunks} chunks</div>
        </div>
        <button class="icon-btn" data-delete="${escapeHtml(d.doc_id)}" title="Delete ${escapeHtml(d.title)}" aria-label="Delete">&times;</button>
      </li>`).join("");
  } catch (err) {
    list.innerHTML = `<li class="empty">Failed to load documents: ${escapeHtml(err.message)}</li>`;
  }
}

$("#doc-list").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-delete]");
  if (!btn) return;
  const title = btn.closest("li").querySelector(".title").textContent;
  if (!confirm(`Delete "${title}" and all of its chunks?`)) return;
  try {
    await api(`/api/documents/${encodeURIComponent(btn.dataset.delete)}`, { method: "DELETE" });
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
  refreshAll();
});

// Upload ------------------------------------------------------------------------------------
const dropzone = $("#dropzone");
const fileInput = $("#file-input");

async function upload(files) {
  if (!files.length) return;
  const log = $("#upload-log");
  log.innerHTML = `<div>Uploading ${files.length} file(s)…</div>`;
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  try {
    const results = await api("/api/documents", { method: "POST", body: form });
    log.innerHTML = results.map((r) =>
      `<div class="${r.status === "failed" ? "failed" : ""}">${escapeHtml(r.status)} · ${escapeHtml(r.source)}` +
      `${r.error ? ` - ${escapeHtml(r.error)}` : r.status !== "failed" ? ` (${r.num_chunks} chunks)` : ""}</div>`).join("");
  } catch (err) {
    log.innerHTML = `<div class="failed">Upload failed: ${escapeHtml(err.message)}</div>`;
  }
  fileInput.value = "";
  refreshAll();
}

fileInput.addEventListener("change", () => upload([...fileInput.files]));
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
  e.preventDefault(); dropzone.classList.add("drag");
}));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
  e.preventDefault(); dropzone.classList.remove("drag");
}));
dropzone.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));

// ---------------------------------------------------------------------------------------------
// Asking
// ---------------------------------------------------------------------------------------------
function renderSuggestions() {
  $("#suggestions").innerHTML = SUGGESTIONS.map((q) =>
    `<button type="button" class="chip">${escapeHtml(q)}</button>`).join("");
}
$("#suggestions").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  $("#question").value = chip.textContent;
  $("#ask-form").requestSubmit();
});
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});

function renderAnswer(text, streaming) {
  const el = $("#answer");
  el.classList.toggle("streaming", streaming);
  el.innerHTML = escapeHtml(text).replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, group) =>
    group.split(",").map((n) => {
      const idx = Number(n.trim());
      const status = state.citeStatus.get(idx) || "";
      return `<button type="button" class="cite ${status}" data-cite="${idx}" title="${escapeHtml(STATUS_LABEL[status] || status || "Source")} - passage ${idx}">${idx}</button>`;
    }).join(""));
}

$("#answer").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-cite]");
  if (!btn) return;
  const card = document.getElementById(`src-${btn.dataset.cite}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.querySelector(".snippet")?.classList.add("open");
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1600);
});

function computeCiteStatus(verification) {
  const map = new Map();
  if (!verification) return map;
  const agg = new Map();
  for (const claim of verification.claims) {
    for (const c of claim.citations) {
      const a = agg.get(c.index) || { ok: 0, bad: 0, invalid: false };
      if (!c.valid) a.invalid = true;
      else if (c.supported) a.ok += 1;
      else a.bad += 1;
      agg.set(c.index, a);
    }
  }
  for (const [idx, a] of agg) {
    map.set(idx, a.invalid ? "invalid" : a.bad === 0 ? "supported" : a.ok === 0 ? "unsupported" : "partially_supported");
  }
  return map;
}

function bestCheck(idx) {
  const v = state.verification;
  if (!v) return null;
  let best = null;
  for (const claim of v.claims) {
    for (const c of claim.citations) {
      if (c.index === idx && (!best || c.score > best.score)) best = c;
    }
  }
  return best;
}

function renderSources() {
  const card = $("#sources-card");
  const cits = state.citations;
  card.classList.toggle("hidden", !cits.length);
  // Cited passages first, then the rest of the context window.
  const ordered = [...cits].sort((a, b) => (b.cited - a.cited) || (a.index - b.index));
  $("#sources").innerHTML = ordered.map((c) => {
    const status = state.citeStatus.get(c.index);
    const check = bestCheck(c.index);
    const crumb = [...c.heading_path, c.page ? `p. ${c.page}` : null].filter(Boolean).join(" › ");
    const badge = status
      ? `<span class="badge ${status === "supported" ? "ok" : status === "partially_supported" ? "warn" : "bad"}">${escapeHtml(STATUS_LABEL[status] || "Invalid")}${check ? ` · ${check.score.toFixed(2)}` : ""}</span>`
      : c.cited ? '<span class="badge">cited</span>' : '<span class="badge">not cited</span>';
    const evidence = check && check.evidence
      ? `<div class="evidence">Best evidence: <mark>${escapeHtml(check.evidence)}</mark> · lexical ${check.lexical.toFixed(2)} · semantic ${check.semantic.toFixed(2)}${check.numbers_ok ? "" : " · numbers mismatch"}${check.judge === null || check.judge === undefined ? "" : ` · judge ${check.judge ? "yes" : "no"}`}</div>`
      : "";
    return `
      <article class="source ${c.cited ? "cited" : ""}" id="src-${c.index}">
        <div class="source-head">
          <div><span class="cite">${c.index}</span> <span class="source-title">${escapeHtml(c.title)}</span>
            <div class="crumb">${escapeHtml(c.source)}${crumb ? ` › ${escapeHtml(crumb)}` : ""}</div></div>
          <div class="source-meta">${badge}</div>
        </div>
        <div class="snippet" title="Click to expand">${escapeHtml(unwrap(c.snippet))}</div>
        ${evidence}
      </article>`;
  }).join("");
}

$("#sources").addEventListener("click", (e) => {
  e.target.closest(".snippet")?.classList.toggle("open");
});

function renderBadges(extra = {}) {
  const v = state.verification;
  const badges = [];
  if (extra.refused) badges.push('<span class="badge warn">Insufficient context</span>');
  if (v) {
    const g = v.groundedness;
    const cls = g >= 0.75 ? "ok" : g >= 0.5 ? "warn" : "bad";
    badges.push(`<span class="badge ${cls}" title="${escapeHtml(v.method)}">Groundedness ${(g * 100).toFixed(0)}%</span>`);
    badges.push(`<span class="badge">${v.supported_claims}/${v.total_claims} claims supported</span>`);
    if (v.unsupported_citations.length) badges.push(`<span class="badge bad">Unsupported: ${v.unsupported_citations.map((i) => `[${i}]`).join(" ")}</span>`);
    if (v.invalid_citations.length) badges.push(`<span class="badge bad">Invalid: ${v.invalid_citations.map((i) => `[${i}]`).join(" ")}</span>`);
  }
  if (extra.latency_ms) badges.push(`<span class="badge">${(extra.latency_ms / 1000).toFixed(2)} s</span>`);
  if (extra.model) badges.push(`<span class="badge">${escapeHtml(extra.provider)}:${escapeHtml(extra.model)}</span>`);
  $("#badges").innerHTML = badges.join("");
}

function renderClaims() {
  const el = $("#claims");
  const v = state.verification;
  if (!v || !v.claims.length) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  el.innerHTML = v.claims.map((c) => {
    const cls = c.status === "supported" ? "ok" : c.status === "partially_supported" ? "warn" : "bad";
    return `<div class="claim"><span class="badge ${cls}">${STATUS_LABEL[c.status]}</span><span>${escapeHtml(c.claim)}</span><span class="score">${c.score.toFixed(2)}</span></div>`;
  }).join("");
}

function fmt(v, digits = 3) {
  return v === null || v === undefined ? "–" : typeof v === "number" && !Number.isInteger(v) ? v.toFixed(digits) : v;
}

function renderDebug(debug) {
  const card = $("#debug-card");
  if (!debug) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  const t = debug.timings_ms;
  $("#debug-summary").textContent =
    `· ${debug.mode} · dense ${debug.dense_candidates} · bm25 ${debug.bm25_candidates} · fused ${debug.fused_candidates} · rerank ${debug.reranker} · ${fmt(t.total, 1)} ms`;
  const rows = debug.candidates.map((c) => `
    <tr class="${c.selected ? "selected" : ""}">
      <td>${escapeHtml(c.title)}<div class="crumb">${escapeHtml(c.heading_path || "")}</div></td>
      <td>${fmt(c.dense_rank)}</td><td>${fmt(c.dense_score)}</td>
      <td>${fmt(c.bm25_rank)}</td><td>${fmt(c.bm25_score)}</td>
      <td>${fmt(c.rrf_rank)}</td><td>${fmt(c.rrf_score, 4)}</td>
      <td>${fmt(c.rerank_rank)}</td><td>${fmt(c.rerank_score)}</td>
    </tr>`).join("");
  $("#debug-table").innerHTML = `
    <thead><tr><th>Chunk</th><th>Dense #</th><th>Cosine</th><th>BM25 #</th><th>BM25</th>
    <th>RRF #</th><th>RRF</th><th>Rerank #</th><th>Rerank</th></tr></thead><tbody>${rows}</tbody>`;
}

async function* sse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trim());
      }
      if (data.length) yield { event, data: JSON.parse(data.join("\n")) };
    }
  }
}

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  const btn = $("#ask-btn");
  btn.disabled = true;
  btn.textContent = "Thinking…";

  Object.assign(state, { citations: [], verification: null, citeStatus: new Map() });
  $("#answer-card").classList.remove("hidden");
  $("#claims").classList.add("hidden");
  $("#badges").innerHTML = '<span class="badge">Retrieving…</span>';
  renderAnswer("", true);
  renderSources();

  const department = $("#department").value;
  const body = {
    question,
    mode: document.querySelector('input[name="mode"]:checked').value,
    top_k: Number($("#top-k").value),
    rerank: $("#rerank").checked,
    verify: $("#verify").checked,
    filters: department ? { department } : {},
  };

  let text = "";
  try {
    const res = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    for await (const { event, data } of sse(res)) {
      if (event === "retrieval") {
        state.citations = data.citations;
        renderSources();
        renderDebug(data.retrieval);
        $("#badges").innerHTML = '<span class="badge">Generating…</span>';
      } else if (event === "token") {
        text += data.text;
        renderAnswer(text, true);
      } else if (event === "verification") {
        state.verification = data.verification;
        state.citeStatus = computeCiteStatus(data.verification);
        state.citations = data.citations;
        text = data.answer;
        renderAnswer(text, false);
        renderSources();
        renderClaims();
        renderBadges();
      } else if (event === "done") {
        renderAnswer(text, false);
        renderBadges(data);
      } else if (event === "error") {
        throw new Error(data.message);
      }
    }
  } catch (err) {
    renderAnswer(text, false);
    $("#badges").innerHTML = `<span class="badge bad">Error: ${escapeHtml(err.message)}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
});

function refreshAll() {
  refreshDocuments();
  refreshConfig();
  refreshHealth();
}

renderSuggestions();
refreshAll();
setInterval(refreshHealth, 30000);
