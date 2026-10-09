// Renders the cleaning report from data/site.json.
// Every number on the page is filled in here from that file.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const get = (o, path) => path.split(".").reduce((x, k) => (x == null ? undefined : x[k]), o);
const nf = new Intl.NumberFormat("en-US");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const SRC = { glaive: "Glaive FC v2", anudesh: "Anudesh (IndicAlign)" };
const SHORT_NAMES = { extract: "1 Extract", normalize: "2 Normalize", format: "3 Format", langid: "4 Lang ID",
  quality: "5 Quality", dedup: "6 Dedup", pii: "7 PII", decontam: "8 Decontam", manifest: "9 Manifest" };
const STAGE_NAMES = { extract: "1 Extract", normalize: "2 Normalize", format: "3 Format", langid: "4 Language ID",
  quality: "5 Quality", dedup: "6 Dedup", pii: "7 PII", decontam: "8 Decontam", manifest: "9 Manifest" };

function compact(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(0) + "K";
  return nf.format(n);
}
function fmt(v, kind) {
  if (v == null) return "–";
  switch (kind) {
    case "m": return (v / 1e6).toFixed(1) + "M";
    case "c": return compact(v);
    case "pct": return Number(v).toFixed(1) + "%";
    case "pct2": return Number(v).toFixed(2) + "%";
    case "rate": return Math.round(Number(v) * 100) + "%";
    case "raw": return String(v);
    case "short": return String(v).slice(0, 12);
    default: return typeof v === "number" ? nf.format(v) : String(v);
  }
}

function bind(data) {
  $$("[data-k]").forEach((el) => { el.textContent = fmt(get(data, el.dataset.k), el.dataset.fmt); });
}

// ---------- tooltip shared by all charts
const tip = document.createElement("div");
tip.className = "tip";
document.body.appendChild(tip);
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const pad = 12, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = x + window.scrollX + "px";
  tip.style.top = y + window.scrollY + "px";
}
function hideTip() { tip.classList.remove("on"); }

const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
// bar with 4px rounded data-end, square at the baseline (horizontal bars grow to the right)
function hbarPath(x, y, w, h, r = 4) {
  r = Math.min(r, w, h / 2);
  if (w <= 0) return "";
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}
function vbarPath(x, y0, w, h, r = 4) {
  r = Math.min(r, h, w / 2);
  if (h <= 0) return "";
  const y = y0 - h;
  return `M${x},${y0}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y0}Z`;
}

// ---------- waterfall: tokens surviving after each stage, one bar per source
function waterfall(data) {
  const host = $("#waterfall");
  if (!host) return;
  const rows = [{ stage: "input", glaive_tokens: data.input.glaive_tokens, anudesh_tokens: data.input.anudesh_tokens,
    glaive_docs: data.input.glaive_docs, anudesh_docs: data.input.anudesh_docs, glaive_removed_docs: 0, anudesh_removed_docs: 0 },
    ...data.waterfall];
  host.querySelectorAll("svg").forEach((x) => x.remove());
  const W = Math.max(300, Math.round(host.clientWidth || 720)), narrow = W < 520;
  const labelW = narrow ? 88 : 110, valW = narrow ? 46 : 64, barH = 11, gap = 2, rowH = barH * 2 + gap + 12;
  const H = rows.length * rowH + 24;
  const max = Math.max(...rows.map((r) => Math.max(r.glaive_tokens, r.anudesh_tokens)));
  const x0 = labelW, plotW = W - labelW - valW;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Tokens surviving after each stage, per source" });
  for (let t = 0; t <= 4; t++) {
    const gx = x0 + (plotW * t) / 4;
    el("line", { x1: gx, x2: gx, y1: 0, y2: H - 20, class: "grid" }, svg);
    const tx = el("text", { x: gx, y: H - 6, "text-anchor": t === 0 ? "start" : "middle" }, svg);
    tx.textContent = compact((max * t) / 4);
  }
  rows.forEach((r, i) => {
    const y = i * rowH + 4;
    const lab = el("text", { x: 0, y: y + barH + 4, class: "lbl" }, svg);
    lab.textContent = r.stage === "input" ? "Raw input" : (narrow ? SHORT_NAMES[r.stage] : STAGE_NAMES[r.stage]);
    [["glaive", "bar-g", 0], ["anudesh", "bar-a", 1]].forEach(([s, cls, k]) => {
      const v = r[`${s}_tokens`], w = (plotW * v) / max, by = y + k * (barH + gap);
      el("path", { d: hbarPath(x0, by, w, barH), class: cls }, svg);
      const t = el("text", { x: x0 + w + 6, y: by + barH - 2, class: "val" }, svg);
      t.textContent = compact(v);
      const hit = el("rect", { x: 0, y: by - 1, width: W, height: barH + 2, class: "hit" }, svg);
      const html = `<b>${r.stage === "input" ? "Raw input" : STAGE_NAMES[r.stage]}</b> · ${SRC[s]}<br>` +
        `${nf.format(v)} tokens · ${nf.format(r[`${s}_docs`])} conversations` +
        (r[`${s}_removed_docs`] ? `<br>removed here: ${nf.format(r[`${s}_removed_docs`])}` : "") +
        (r[`${s}_modified_docs`] ? `<br>modified here: ${nf.format(r[`${s}_modified_docs`])}` : "");
      hit.addEventListener("mousemove", (e) => showTip(e, html));
      hit.addEventListener("mouseleave", hideTip);
    });
  });
  el("line", { x1: x0, x2: x0, y1: 0, y2: H - 20, class: "base" }, svg);
  host.appendChild(svg);
  // table view
  const tv = $("#waterfall-table");
  if (tv) {
    tv.innerHTML = `<table><thead><tr><th>After stage</th><th class="num">Glaive convs</th><th class="num">Glaive tokens</th>` +
      `<th class="num">Anudesh convs</th><th class="num">Anudesh tokens</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${r.stage === "input" ? "Raw input" : STAGE_NAMES[r.stage]}</td>` +
        `<td class="num">${nf.format(r.glaive_docs)}</td><td class="num">${nf.format(r.glaive_tokens)}</td>` +
        `<td class="num">${nf.format(r.anudesh_docs)}</td><td class="num">${nf.format(r.anudesh_tokens)}</td></tr>`).join("") +
      `</tbody></table>`;
  }
}

// ---------- removals by reason for one stage
const REASON_TEXT = {
  null_or_empty_turn: "a prompt or response is null/empty",
  empty_chat: "empty conversation",
  unrecoverable_function_call: "function call can't be repaired (truncated or not JSON)",
  tool_response_truncated: "function response cut off mid-JSON",
  tool_response_invalid_json: "function response is not valid JSON",
  residual_ghost_marker_in_content: "role marker leaked inside message text",
  no_complete_exchange: "no complete user → assistant exchange",
  user_language_outside_source_claim: "user language not a target language",
  assistant_language_outside_source_claim: "reply language not a target language",
  response_in_different_language: "reply in a different language",
  greeting_only_prompt: "prompt is only a greeting",
  empty_assistant_turn: "empty assistant reply",
  degenerate_repetition: "reply stuck repeating itself (first version)",
  looping_output: "reply stuck in a loop",
  repeated_lines: "reply repeats the same lines",
  latex_backslashes_stripped: "LaTeX in the reply lost its backslashes upstream",
  low_classifier_score: "classifier score below threshold",
  exact_duplicate: "exact duplicate",
  exact_duplicate_after_masking: "became an exact duplicate once PII was masked",
  near_duplicate: "near duplicate (Jaccard ≥ threshold)",
  eval_overlap: "shares a 13-gram with a benchmark test item",
};
function removals(data) {
  $$("[data-removals]").forEach((host) => {
    const st = data.stages[host.dataset.removals];
    const reasons = new Set();
    Object.values(st.removed).forEach((o) => Object.keys(o).forEach((k) => reasons.add(k)));
    if (!reasons.size) { host.innerHTML = `<p class="small">Nothing removed at this stage.</p>`; return; }
    const rowsHtml = [...reasons].sort().map((r) => {
      const g = st.removed.glaive?.[r] || 0, a = st.removed.anudesh?.[r] || 0;
      const gt = st.removed_tokens.glaive?.[r] || 0, at = st.removed_tokens.anudesh?.[r] || 0;
      return `<tr><td>${esc(REASON_TEXT[r] || r)}<br><span class="small mono">${esc(r)}</span></td>` +
        `<td class="num">${nf.format(g)}</td><td class="num">${nf.format(a)}</td><td class="num">${compact(gt + at)}</td></tr>`;
    }).join("");
    host.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Removed because</th><th class="num">Glaive</th>` +
      `<th class="num">Anudesh</th><th class="num">Tokens</th></tr></thead><tbody>${rowsHtml}</tbody></table></div>`;
  });
}

// ---------- widget-4 audit: fail rate per rule, Latin vs Indic answers
function filterBias(data) {
  const host = $("#filter-bias");
  if (!host) return;
  const rules = ["meanlen", "symratio", "termpunc", "duplines", "repeat", "stopwords", "bullets", "ellipsis", "wordcount"];
  const names = { meanlen: "mean word length", symratio: "symbol ratio", termpunc: "ends in . ! ?", duplines: "duplicate lines",
    repeat: "char repetition", stopwords: "≥2 stop words", bullets: "bullet lines", ellipsis: "ellipsis lines", wordcount: "50+ words" };
  // within Anudesh only: same prompts, same model, the only difference is the script of the answer
  const c = data.stages.quality.counters;
  const sum = (key) => c.anudesh?.[key] || 0;
  const nLat = sum("docs__latin_or_other"), nInd = sum("docs__indic");
  const rows = rules.map((r) => ({ r, lat: (100 * sum(`widget4_fail_${r}__latin_or_other`)) / Math.max(1, nLat),
    ind: (100 * sum(`widget4_fail_${r}__indic`)) / Math.max(1, nInd) }));
  host.querySelectorAll("svg").forEach((x) => x.remove());
  const W = Math.max(300, Math.round(host.clientWidth || 720)), narrow = W < 520;
  const labelW = narrow ? 112 : 150, valW = 40, bh = 9, gap = 2, rowH = bh * 2 + gap + 10, H = rows.length * rowH + 22;
  const x0 = labelW, plotW = W - labelW - valW;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share of answers failing each heuristic quality rule" });
  for (let t = 0; t <= 4; t++) {
    const gx = x0 + (plotW * t) / 4;
    el("line", { x1: gx, x2: gx, y1: 0, y2: H - 18, class: "grid" }, svg);
    const tx = el("text", { x: gx, y: H - 4, "text-anchor": t === 0 ? "start" : "middle" }, svg);
    tx.textContent = `${25 * t}%`;
  }
  rows.forEach((row, i) => {
    const y = i * rowH + 3;
    el("text", { x: 0, y: y + bh + 3, class: "lbl" }, svg).textContent = names[row.r];
    [["lat", "bar-l", 0, "Latin-script answers", nLat], ["ind", "bar-i", 1, "Indic-script answers", nInd]].forEach(([k, cls, j, label, n]) => {
      const v = row[k], w = (plotW * v) / 100, by = y + j * (bh + gap);
      el("path", { d: hbarPath(x0, by, Math.max(w, 0.01), bh, 3), class: cls }, svg);
      el("text", { x: x0 + w + 5, y: by + bh - 1, class: "val" }, svg).textContent = `${v.toFixed(0)}%`;
      const hit = el("rect", { x: 0, y: by - 1, width: W, height: bh + 2, class: "hit" }, svg);
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${names[row.r]}</b><br>${label}: ${v.toFixed(1)}% fail (n=${nf.format(n)})`));
      hit.addEventListener("mouseleave", hideTip);
    });
  });
  el("line", { x1: x0, x2: x0, y1: 0, y2: H - 18, class: "base" }, svg);
  host.appendChild(svg);
  const tv = $("#filter-bias-table");
  if (tv) tv.innerHTML = `<table><thead><tr><th>Rule</th><th class="num">Latin-script fail %</th><th class="num">Indic-script fail %</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${names[r.r]}</td><td class="num">${r.lat.toFixed(1)}</td><td class="num">${r.ind.toFixed(1)}</td></tr>`).join("") + `</tbody></table>`;
}

// ---------- dedup: Jaccard of LSH candidate pairs
function jaccardHist(data) {
  const host = $("#jaccard-hist");
  if (!host) return;
  const v = data.dedup.verified_jaccard_hist, r = data.dedup.rejected_jaccard_hist;
  const bins = Object.keys(v);
  host.querySelectorAll("svg").forEach((x) => x.remove());
  const W = Math.max(300, Math.round(host.clientWidth || 720)), H = 220, x0 = 46, y0 = H - 30, plotW = W - x0 - 10, plotH = y0 - 10;
  const max = Math.max(...bins.map((b) => (v[b] || 0) + (r[b] || 0)), 1);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Jaccard similarity of candidate pairs" });
  for (let t = 0; t <= 4; t++) {
    const gy = y0 - (plotH * t) / 4;
    el("line", { x1: x0, x2: W - 10, y1: gy, y2: gy, class: "grid" }, svg);
    el("text", { x: x0 - 6, y: gy + 4, "text-anchor": "end" }, svg).textContent = compact(Math.round((max * t) / 4));
  }
  const bw = plotW / bins.length;
  bins.forEach((b, i) => {
    const x = x0 + i * bw + bw * 0.2, w = Math.min(24, bw * 0.6);
    const hr = (plotH * (r[b] || 0)) / max, hv = (plotH * (v[b] || 0)) / max;
    // stacked: rejected at the bottom, verified on top, 2px surface gap
    if (hr > 0) el("path", { d: vbarPath(x, y0, w, hr, hv > 0 ? 0 : 3), class: "bar-rej" , style: "fill: var(--baseline)" }, svg);
    if (hv > 0) el("path", { d: vbarPath(x, y0 - hr - (hr > 0 ? 2 : 0), w, hv, 3), class: "bar-g" }, svg);
    el("text", { x: x + w / 2, y: y0 + 14, "text-anchor": "middle" }, svg).textContent = b.split("-")[0];
    const hit = el("rect", { x: x0 + i * bw, y: 0, width: bw, height: y0, class: "hit" }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, `<b>Jaccard ${b}</b><br>removed as duplicate: ${nf.format(v[b] || 0)}<br>candidate, kept: ${nf.format(r[b] || 0)}`));
    hit.addEventListener("mouseleave", hideTip);
  });
  el("line", { x1: x0, x2: W - 10, y1: y0, y2: y0, class: "base" }, svg);
  host.appendChild(svg);
}

// ---------- examples
function showInvisible(s) {
  // make joiners and removed characters visible as chips
  return esc(s)
    .replace(/\u200d/g, '<span class="cp keep" title="ZWJ U+200D kept">ZWJ</span>')
    .replace(/\u200c/g, '<span class="cp keep" title="ZWNJ U+200C kept">ZWNJ</span>')
    .replace(/\u200b/g, '<span class="cp drop" title="zero-width space U+200B">ZWSP</span>')
    .replace(/\ufeff/g, '<span class="cp drop" title="byte-order mark U+FEFF">BOM</span>');
}
function examples(data) {
  const ex = data.examples || {};
  $$("[data-example]").forEach((host) => {
    const [group, key, idx] = host.dataset.example.split(":");
    let item = ex[group]?.[key];
    if (Array.isArray(item)) item = item[Number(idx || 0)];
    if (!item) { host.innerHTML = `<p class="small">(no example of this kind in the data)</p>`; return; }
    const mode = host.dataset.mode || "ba";
    if (mode === "ba") {
      host.innerHTML = `<div class="ba"><div><div class="cap">before</div><pre class="snip">${showInvisible(item.before)}</pre></div>` +
        `<div><div class="cap">after</div><pre class="snip">${showInvisible(item.after_doc_excerpt)}</pre></div></div>` +
        `<p class="small">${esc(item.source)} row ${item.src_index}</p>`;
    } else if (mode === "pair") {
      host.innerHTML = `<p class="small">Jaccard ${item.jaccard} (MinHash estimate ${item.minhash_estimate}) · ${esc(item.a[0])} #${item.a[1]} vs ${esc(item.b[0])} #${item.b[1]}</p>` +
        `<div class="ba"><div><div class="cap">kept</div><pre class="snip">${esc(item.a_text)}</pre></div><div><div class="cap">removed</div><pre class="snip">${esc(item.b_text)}</pre></div></div>`;
    } else if (mode === "qa") {
      host.innerHTML = `<div class="ba"><div><div class="cap">user</div><pre class="snip">${esc(item.user)}</pre></div>` +
        `<div><div class="cap">assistant${item.detail ? " · " + esc(item.detail) : ""}</div><pre class="snip">${esc(item.assistant)}</pre></div></div>` +
        `<p class="small">${esc(item.source)} row ${item.src_index}</p>`;
    } else if (mode === "pii") {
      host.innerHTML = `<pre class="snip">${esc(item.masked_excerpt)}</pre><p class="small">${esc(item.source)} row ${item.src_index} · masked: ${esc(item.kinds.join(", "))}</p>`;
    } else if (mode === "format") {
      host.innerHTML = `<div class="ba"><div><div class="cap">as stored upstream</div><pre class="snip">${esc(item.raw_chat)}</pre></div>` +
        `<div><div class="cap">canonical (one special token per turn)</div><pre class="snip">${esc(item.canonical)}</pre></div></div>`;
    } else if (mode === "decontam") {
      host.innerHTML = `<pre class="snip">${esc(item.user)}</pre><p class="small">${esc(item.source)} row ${item.src_index} · matched ${esc(item.hits.map((h) => `${h[0]} test item ${h[1]}: ${h[2]} shared 13-grams, ${Math.round(100 * (h[3] || 0))}% of the item`).join("; "))}</p>`;
    }
  });
}

function fpTable(data) {
  const host = $("#fp-table");
  const a = data.false_positive_audit;
  if (!host || !a) return;
  const rows = a.files.map((f) => `<tr><td>${esc(REASON_TEXT[f.reason] || f.reason)}<br><span class="small mono">${esc(f.stage)}</span></td>` +
    `<td class="num">${f.sampled}</td><td class="num">${f.correct}</td><td class="num">${f.false_positive}</td>` +
    `<td class="num">${f.unclear}</td><td class="small">${esc(f.action)}</td></tr>`).join("");
  const p = a.pii_sample;
  host.innerHTML = `<table><thead><tr><th>Removed because</th><th class="num">Read</th><th class="num">Right</th>` +
    `<th class="num">Wrong</th><th class="num">Unclear</th><th>What I did</th></tr></thead><tbody>${rows}` +
    `<tr><td>PII values masked (sample)</td><td class="num">${p.masked_values_reviewed}</td>` +
    `<td class="num">${p.personal + p.synthetic_realistic}</td><td class="num">${p.not_pii}</td><td class="num">0</td>` +
    `<td class="small">${esc(p.action)}. "Right" here means real contacts (${p.personal}) or realistic fake ones (${p.synthetic_realistic}).</td></tr>` +
    `</tbody></table>`;
}

function manifestView(data) {
  const host = $("#manifest-json");
  if (!host) return;
  const sh = data.manifest.shards.map((s) => ({ ...s, cleaning_scripts: s.cleaning_scripts.map((c) => `${c.name} sha256:${c.sha256.slice(0, 12)}…`) }));
  host.textContent = JSON.stringify({ corpus_id: data.manifest.corpus_id, shards: sh }, null, 1);
}

function themeToggle() {
  const b = $("#theme");
  if (!b) return;
  const cur = () => document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const label = () => { b.textContent = cur() === "dark" ? "Light" : "Dark"; };
  try { const s = localStorage.getItem("theme"); if (s) document.documentElement.dataset.theme = s; } catch (e) { /* storage blocked */ }
  label();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", label);
  b.addEventListener("click", () => {
    const next = cur() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    label();
  });
}

fetch("data/site.json")
  .then((r) => r.json())
  .then((data) => {
    bind(data);
    const charts = () => { waterfall(data); filterBias(data); jaccardHist(data); };
    charts();
    let lastW = innerWidth, timer = null;
    addEventListener("resize", () => {
      if (innerWidth === lastW) return;
      lastW = innerWidth;
      clearTimeout(timer);
      timer = setTimeout(charts, 150);
    });
    removals(data);
    examples(data);
    manifestView(data);
    fpTable(data);
    document.body.dataset.ready = "1";
  })
  .catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p style="padding:16px">Could not load data/site.json: ${esc(e)}</p>`); });
themeToggle();
