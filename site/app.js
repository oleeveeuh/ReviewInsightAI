// reviewinsight showcase — renders frozen data (data.js) into fieldnotes-styled
// charts and tables. plain js + inline svg; no dependencies, no network calls.
const d = window.showcaseData;
const esc = v => String(v).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const INK2 = "#52514e", MUTED = "#77776c", GRID = "#d8d1c4", SAGE = "#89927a",
      SAGE_L = "#dfe1d5", WINE = "#482728", SUMI = "#1e342e";

// vertical bar chart → svg string. items: [{label, value}]
function vbars(items, { color = SAGE, highlight = null, unit = "", W = 340, H = 190 } = {}) {
  const max = Math.max(...items.map(i => i.value));
  const pad = { t: 26, b: 34, l: 8, r: 8 };
  const iw = (W - pad.l - pad.r) / items.length;
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img">`;
  s += `<line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="${GRID}" stroke-width="1"/>`;
  items.forEach((it, i) => {
    const h = (H - pad.t - pad.b) * (it.value / max);
    const x = pad.l + i * iw + iw * 0.18, w = iw * 0.64;
    const y = H - pad.b - h;
    const fill = highlight && highlight(it) ? WINE : color;
    s += `<rect x="${x}" y="${y}" width="${w}" height="${Math.max(h, 1)}" fill="${fill}"/>`;
    s += `<text x="${x + w / 2}" y="${y - 6}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="${INK2}">${it.value}${unit}</text>`;
    s += `<text x="${x + w / 2}" y="${H - pad.b + 15}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="${MUTED}">${esc(it.label)}</text>`;
  });
  return s + "</svg>";
}

// horizontal bar chart → svg string
function hbars(items, { color = SAGE, wineBelow = null, unit = "" } = {}) {
  const max = Math.max(...items.map(i => i.value));
  const rowH = 24, W = 460, H = items.length * rowH + 6;
  const labelW = 128;
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img">`;
  items.forEach((it, i) => {
    const y = i * rowH + 4;
    const bw = (W - labelW - 46) * (it.value / max);
    const fill = wineBelow != null && it.value < wineBelow ? WINE : color;
    s += `<text x="${labelW - 8}" y="${y + 13}" text-anchor="end" font-family="Source Sans 3, sans-serif" font-size="12.5" fill="${INK2}">${esc(it.label)}</text>`;
    s += `<rect x="${labelW}" y="${y}" width="${Math.max(bw, 1)}" height="15" fill="${fill}"/>`;
    s += `<text x="${labelW + Math.max(bw, 1) + 6}" y="${y + 12.5}" font-family="IBM Plex Mono, monospace" font-size="11" fill="${INK2}">${it.value}${unit}</text>`;
  });
  return s + "</svg>";
}

// grouped base-vs-tuned bars for the DPO panel
function dpoBars(metrics) {
  const W = 980, H = 280, pad = { t: 46, b: 52, l: 16, r: 16 };
  const gw = (W - pad.l - pad.r) / metrics.length;
  const max = Math.max(...metrics.flatMap(m => [m.base, m.tuned]));
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img">`;
  s += `<line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="${GRID}"/>`;
  metrics.forEach((m, i) => {
    const x0 = pad.l + i * gw;
    const bw = gw * 0.26, gap = gw * 0.06;
    const bh1 = (H - pad.t - pad.b) * (m.base / max);
    const bh2 = (H - pad.t - pad.b) * (m.tuned / max);
    s += `<rect x="${x0 + gw / 2 - bw - gap / 2}" y="${H - pad.b - bh1}" width="${bw}" height="${bh1}" fill="${SUMI}"/>`;
    s += `<rect x="${x0 + gw / 2 + gap / 2}" y="${H - pad.b - bh2}" width="${bw}" height="${bh2}" fill="${SAGE}"/>`;
    s += `<text x="${x0 + gw / 2 - bw / 2 - gap / 2}" y="${H - pad.b - bh1 - 6}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="${INK2}">${m.base}%</text>`;
    s += `<text x="${x0 + gw / 2 + bw / 2 + gap / 2}" y="${H - pad.b - bh2 - 6}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="${INK2}">${m.tuned}%</text>`;
    const dcol = m.delta > 0.05 ? SAGE : (m.delta < -0.05 ? WINE : MUTED);
    s += `<text x="${x0 + gw / 2}" y="16" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11.5" fill="${dcol}" font-weight="500">${m.delta >= 0 ? "+" : "−"}${Math.abs(m.delta).toFixed(1)} pp</text>`;
    const words = m.name.split(" ");
    const l1 = words.slice(0, 2).join(" "), l2 = words.slice(2).join(" ");
    s += `<text x="${x0 + gw / 2}" y="${H - pad.b + 17}" text-anchor="middle" font-family="Source Sans 3, sans-serif" font-size="12" fill="${INK2}">${esc(l1)}</text>`;
    if (l2) s += `<text x="${x0 + gw / 2}" y="${H - pad.b + 32}" text-anchor="middle" font-family="Source Sans 3, sans-serif" font-size="12" fill="${INK2}">${esc(l2)}</text>`;
  });
  s += `<rect x="${pad.l}" y="${H - 14}" width="10" height="10" fill="${SUMI}"/><text x="${pad.l + 15}" y="${H - 5}" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="${MUTED}">base</text>`;
  s += `<rect x="${pad.l + 62}" y="${H - 14}" width="10" height="10" fill="${SAGE}"/><text x="${pad.l + 77}" y="${H - 5}" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="${MUTED}">lora-dpo tuned</text>`;
  return s + "</svg>";
}

function render() {
  const c = d.corpus, l = d.labels, p = d.preferences;

  // kpis
  document.querySelector("#kpis").innerHTML = [
    [c.n_reviews, "reviews · 3 sources"],
    [Object.keys(d.lopo).length ? 6 : 6, "prompt templates"],
    [525, "silver label rows"],
    [p.n_pairs, "preference pairs"],
    [p.splits.train + "/" + p.splits.eval, "train / eval · by review id"],
    [d.dpo.n_eval_pairs, "dpo held-out pairs"],
  ].map(([n, l2]) => `<div class="kpi"><div class="n">${n}</div><div class="l">${esc(l2)}</div></div>`).join("");

  // corpus charts
  const src = vbars(Object.entries(c.sources).map(([label, value]) => ({ label, value })), { color: SUMI });
  document.querySelector("#c-sources").innerHTML = src;
  document.querySelector("#c-sources-note").textContent =
    `glassdoor ${c.sources.glassdoor} · reddit ${c.sources.reddit} · youtube ${c.sources.youtube} — merged ${c.n_reviews} reviews, ${c.date_range[0]} → ${c.date_range[1]}`;

  const rat = vbars(Object.entries(c.ratings).map(([label, value]) => ({ label: label.replace(".0", "★"), value })));
  document.querySelector("#c-ratings").innerHTML = rat;
  document.querySelector("#c-ratings-note").textContent =
    `mean ${c.mean_rating} / 5 across the ${Object.values(c.ratings).reduce((a, b) => a + b, 0)} glassdoor reviews carrying a rating (reddit and youtube posts have none)`;

  document.querySelector("#c-years").innerHTML = vbars(
    Object.entries(c.years).map(([label, value]) => ({ label, value })),
    { color: SAGE, W: 980, H: 220 });
  document.querySelector("#c-years-note").textContent =
    "collection is clustered in 2024–2025 glassdoor review dates; median length " + c.median_words + " words";

  document.querySelector("#c-themes").innerHTML = hbars(
    Object.entries(l.themes).map(([label, value]) => ({ label, value })), { unit: "" });
  document.querySelector("#c-themes-note").textContent =
    "theme mentions across 175 v5 silver labels (a review can carry up to 3) — workload dominates; " +
    "safety is rare but every mention of it matters. label_type: " + l.label_type;

  document.querySelector("#c-sentiment").innerHTML = vbars(
    Object.entries(l.sentiment).map(([label, value]) => ({ label: label + "★", value })));
  document.querySelector("#c-sentiment-note").textContent =
    "v5 silver sentiment skews mid-scale — 1–5, one label per review";

  document.querySelector("#c-risk").innerHTML = vbars([
    { label: "low", value: l.risk.low }, { label: "medium", value: l.risk.medium },
    { label: "high", value: l.risk.high },
  ], { color: SUMI });
  document.querySelector("#c-risk-note").textContent =
    "retention risk is an llm classification of review text — not observed attrition, and not a prediction about any person";

  // lopo table
  const best = {
    theme_f1: Math.max(...d.lopo.map(r => r.theme_f1)),
    sentiment_agreement: Math.max(...d.lopo.map(r => r.sentiment_agreement)),
    risk_agreement: Math.max(...d.lopo.map(r => r.risk_agreement)),
  };
  document.querySelector("#lopo-body").innerHTML = d.lopo.map(r => {
    const cls = r.theme_f1 < 20 ? " class=\"collapse\"" : "";
    const bestRow = r.theme_f1 === best.theme_f1;
    const cell = (v, k) => `<td class="num"${v === best[k] ? " style=\"color:#89927a;font-weight:600\"" : ""}>${v.toFixed(1)}%</td>`;
    return `<tr${bestRow ? " class=\"best\"" : ""}><td>${esc(r.prompt)}</td>${cell(r.theme_f1, "theme_f1")}${cell(r.sentiment_agreement, "sentiment_agreement")}${cell(r.risk_agreement, "risk_agreement")}<td class="num">${r.label_rows}</td><td class="num">${r.unique_reviews}</td></tr>`;
  }).join("");
  document.querySelector("#lopo-note").textContent =
    `leave-one-prompt-out over ${d.lopo[0].label_rows} silver label rows (${d.lopo[0].unique_reviews} unique reviews) — the tested prompt's own labels are excluded before scoring. ` +
    `${d.lopo_meta.n_calls} gpt-4o-mini calls at temperature 0.3, ${d.lopo_meta.failures} failures. sage = column leader · wine = the theme-f1 collapse. ` +
    "these are agreement with an llm labeler, not accuracy — absolute numbers are structurally optimistic.";

  // dpo panel
  document.querySelector("#dpo-bars").innerHTML = dpoBars(d.dpo.metrics);
  document.querySelector("#dpo-note").textContent =
    `base vs lora-dpo-tuned ${d.dpo.base_model}, scored on ${d.dpo.n_eval_pairs} review-disjoint held-out pairs against consensus labels; ` +
    `${d.dpo.n_unparsed} parse failures. n = 27 → roughly ±18 pp on any proportion — the honest reading is the +33.3 pp risk delta and nothing finer. ` +
    "agreement with an llm consensus labeler, not accuracy.";
  document.querySelector("#dpo-construction").innerHTML =
    `<span class="chip"><b>525</b>silver labels</span><span class="chip"><b>137</b>preference pairs · consensus = chosen, outlier = rejected</span>` +
    `<span class="chip"><b>38</b>unanimous reviews skipped</span><span class="chip"><b>110 / 27</b>split by review id</span>` +
    `<span class="chip"><b>26</b>optimizer steps · loss 0.693 → 0.669</span><span class="chip"><b>16%</b>of pairs review-truncated to 300 tokens</span>`;
}

render();
