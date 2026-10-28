const $ = (id) => document.getElementById(id);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ========================= Theme ========================= */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const toggle = $("theme-toggle");
  const moonIcon = toggle?.querySelector(".icon-moon");
  const sunIcon = toggle?.querySelector(".icon-sun");
  const label = $("theme-toggle-label");
  if (theme === "light") {
    moonIcon?.classList.add("hidden");
    sunIcon?.classList.remove("hidden");
    if (label) label.textContent = "Light mode";
    toggle?.setAttribute("aria-pressed", "true");
  } else {
    sunIcon?.classList.add("hidden");
    moonIcon?.classList.remove("hidden");
    if (label) label.textContent = "Dark mode";
    toggle?.setAttribute("aria-pressed", "false");
  }
}
(function initTheme() {
  let saved = "dark";
  try { saved = localStorage.getItem("koa_theme") || "dark"; } catch (e) { /* ignore */ }
  applyTheme(saved);
})();
$("theme-toggle")?.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("koa_theme", next); } catch (e) { /* ignore */ }
});

/* ========================= View switching ========================= */
const viewNames = ["analysis", "causal"];
function openView(name) {
  viewNames.forEach((view) => {
    const element = $("view-" + view);
    if (element) element.classList.toggle("hidden", view !== name);
  });
  document.querySelectorAll(".nav").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  window.scrollTo({ top: 0, behavior: prefersReducedMotion ? "auto" : "smooth" });
  if (name === "causal" && !window.__causalLoaded) {
    window.__causalLoaded = true;
    loadCausalInsights();
  }
}
document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => openView(button.dataset.view));
});

/* ========================= Toast ========================= */
function toast(message) {
  const element = $("toast");
  if (!element) return;
  element.textContent = message;
  element.style.display = "block";
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => { element.style.display = "none"; }, 3200);
}

/* ========================= Zoom modal ========================= */
function openZoom(src, alt) {
  const modal = $("zoom-modal");
  const img = $("modal-img");
  if (!modal || !img) return;
  img.src = src;
  img.alt = alt || "Enlarged view";
  modal.classList.remove("hidden");
}
function closeZoom() {
  $("zoom-modal")?.classList.add("hidden");
  if ($("modal-img")) $("modal-img").src = "";
}
$("modal-close")?.addEventListener("click", closeZoom);
$("zoom-modal")?.addEventListener("click", (e) => { if (e.target.id === "zoom-modal") closeZoom(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeZoom(); });

/* ========================= X-ray selection (upload / drag-drop / demo) ========================= */
let selectedImageUrl = null;
let selectedFile = null;

function acceptFile(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    toast("Choose a PNG or JPEG image.");
    return;
  }
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedFile = file;
  selectedImageUrl = URL.createObjectURL(file);
  if ($("xray-preview")) $("xray-preview").src = selectedImageUrl;
  if ($("image-name")) $("image-name").textContent = file.name;
  $("image-view")?.classList.remove("hidden");
  $("empty-image")?.classList.add("hidden");
  resetResults();
  toast("Image loaded. Run the assessment when ready.");
}

$("xray-input")?.addEventListener("change", (event) => {
  acceptFile(event.target.files?.[0]);
});

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((evt) => {
  dropzone?.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone?.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag-over"); });
});
dropzone?.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) acceptFile(file);
});

$("remove-image")?.addEventListener("click", () => {
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = null;
  selectedFile = null;
  if ($("xray-input")) $("xray-input").value = "";
  $("xray-preview")?.removeAttribute("src");
  $("image-view")?.classList.add("hidden");
  $("empty-image")?.classList.remove("hidden");
  resetResults();
});

$("xray-preview")?.addEventListener("click", () => {
  if (selectedImageUrl) openZoom(selectedImageUrl, "Selected knee X-ray, enlarged");
});

/* Demo samples: one real test image per KL grade, so the demo works with no file hunting. */
const DEMO_SAMPLES = [
  { grade: 0, file: "sample_kl0.png", label: "KL0" },
  { grade: 1, file: "sample_kl1.png", label: "KL1" },
  { grade: 2, file: "sample_kl2.png", label: "KL2" },
  { grade: 3, file: "sample_kl3.png", label: "KL3" },
  { grade: 4, file: "sample_kl4.png", label: "KL4" },
];
(function buildDemoRow() {
  const row = $("demo-row");
  if (!row) return;
  DEMO_SAMPLES.forEach((sample) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "demo-chip";
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 3l14 9-14 9V3z"/></svg> ${sample.label} sample`;
    btn.addEventListener("click", async () => {
      try {
        const resp = await fetch(`/static/samples/${sample.file}`);
        const blob = await resp.blob();
        const file = new File([blob], sample.file, { type: "image/png" });
        acceptFile(file);
      } catch (e) {
        toast("Could not load the sample image.");
      }
    });
    row.appendChild(btn);
  });
})();

function resetResults() {
  $("download-report-btn")?.setAttribute("disabled", "true");
  $("progress-list")?.classList.add("hidden");
  if ($("progress-list")) $("progress-list").innerHTML = "";
  const jswBadge = $("jsw-status-badge");
  if (jswBadge) { jswBadge.textContent = "NOT RUN"; jswBadge.className = "tag muted"; }
}

/* ========================= Grade reference data ========================= */
const GRADE_LABELS = ["None", "Doubtful", "Minimal", "Moderate", "Severe"];
const GRADE_DESCRIPTIONS = [
  "No radiographic signs of osteoarthritis.",
  "Doubtful — a minute osteophyte of uncertain significance.",
  "Minimal — definite osteophytes; joint space narrowing possible.",
  "Moderate — multiple osteophytes, definite joint-space narrowing, some sclerosis.",
  "Severe — large osteophytes, marked joint-space narrowing, severe sclerosis.",
];
const GRADE_COLOR_VARS = ["--kl0", "--kl1", "--kl2", "--kl3", "--kl4"];
const MEMBER_LABELS = {
  kl_final: "ConvNeXt (kl_final)",
  kl_final_seed7: "ConvNeXt (kl_final)",
  hybrid1: "Transformer hybrid",
  kneenext_seed42: "KneeNeXt",
  kneenext_seed7: "KneeNeXt",
};
function memberLabel(key) { return MEMBER_LABELS[key] || key; }

/* ========================= Analysis progress animation (visual only) ========================= */
const PROGRESS_STEPS = [
  "Enhancing contrast (CLAHE)",
  "Running model A",
  "Running model B",
  "Combining results",
  "Generating explanations",
  "Causal analysis",
];
function runProgressAnimation() {
  const list = $("progress-list");
  if (!list) return { finish: () => {} };
  list.classList.remove("hidden");
  list.innerHTML = PROGRESS_STEPS.map((label, i) =>
    `<li data-i="${i}"><span class="dot"></span><span>${label}</span></li>`).join("");
  const items = Array.from(list.children);
  let i = 0;
  function tick() {
    if (i > 0) items[i - 1].classList.replace("active", "done");
    if (i < items.length - 1) {
      items[i].classList.add("active");
      i += 1;
      timer = setTimeout(tick, prefersReducedMotion ? 0 : 480);
    } else {
      items[i].classList.add("active");
    }
  }
  let timer = setTimeout(tick, 0);
  $("image-view")?.classList.add("scanning");
  return {
    finish(success) {
      clearTimeout(timer);
      items.forEach((el) => { el.classList.remove("active"); el.classList.add(success ? "done" : "active"); });
      $("image-view")?.classList.remove("scanning");
    },
  };
}

/* ========================= Rendering: prediction ========================= */
function renderPrediction(data) {
  const panel = $("prediction-panel");
  if (!panel) return;
  panel.replaceChildren();
  panel.classList.remove("prediction-empty");

  const top = document.createElement("div");
  top.className = "result-top";

  const badge = document.createElement("div");
  badge.className = "grade-badge";
  badge.style.borderColor = `var(${GRADE_COLOR_VARS[data.grade]})`;
  badge.innerHTML = `<span class="num">${data.grade}</span><span class="lbl">KL GRADE</span>`;

  const ring = document.createElement("div");
  ring.className = "conf-ring-wrap";
  const circumference = 2 * Math.PI * 36;
  ring.innerHTML = `
    <svg viewBox="0 0 80 80">
      <circle class="ring-bg" cx="40" cy="40" r="36"/>
      <circle class="ring-fg" cx="40" cy="40" r="36" style="stroke-dasharray:${circumference};stroke-dashoffset:${circumference}"/>
    </svg>
    <div class="ring-label">${Math.round(data.confidence * 100)}%</div>`;

  const text = document.createElement("div");
  text.className = "result-text";
  const memberNames = (data.model_members || []).map(memberLabel);
  const modelLine = data.model_used === "ensemble"
    ? `Final model: ensemble of ${memberNames.join(" and ")}.`
    : `Final model: ${memberNames[0] || "ConvNeXt (kl_final)"}.`;
  text.innerHTML = `
    <b>KL ${data.grade} — ${GRADE_LABELS[data.grade] || ""}</b>
    <p class="grade-desc">${GRADE_DESCRIPTIONS[data.grade] || ""}</p>
    <p class="model-line">${modelLine}</p>`;

  top.append(badge, ring, text);
  panel.appendChild(top);

  requestAnimationFrame(() => {
    const offset = circumference * (1 - data.confidence);
    const fg = ring.querySelector(".ring-fg");
    if (fg) fg.style.strokeDashoffset = String(offset);
  });

  const scale = document.createElement("div");
  scale.className = "severity-scale";
  scale.innerHTML = `
    <div class="severity-track"><span></span><span></span><span></span><span></span><span></span></div>
    <div class="severity-labels"><span>KL0</span><span>KL1</span><span>KL2</span><span>KL3</span><span>KL4</span></div>
    <div class="severity-marker" id="severity-marker"></div>`;
  panel.appendChild(scale);
  requestAnimationFrame(() => {
    const marker = scale.querySelector("#severity-marker");
    if (marker) marker.style.left = `${(data.grade + 0.5) * 20}%`;
  });

  const bars = document.createElement("div");
  bars.className = "prob-bars";
  Object.entries(data.probabilities).forEach(([label, value], idx) => {
    const pct = Math.max(0, Math.min(100, value * 100));
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `<span class="k-label">${label}</span><div class="track"><div class="fill" style="background:var(${GRADE_COLOR_VARS[idx]})"></div></div><span class="pct">${pct.toFixed(1)}%</span>`;
    bars.appendChild(row);
    requestAnimationFrame(() => {
      const fill = row.querySelector(".fill");
      if (fill) fill.style.width = `${pct}%`;
    });
  });
  panel.appendChild(bars);

  const banner = document.createElement("div");
  const isUncertain = !!data.uncertainty?.flag;
  banner.className = `uncertainty-banner ${isUncertain ? "warn" : "ok"}`;
  banner.innerHTML = isUncertain
    ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2 1 21h22L12 2z"/><path d="M12 9v5M12 17v.01"/></svg><span>Low confidence — refer to a radiologist.</span>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg><span>Confident prediction.</span>`;
  panel.appendChild(banner);

  const tag = $("model-used-tag");
  if (tag) { tag.textContent = "ASSESSED"; tag.classList.remove("muted"); }
  const hint = $("model-line-hint");
  if (hint) hint.textContent = modelLine;

  $("download-report-btn")?.removeAttribute("disabled");
}

/* ========================= Rendering: explainability viewer ========================= */
function renderXai(data) {
  const el = $("xai-panel");
  if (!el || !selectedImageUrl) return;
  el.replaceChildren();

  const methods = [];
  if (data.integrated_gradients_png_base64) methods.push({ key: "ig", label: "Integrated Gradients", src: `data:image/png;base64,${data.integrated_gradients_png_base64}` });
  if (data.gradcam_png_base64) methods.push({ key: "cam", label: "Grad-CAM", src: `data:image/png;base64,${data.gradcam_png_base64}` });
  if (methods.length === 0) {
    el.innerHTML = `<div class="empty-state"><h3>No explanation available</h3><p>Explanation images could not be generated for this image.</p></div>`;
    return;
  }

  const tabs = document.createElement("div");
  tabs.className = "xai-tabs";
  methods.forEach((m, i) => {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = `xai-tab ${i === 0 ? "active" : ""}`;
    tab.textContent = m.label;
    tab.dataset.key = m.key;
    tabs.appendChild(tab);
  });
  el.appendChild(tabs);

  const viewer = document.createElement("div");
  viewer.className = "xai-viewer";
  const baseImg = document.createElement("img");
  baseImg.className = "layer base";
  baseImg.src = selectedImageUrl;
  baseImg.alt = "Original knee X-ray";
  const overlayImg = document.createElement("img");
  overlayImg.className = "layer overlay";
  overlayImg.alt = "Explanation heatmap overlay";
  overlayImg.src = methods[0].src;
  overlayImg.style.opacity = "0.85";
  viewer.append(baseImg, overlayImg);
  el.appendChild(viewer);

  [baseImg, overlayImg].forEach((img) => img.addEventListener("click", () => openZoom(overlayImg.src, overlayImg.alt)));

  const opacityRow = document.createElement("div");
  opacityRow.className = "xai-opacity-row";
  opacityRow.innerHTML = `<label for="xai-opacity">Heatmap opacity</label><input type="range" id="xai-opacity" min="0" max="100" value="85" aria-label="Heatmap overlay opacity">`;
  el.appendChild(opacityRow);
  opacityRow.querySelector("input").addEventListener("input", (e) => {
    overlayImg.style.opacity = String(Number(e.target.value) / 100);
  });

  tabs.querySelectorAll(".xai-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.querySelectorAll(".xai-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const m = methods.find((mm) => mm.key === tab.dataset.key);
      if (m) overlayImg.src = m.src;
    });
  });

  const caption = document.createElement("p");
  caption.className = "xai-caption";
  caption.textContent = "Integrated Gradients is our main explanation; Grad-CAM is a visual aid. Both show regions that most influenced the prediction — not proof of clinical reasoning.";
  el.appendChild(caption);
}

/* ========================= Rendering: similar cases ========================= */
function renderSimilarCases(data) {
  const el = $("similar-cases-panel");
  if (!el) return;
  el.replaceChildren();
  if (!data.similar_cases || data.similar_cases.length === 0) {
    el.innerHTML = `<div class="empty-state"><h3>No similar cases found</h3><p>The training-feature cache may be missing.</p></div>`;
    return;
  }
  const grid = document.createElement("div");
  grid.className = "case-grid";
  data.similar_cases.forEach((c) => {
    const card = document.createElement("div");
    card.className = "case-card";
    const thumbSrc = c.thumbnail_png_base64 ? `data:image/png;base64,${c.thumbnail_png_base64}` : null;
    card.innerHTML = `
      ${thumbSrc ? `<img src="${thumbSrc}" alt="Training case, KL grade ${c.grade}">` : `<div class="empty-image" style="height:92px"><span>No preview</span></div>`}
      <div class="meta"><b>KL ${c.grade} — ${GRADE_LABELS[c.grade]}</b><span>${(c.similarity * 100).toFixed(1)}% similar</span></div>`;
    if (thumbSrc) card.addEventListener("click", () => openZoom(thumbSrc, `Similar training case, KL grade ${c.grade}`));
    grid.appendChild(card);
  });
  el.appendChild(grid);
}

/* ========================= Rendering: causal panel (with live BMI slider) ========================= */
function renderCausalPanel(panel, baseAge, baseFamilyHistory) {
  const el = $("causal-panel-result");
  if (!el) return;
  el.replaceChildren();

  if (panel.refused) {
    const div = document.createElement("div");
    div.className = "small-warning";
    div.textContent = panel.messages.join(" ");
    el.appendChild(div);
    return;
  }
  if (!panel.available) {
    el.innerHTML = `<div class="empty-state"><h3>Add age and BMI</h3><p>${panel.message || "Provide both for a per-factor causal breakdown."}</p></div>`;
    return;
  }

  const summary = document.createElement("div");
  summary.className = "causal-block";
  const lines = [];
  lines.push(`<div class="stat-line"><span>This patient</span><b>${panel.obese ? "Obese" : "Not obese"} (BMI ${panel.bmi})</b></div>`);
  lines.push(`<div class="stat-line"><span>Expected KL grade for this profile</span><b>${panel.expected_kl.value.toFixed(2)}${panel.expected_kl.clipped ? " (clipped)" : ""}</b></div>`);
  if (panel.individual_obesity_effect) {
    const e = panel.individual_obesity_effect;
    lines.push(`<div class="stat-line"><span>This patient's individual obesity effect</span><b>${e.value >= 0 ? "+" : ""}${e.value.toFixed(2)} KL pts</b></div>`);
  }
  summary.innerHTML = `<h4>Individual estimate</h4>${lines.join("")}`;
  el.appendChild(summary);

  if (panel.knee_injury) {
    const injury = document.createElement("div");
    injury.className = "causal-block";
    injury.innerHTML = `<h4>Knee injury (association, not a demonstrated cause)</h4>
      <div class="stat-line"><span>Regression adjustment</span><b>+${panel.knee_injury.linear_regression_ate.toFixed(2)} KL pts</b></div>
      <div class="stat-line"><span>Propensity score matching</span><b>+${panel.knee_injury.propensity_score_matching_ate.toFixed(2)} KL pts</b></div>`;
    el.appendChild(injury);
  }

  const whatif = document.createElement("div");
  whatif.className = "causal-block";
  whatif.innerHTML = `<h4>BMI what-if</h4>
    <div class="slider-row">
      <input type="range" id="bmi-whatif-slider" min="16.9" max="48.7" step="0.1" value="${panel.bmi}" aria-label="What-if BMI slider">
      <span class="slider-readout" id="bmi-whatif-readout">${panel.bmi}</span>
    </div>
    <div class="whatif-result" id="bmi-whatif-result">Expected KL grade at BMI ${panel.bmi}: ${panel.expected_kl.value.toFixed(2)}</div>`;
  el.appendChild(whatif);

  const slider = whatif.querySelector("#bmi-whatif-slider");
  const readout = whatif.querySelector("#bmi-whatif-readout");
  const resultBox = whatif.querySelector("#bmi-whatif-result");
  let debounceTimer = null;
  slider.addEventListener("input", () => {
    const bmiVal = Number(slider.value);
    readout.textContent = bmiVal.toFixed(1);
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      try {
        const resp = await fetch("/api/whatif", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ age: baseAge, bmi: bmiVal, family_history: baseFamilyHistory }),
        });
        const d = await resp.json();
        if (d.ok) {
          resultBox.textContent = `Expected KL grade at BMI ${bmiVal.toFixed(1)}: ${d.expected_kl.value.toFixed(2)}${d.expected_kl.clipped ? " (clipped)" : ""}`;
        } else if (d.refused) {
          resultBox.textContent = d.messages.join(" ");
        }
      } catch (e) { /* keep showing the last good value */ }
    }, 180);
  });

  const caveat = document.createElement("div");
  caveat.className = "small-warning";
  caveat.textContent = panel.caveat;
  el.appendChild(caveat);
}

/* ========================= Rendering: joint space width panel ========================= */
function renderJsw(data) {
  const el = $("jsw-panel");
  const badge = $("jsw-status-badge");
  if (!el) return;
  el.replaceChildren();

  if (badge) {
    const rho = data.status_rho;
    const rhoText = typeof rho === "number" ? `ρ = ${rho.toFixed(3)}` : "ρ unavailable";
    if (data.status === "validated") {
      badge.textContent = `Validated (${rhoText})`;
      badge.className = "jsw-badge validated";
    } else {
      badge.textContent = `Experimental — not validated (${rhoText})`;
      badge.className = "jsw-badge experimental";
    }
  }

  const toggleRow = document.createElement("div");
  toggleRow.className = "jsw-toggle-row";
  toggleRow.innerHTML = `<input type="checkbox" id="jsw-lines-toggle" checked><label for="jsw-lines-toggle">Show measurement lines</label>`;
  el.appendChild(toggleRow);

  const imgWrap = document.createElement("div");
  imgWrap.className = "jsw-image-wrap";
  const overlaySrc = data.overlay_png_base64 ? `data:image/png;base64,${data.overlay_png_base64}` : null;
  const baseSrc = data.base_png_base64 ? `data:image/png;base64,${data.base_png_base64}` : null;
  const img = document.createElement("img");
  img.src = overlaySrc || baseSrc || "";
  img.alt = "Knee X-ray with the joint line and measured femur/tibia edges drawn on it";
  img.addEventListener("click", () => openZoom(img.src, img.alt));
  imgWrap.appendChild(img);
  el.appendChild(imgWrap);

  toggleRow.querySelector("input").addEventListener("change", (e) => {
    img.src = e.target.checked ? (overlaySrc || baseSrc) : (baseSrc || overlaySrc);
  });

  const compartments = data.compartments || [];
  const compGrid = document.createElement("div");
  compGrid.className = "jsw-compartments";
  compartments.forEach((c) => {
    const card = document.createElement("div");
    card.className = "jsw-comp-card";
    card.innerHTML = c.failed
      ? `<h5>${c.label}</h5><p class="fail-msg">${c.message}</p>`
      : `<h5>${c.label}</h5><div class="val">${c.min_jsw.toFixed(3)}<small>min</small></div><div class="val">${c.mean_jsw.toFixed(3)}<small>mean</small></div>`;
    compGrid.appendChild(card);
  });
  el.appendChild(compGrid);

  if (compartments.length === 2 && !compartments[0].failed && !compartments[1].failed) {
    const compare = document.createElement("div");
    compare.className = "jsw-compare";
    const [a, b] = compartments;
    const scale = Math.max(a.min_jsw, b.min_jsw, 0.01);
    compare.innerHTML = `
      <div class="jsw-compare-row"><span>${a.label}</span><div class="jsw-compare-track"><div class="jsw-compare-fill medial" style="width:0"></div></div></div>
      <div class="jsw-compare-row"><span>${b.label}</span><div class="jsw-compare-track"><div class="jsw-compare-fill lateral" style="width:0"></div></div></div>`;
    el.appendChild(compare);
    requestAnimationFrame(() => {
      const fills = compare.querySelectorAll(".jsw-compare-fill");
      if (fills[0]) fills[0].style.width = `${(a.min_jsw / scale) * 100}%`;
      if (fills[1]) fills[1].style.width = `${(b.min_jsw / scale) * 100}%`;
    });
  }

  const notes = document.createElement("div");
  notes.className = "small-warning";
  notes.innerHTML = `${data.note_units}<br>${data.note_not_kl}`;
  el.appendChild(notes);
}

/* ========================= Run assessment ========================= */
$("analyze-btn")?.addEventListener("click", async () => {
  if (!selectedFile) { toast("Select an X-ray first."); return; }

  const button = $("analyze-btn");
  button.disabled = true;
  if ($("analysis-status")) $("analysis-status").textContent = "Running the model…";
  const progress = runProgressAnimation();

  const formData = new FormData();
  formData.append("xray", selectedFile);
  const age = $("age")?.value; if (age) formData.append("age", age);
  const bmi = $("bmi")?.value; if (bmi) formData.append("bmi", bmi);
  const fh = $("family_history")?.value; if (fh) formData.append("family_history", fh);
  const kneeSide = $("knee_side")?.value;

  const jswForm = new FormData();
  jswForm.append("xray", selectedFile);
  if (kneeSide) jswForm.append("side", kneeSide);
  const jswPromise = fetch("/api/jsw", { method: "POST", body: jswForm, cache: "no-store" })
    .then((r) => r.json()).catch(() => ({ ok: false }));

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData, cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.message || `Assessment failed (HTTP ${response.status}).`);
    }
    progress.finish(true);
    renderPrediction(data);
    renderXai(data);
    renderSimilarCases(data);
    renderCausalPanel(data.causal_panel, age ? Number(age) : null, fh || null);
    window.__lastResult = data;
    if ($("analysis-status")) $("analysis-status").textContent = `Done. ${data.disclaimer}`;
    toast("Assessment completed.");

    jswPromise.then((jswData) => {
      if (jswData && jswData.ok) {
        renderJsw(jswData);
        window.__lastJsw = jswData;
      }
    });
  } catch (error) {
    progress.finish(false);
    if ($("analysis-status")) $("analysis-status").textContent = error.message || "Assessment failed.";
    toast(error.message || "Assessment failed.");
  } finally {
    button.disabled = false;
  }
});

/* ========================= Download report (print to PDF) ========================= */
$("download-report-btn")?.addEventListener("click", () => window.print());

/* ========================= Causal insights page ========================= */
const METHOD_LABELS = {
  propensity_score_matching: "Propensity score matching",
  linear_regression: "Regression adjustment",
  random_common_cause: "Random common cause",
  placebo_treatment_refuter: "Placebo treatment",
  data_subset_refuter: "Data subset",
};
function readable(name) { return METHOD_LABELS[name] || name; }

async function loadCausalInsights() {
  const methodsEl = $("causal-methods");
  const ciEl = $("ci-bars");
  const robustEl = $("causal-robustness");
  if (methodsEl) methodsEl.textContent = "Loading…";
  try {
    const resp = await fetch("/api/causal_insights", { cache: "no-store" });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error("Could not load causal insights.");

    if (methodsEl) {
      methodsEl.replaceChildren();
      data.methods.forEach((m) => {
        const card = document.createElement("div");
        card.className = "method-card";
        card.innerHTML = `<b>+${m.ate.toFixed(3)}</b><span>${m.label}</span><small>KL grade points${m.n ? `, n = ${m.n}` : ""}</small>`;
        methodsEl.appendChild(card);
      });
    }

    if (ciEl) {
      ciEl.replaceChildren();
      ciEl.style.display = "block";
      const wrap = document.createElement("div");
      data.age_group_ci.forEach((row) => {
        const lo = row.ci_low, hi = row.ci_high, est = row.estimate;
        const scaleMax = 0.8;
        const pct = (v) => Math.max(0, Math.min(100, (v / scaleMax) * 100));
        const line = document.createElement("div");
        line.className = "ci-bar-row";
        line.innerHTML = `
          <span class="name">${row.label.replace(" (adj. age+family history)", "")}</span>
          <div class="ci-bar-track">
            <div class="ci-bar-fill" style="left:${pct(lo)}%;width:${pct(hi) - pct(lo)}%"></div>
            <div class="ci-bar-point" style="left:${pct(est)}%"></div>
          </div>`;
        wrap.appendChild(line);
      });
      ciEl.appendChild(wrap);
      const caption = document.createElement("p");
      caption.className = "hint";
      caption.style.marginTop = "6px";
      caption.textContent = "Obesity effect on KL grade by age group, with 95% confidence intervals.";
      ciEl.appendChild(caption);
    }

    if (robustEl) {
      robustEl.replaceChildren();
      if (data.evalue.point) {
        const card = document.createElement("div");
        card.className = "method-card";
        card.innerHTML = `<b>${data.evalue.point.toFixed(2)}</b><span>E-value</span><small>An unmeasured confounder would need at least this strong an association with both obesity and KL grade to explain the effect away (CI-bound E-value: ${data.evalue.ci_bound?.toFixed(2) ?? "—"}).</small>`;
        robustEl.appendChild(card);
      }
      const refCard = document.createElement("div");
      refCard.className = "method-card";
      refCard.innerHTML = `<b>Stable</b><span>Refutation checks</span><small>${data.refuters.map((r) => `${readable(r.name)}: ${r.new_effect.toFixed(3)}`).join(" · ")}</small>`;
      robustEl.appendChild(refCard);

      if (data.injury.linear_regression !== undefined) {
        const injuryCard = document.createElement("div");
        injuryCard.className = "method-card";
        injuryCard.innerHTML = `<b>+${data.injury.propensity_score_matching.toFixed(2)}</b><span>Knee injury (association)</span><small>Regression adjustment: +${data.injury.linear_regression.toFixed(2)} KL pts. Self-reported history; not a demonstrated cause.</small>`;
        robustEl.appendChild(injuryCard);
      }
    }
  } catch (error) {
    if (methodsEl) methodsEl.textContent = error.message || "Could not load causal insights.";
  }
}

window.addEventListener("beforeunload", () => {
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
});
