// In-node live preview for H3 Preview Override.
//
// Ported from ComfyUI-KJNodes web/js/preview_override/preview_override.js (GPL-3.0).
// Kijai's behaviour is kept: per-step frame cache with hover/lock scrubbing, animated
// WebP decoded through ImageDecoder and driven by a global timer, double-buffered MP4
// playback, a playback scrub bar with click-to-pause, the sigma/delta + step-time graphs,
// and the drag grip that resizes the graph panel.
//
// Differences from the original:
//   * `preview_fps` is gone. The playback rate is derived server-side from the shot's
//     real duration ("true speed") and arrives in the payload, so `currentFps()` reads
//     the last rate the server sent instead of a widget.
//   * The graph panel collapses to its header via the button in that header, persisted
//     on node.properties alongside the panel height.
//   * The SamplerDetailBoost curve overlay is not ported (it needs a KJNodes sampler
//     this pack does not ship).
//   * Pointer input is handled by document-level capture-phase listeners with rect
//     hit-testing rather than listeners on the elements themselves — see the note above
//     `cellUnderPointer`.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_TYPE = "H3PreviewOverride";
const STYLE_ID = "h3-pov-stylesheet";
const _cssUrl = new URL("./h3_preview_override.css", import.meta.url).href;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const link = document.createElement("link");
  link.id = STYLE_ID;
  link.rel = "stylesheet";
  link.href = _cssUrl;
  document.head.appendChild(link);
}

// Ported from KJNodes utility.js — appends a callback rather than clobbering an existing one.
function chainCallback(object, property, callback) {
  if (object == undefined) return;
  if (property in object) {
    const callback_orig = object[property];
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r;
    };
  } else {
    object[property] = callback;
  }
}

// Wheel over the panel should still zoom the graph, not scroll the widget.
function addWheelPassthrough(element) {
  element.addEventListener("wheel", (e) => {
    const gc = document.getElementById("graph-canvas");
    if (gc) {
      gc.dispatchEvent(new WheelEvent(e.type, e));
      e.preventDefault();
    }
  }, { passive: false });
}

// Walks subgraph chain for IDs like "12:7:5". Mirrors getNodeByExecutionId (not exported).
function findNodeByQualifiedId(rootGraph, qid) {
  if (!rootGraph || qid == null) return null;
  const parts = String(qid).split(":");
  let graph = rootGraph;
  for (let i = 0; i < parts.length - 1; i++) {
    const parentId = parseInt(parts[i], 10);
    if (!Number.isFinite(parentId)) return null;
    const parentNode = graph?.getNodeById?.(parentId);
    if (!parentNode?.subgraph) return null;
    graph = parentNode.subgraph;
  }
  const leafId = parseInt(parts[parts.length - 1], 10);
  if (!Number.isFinite(leafId)) return null;
  return graph?.getNodeById?.(leafId) || null;
}

api.addEventListener("h3_preview_override", (e) => {
  const data = e.detail;
  if (!data || data.node_id == null) return;
  const node = findNodeByQualifiedId(app.graph, data.node_id);
  if (node?._h3PreviewHandler) node._h3PreviewHandler(data);
});

const GRAPH_PAD_X = 4;
const GRAPH_PAD_Y = 3;
const DEFAULT_PANEL_H = 140;
const MIN_PANEL_H = 60;

function fmt(n, d) {
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

function el(tag, className, parent) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (parent) parent.appendChild(e);
  return e;
}

function b64ToBlob(b64, mime) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

// ------------------------------------------------------------------------ graph canvases

function syncCanvasDPR(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = canvas.clientHeight || canvas.height;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: cssW, H: cssH };
}

function drawGridlines(ctx, W, H, padX, padY) {
  ctx.strokeStyle = "#1e1e1e";
  ctx.lineWidth = 1;
  for (let g = 1; g < 4; g++) {
    const y = Math.round(padY + (g / 4) * (H - 2 * padY)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(W - padX, y);
    ctx.stroke();
  }
}

// Shared cursor overlay so both graphs mark the same step the same way.
function drawCursors(ctx, xAt, padY, H, xSteps, hoverStep, lockedStep) {
  if (lockedStep != null && lockedStep >= 0 && lockedStep < xSteps) {
    const lx = xAt(lockedStep) + 0.5;
    ctx.strokeStyle = "rgba(245, 200, 60, 0.9)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(lx, padY);
    ctx.lineTo(lx, H - padY);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (hoverStep != null && hoverStep >= 0 && hoverStep < xSteps) {
    const hx = xAt(hoverStep) + 0.5;
    ctx.strokeStyle = "rgba(208, 208, 208, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hx, padY);
    ctx.lineTo(hx, H - padY);
    ctx.stroke();
  }
}

// σ + Δ overlaid; each series normalised to its own max for comparable shapes.
// xSteps (the step axis) is passed in rather than derived here, so hit-testing and drawing
// cannot disagree about where a step sits — that mismatch is what puts the hover line
// somewhere other than under the cursor once one series is longer than another.
function drawSigmaDeltaGraph(canvas, sigmas, deltas, step, xSteps, hoverStep, lockedStep) {
  const { ctx, W, H } = syncCanvasDPR(canvas);
  const padX = GRAPH_PAD_X, padY = GRAPH_PAD_Y;
  const iW = W - 2 * padX, iH = H - 2 * padY;
  ctx.clearRect(0, 0, W, H);
  drawGridlines(ctx, W, H, padX, padY);

  const axis = Math.max(1, xSteps);
  const xAt = (i) => padX + (i / Math.max(1, axis - 1)) * iW;
  const n = sigmas?.length || 0;

  let sYAt = null;
  if (n > 1) {
    let sMax = -Infinity, sMin = Infinity;
    for (const s of sigmas) { if (s > sMax) sMax = s; if (s < sMin) sMin = s; }
    if (sMin > 0) sMin = 0;
    const sRange = Math.max(sMax - sMin, 1e-6);
    sYAt = (v) => padY + (1 - (v - sMin) / sRange) * iH;

    ctx.strokeStyle = "rgba(208, 208, 208, 0.55)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = xAt(i), py = sYAt(sigmas[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    if (step >= 0) {
      const i = Math.max(0, Math.min(n - 1, step));
      ctx.fillStyle = "#d0d0d0";
      ctx.beginPath();
      ctx.arc(xAt(i), sYAt(sigmas[i]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  let dYAt = null;
  if (deltas && deltas.length >= 1) {
    let dMax = -Infinity;
    for (const v of deltas) if (Number.isFinite(v) && v > dMax) dMax = v;
    const dRange = Math.max(dMax, 1e-6);
    dYAt = (v) => padY + (1 - v / dRange) * iH;

    // delta[i] is plotted at boundary (i+1); flat-extend delta[0] back to boundary 0.
    ctx.beginPath();
    ctx.moveTo(xAt(0), H - padY);
    ctx.lineTo(xAt(0), dYAt(deltas[0]));
    for (let i = 0; i < deltas.length; i++) ctx.lineTo(xAt(i + 1), dYAt(deltas[i]));
    ctx.lineTo(xAt(deltas.length), H - padY);
    ctx.closePath();
    ctx.fillStyle = "rgba(230, 126, 34, 0.15)";
    ctx.fill();

    ctx.strokeStyle = "#e67e22";
    ctx.lineWidth = 1.3;
    if (deltas.length === 1) {
      ctx.fillStyle = "#e67e22";
      ctx.beginPath();
      ctx.arc(xAt(1), dYAt(deltas[0]), 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      for (let i = 0; i < deltas.length; i++) {
        const px = xAt(i + 1), py = dYAt(deltas[i]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
  }

  drawCursors(ctx, xAt, padY, H, axis, hoverStep, lockedStep);

  if (hoverStep != null && hoverStep >= 0 && hoverStep < axis) {
    const hx = xAt(hoverStep) - 0.5;
    if (sYAt && hoverStep < n) {
      ctx.fillStyle = "#d0d0d0";
      ctx.beginPath();
      ctx.arc(hx, sYAt(sigmas[hoverStep]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (dYAt && hoverStep >= 1 && (hoverStep - 1) < deltas.length) {
      ctx.fillStyle = "#e67e22";
      ctx.beginPath();
      ctx.arc(hx, dYAt(deltas[hoverStep - 1]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawLineGraph(canvas, values, xSteps, hoverStep, lockedStep) {
  const { ctx, W, H } = syncCanvasDPR(canvas);
  const padX = GRAPH_PAD_X, padY = GRAPH_PAD_Y;
  const iW = W - 2 * padX, iH = H - 2 * padY;
  ctx.clearRect(0, 0, W, H);
  drawGridlines(ctx, W, H, padX, padY);

  const axis = Math.max(1, xSteps);
  const xAt = (i) => padX + (i / Math.max(1, axis - 1)) * iW;

  if (values && values.length >= 1) {
    let vMax = -Infinity, vMin = Infinity;
    for (const v of values) { if (v > vMax) vMax = v; if (v < vMin) vMin = v; }
    if (vMin > 0) vMin = 0;
    const vRange = Math.max(vMax - vMin, 1e-6);
    const yAt = (v) => padY + (1 - (v - vMin) / vRange) * iH;

    // values[i] is the wait before step i+1, so it plots at boundary i+1 — the same
    // convention as the Δ series, which keeps the two graphs' cursors aligned.
    ctx.beginPath();
    ctx.moveTo(xAt(0), H - padY);
    ctx.lineTo(xAt(0), yAt(values[0]));
    for (let i = 0; i < values.length; i++) ctx.lineTo(xAt(i + 1), yAt(values[i]));
    ctx.lineTo(xAt(values.length), H - padY);
    ctx.closePath();
    ctx.fillStyle = "rgba(230, 126, 34, 0.15)";
    ctx.fill();

    ctx.strokeStyle = "#e67e22";
    ctx.lineWidth = 1.3;
    if (values.length === 1) {
      ctx.fillStyle = "#e67e22";
      ctx.beginPath();
      ctx.arc(xAt(1), yAt(values[0]), 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      for (let i = 0; i < values.length; i++) {
        const px = xAt(i + 1), py = yAt(values[i]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    if (hoverStep != null && hoverStep >= 1 && (hoverStep - 1) < values.length) {
      ctx.fillStyle = "#e67e22";
      ctx.beginPath();
      ctx.arc(xAt(hoverStep) - 0.5, yAt(values[hoverStep - 1]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawCursors(ctx, xAt, padY, H, axis, hoverStep, lockedStep);
}

// ------------------------------------------------------------------------- settings popup
// All of the schema widgets (decode, playback, preview_target, preview_frames,
// max_resolution, jpeg_quality, every_n_steps, max_preview_overhead,
// suppress_default_preview, tiny_vae, show_vae_input) are pulled off the node body and
// shown in a popup instead, opened from a "⚙" button in the panel header. This is a
// *display* change only: the same widget objects back the popup's controls and (as
// always) the values ComfyUI serialises into the workflow and sends to the Python node —
// the popup just writes to `widget.value` directly rather than the widget drawing its own
// row on the node. The one exception is `show_vae_input`, which also toggles the actual
// `vae` socket on the node face.

// Hides a widget's own row on the node body without touching its value, its callback, or
// how it serialises — `hidden` is read by LiteGraph's widget layout/draw pass, and the
// zeroed computeSize is a belt-and-suspenders fallback for any pass that only consults
// size. Nothing here changes `widget.type`, so the widget keeps behaving exactly like a
// normal combo/number/toggle widget everywhere except on-canvas.
function hideWidget(widget) {
  widget.hidden = true;
  widget.origComputeSize = widget.computeSize;
  widget.computeSize = () => [0, -4];
}

function widgetTooltip(widget) {
  return widget.tooltip || widget.options?.tooltip || "";
}

function comboValues(widget) {
  const v = widget.options?.values;
  return typeof v === "function" ? v(widget) : (v || []);
}

// `vae` is a socket (io.Vae.Input), not a widget, so "hide/show" means adding/removing
// the actual input slot — there's no vanilla "hidden but still linked" state for a slot,
// so removing it also drops any existing wire (documented in the setting's tooltip).
function syncVaeInputVisibility(node, show) {
  const idx = (node.inputs || []).findIndex((inp) => inp.name === "vae");
  if (show) {
    if (idx === -1) node.addInput("vae", "VAE");
  } else if (idx !== -1) {
    node.removeInput(idx);
  }
  node.setDirtyCanvas?.(true, true);
}

// One label + control (+ optional description, plus a native hover tooltip) per setting,
// built fresh each time the popup opens so it always reflects the widget's current value.
// This modal lives directly on document.body (not inside the node's DOM-widget tree), so
// its own controls use plain listeners — the delivery problem documented above
// `cellUnderPointer` is specific to elements inside the widget/canvas overlay.
function buildSettingsRow(node, widget) {
  const row = document.createElement("div");
  Object.assign(row.style, {
    display: "flex", flexDirection: "column", gap: "3px",
    padding: "7px 0", borderBottom: "1px solid #2c2c2c",
  });
  const tip = widgetTooltip(widget);
  if (tip) row.title = tip; // mouseover caption, on top of the description line below

  const labelRow = document.createElement("div");
  Object.assign(labelRow.style, {
    display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px",
  });
  const label = document.createElement("label");
  label.textContent = widget.label || widget.name;
  Object.assign(label.style, { color: "#ddd", fontSize: "12px", fontWeight: "600" });
  labelRow.appendChild(label);

  const commit = (value) => {
    widget.value = value;
    widget.callback?.(value, app.canvas, node);
    if (widget.name === "show_vae_input") syncVaeInputVisibility(node, value);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
  };

  let control;
  if (widget.type === "combo") {
    control = document.createElement("select");
    for (const opt of comboValues(widget)) {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      if (opt === widget.value) o.selected = true;
      control.appendChild(o);
    }
    control.addEventListener("change", () => commit(control.value));
  } else if (widget.type === "toggle") {
    control = document.createElement("input");
    control.type = "checkbox";
    control.checked = !!widget.value;
    control.style.cursor = "pointer";
    control.addEventListener("change", () => commit(control.checked));
  } else {
    control = document.createElement("input");
    control.type = "number";
    const opts = widget.options || {};
    if (opts.min != null) control.min = String(opts.min);
    if (opts.max != null) control.max = String(opts.max);
    control.step = String(opts.step || (Number.isInteger(widget.value) ? 1 : 0.1));
    control.value = widget.value;
    control.addEventListener("change", () => {
      let v = parseFloat(control.value);
      if (Number.isNaN(v)) v = widget.value;
      if (opts.min != null) v = Math.max(opts.min, v);
      if (opts.max != null) v = Math.min(opts.max, v);
      control.value = v;
      commit(v);
    });
  }
  Object.assign(control.style, {
    background: "#1c1c1c", color: "#eee", border: "1px solid #444", borderRadius: "4px",
    padding: "4px 6px", fontSize: "12px", minWidth: "150px", boxSizing: "border-box",
  });
  if (tip) control.title = tip;
  labelRow.appendChild(control);
  row.appendChild(labelRow);

  if (tip) {
    const desc = document.createElement("div");
    desc.textContent = tip;
    Object.assign(desc.style, { color: "#8a8a8a", fontSize: "10.5px", lineHeight: "1.4" });
    row.appendChild(desc);
  }
  return row;
}

function closeSettingsModal(node) {
  const overlay = node._h3SettingsOverlay;
  if (!overlay) return;
  if (overlay._h3KeyHandler) document.removeEventListener("keydown", overlay._h3KeyHandler);
  overlay.remove();
  node._h3SettingsOverlay = null;
}

function openSettingsModal(node) {
  closeSettingsModal(node); // no stacked popups if the button is clicked twice

  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", background: "rgba(0,0,0,0.55)",
    zIndex: "10000", display: "flex", alignItems: "center", justifyContent: "center",
  });
  overlay.addEventListener("mousedown", (e) => {
    if (e.target === overlay) closeSettingsModal(node);
  });

  const dialog = document.createElement("div");
  Object.assign(dialog.style, {
    background: "#1a1a1a", border: "1px solid #3a3a3a", borderRadius: "8px",
    width: "380px", maxWidth: "90vw", maxHeight: "80vh", display: "flex",
    flexDirection: "column", boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    fontFamily: "sans-serif",
  });
  dialog.addEventListener("mousedown", (e) => e.stopPropagation());

  const header = document.createElement("div");
  Object.assign(header.style, {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "10px 14px", borderBottom: "1px solid #333", flex: "none",
  });
  const title = document.createElement("div");
  title.textContent = "H3 Preview Override — Settings";
  Object.assign(title.style, { color: "#fff", fontSize: "13px", fontWeight: "700" });
  const closeBtn = document.createElement("button");
  closeBtn.textContent = "×";
  Object.assign(closeBtn.style, {
    background: "transparent", border: "none", color: "#aaa", fontSize: "20px",
    cursor: "pointer", lineHeight: "1", padding: "0 2px",
  });
  closeBtn.addEventListener("click", () => closeSettingsModal(node));
  header.appendChild(title);
  header.appendChild(closeBtn);

  const body = document.createElement("div");
  Object.assign(body.style, { padding: "2px 14px", overflowY: "auto" });
  for (const widget of node._h3SettingsWidgets || []) {
    body.appendChild(buildSettingsRow(node, widget));
  }

  dialog.appendChild(header);
  dialog.appendChild(body);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  node._h3SettingsOverlay = overlay;

  const onKeyDown = (e) => {
    if (e.key === "Escape") closeSettingsModal(node);
  };
  document.addEventListener("keydown", onKeyDown);
  overlay._h3KeyHandler = onKeyDown;
}

app.registerExtension({
  name: "H3PreviewOverride",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      ensureStyles();
      const node = this;

      // Everything on `node.widgets` right now is a schema-driven value widget (model and
      // vae are sockets, not widgets) — capture that exact list before anything below adds
      // more, then hide each one so it no longer draws a row. They move into the Settings
      // popup instead (opened from the "⚙" button built into the panel header below).
      node._h3SettingsWidgets = (node.widgets || []).slice();
      for (const w of node._h3SettingsWidgets) hideWidget(w);

      const root = el("div", "h3-pov-root");

      const imageArea = el("div", "h3-pov-image-area", root);
      // Double-buffered: decode() on the visible-to-be element so the bitmap is reused.
      const imgA = el("img", "h3-pov-img", imageArea);
      const imgB = el("img", "h3-pov-img", imageArea);
      imgA.draggable = false;
      imgB.draggable = false;
      imgB.style.opacity = "0";
      let visibleImg = imgA;
      let pendingImg = imgB;
      // WebP path: ImageDecoder -> VideoFrame[] -> canvas, driven by a global timer.
      const videoCanvas = el("canvas", "h3-pov-img", imageArea);
      videoCanvas.style.opacity = "0";
      const videoCtx = videoCanvas.getContext("2d");
      // MP4 path: double-buffered <video> for the same no-flash reason as imgA/B.
      function mkVideo() {
        const v = el("video", "h3-pov-img", imageArea);
        v.style.opacity = "0";
        v.muted = true;
        v.playsInline = true;
        v.loop = true;
        v.autoplay = true;
        v.disablePictureInPicture = true;
        return v;
      }
      const videoA = mkVideo();
      const videoB = mkVideo();
      let visibleVideo = videoA;
      let pendingVideo = videoB;
      const placeholder = el("div", "h3-pov-placeholder", imageArea);
      placeholder.textContent = "waiting for sample…";
      // Playback scrub bar — hidden until animated content is loaded.
      const scrubBar = el("div", "h3-pov-scrub", imageArea);
      const scrubFill = el("div", "h3-pov-scrub-fill", scrubBar);
      scrubBar.style.display = "none";

      const grip = el("div", "h3-pov-panel-grip", root);
      grip.title = "Drag to resize panel";

      const panel = el("div", "h3-pov-panel", root);
      const header = el("div", "h3-pov-panel-header", panel);
      const headerLeft = el("span", "h3-pov-header-left", header);
      const collapseBtn = el("span", "h3-pov-collapse", headerLeft);
      const headerTitle = el("span", "h3-pov-panel-title", headerLeft);
      headerTitle.textContent = "H3 Preview";
      const headerSummary = el("span", "h3-pov-panel-summary", header);
      headerSummary.textContent = "idle";
      const settingsBtn = el("span", "h3-pov-collapse", header);
      settingsBtn.textContent = "⚙";
      settingsBtn.title = "Open all of this node's settings";

      // Reflect show_vae_input's starting value (its schema default, or a value a saved
      // workflow already restored onto it if this callback fires after configure).
      const showVaeWidget = node._h3SettingsWidgets.find((w) => w.name === "show_vae_input");
      if (showVaeWidget) syncVaeInputVisibility(node, showVaeWidget.value);

      const grid = el("div", "h3-pov-graphs-grid", panel);

      function makeCell(labelText) {
        const cell = el("div", "h3-pov-graph-cell", grid);
        const head = el("div", "h3-pov-graph-head", cell);
        const lbl = el("span", "h3-pov-graph-label", head);
        lbl.textContent = labelText;
        const valEl = el("span", "h3-pov-graph-value", head);
        const canvas = el("canvas", "h3-pov-graph-canvas", cell);
        return { canvas, valEl, lbl };
      }
      const sdRow = makeCell("σ / Δ");
      const timeRow = makeCell("step time (ms)");
      sdRow.canvas.style.cursor = "crosshair";
      timeRow.canvas.style.cursor = "crosshair";
      sdRow.canvas.title = "Hover to scrub · click to lock a step · ← → to step";
      timeRow.canvas.title = "Hover to scrub · click to toggle ms ↔ s";

      node.properties = node.properties || {};
      if (typeof node.properties.h3PanelH === "number") {
        panel.style.height = node.properties.h3PanelH + "px";
      }

      // ---- collapsible graph panel -----------------------------------------------
      // Collapsing leaves the header (and its live summary) in place and drops the
      // graphs and the resize grip, handing that space back to the preview image.
      function setCollapsed(collapsed, persist) {
        panel.classList.toggle("h3-pov-collapsed", !!collapsed);
        grid.style.display = collapsed ? "none" : "";
        grip.style.display = collapsed ? "none" : "";
        collapseBtn.textContent = collapsed ? "▸" : "▾";
        collapseBtn.title = collapsed ? "Show the graphs" : "Hide the graphs";
        if (!collapsed && typeof node.properties.h3PanelH === "number") {
          panel.style.height = node.properties.h3PanelH + "px";
        }
        if (persist) {
          node.properties.h3PanelCollapsed = !!collapsed;
          node.graph?.change?.();
        }
        // A canvas laid out while display:none has no size, so anything drawn into it
        // while collapsed is lost. Redraw once the browser has given it a box again.
        if (!collapsed) requestAnimationFrame(() => redrawSd());
      }
      setCollapsed(!!node.properties.h3PanelCollapsed, false);

      chainCallback(node, "onConfigure", function () {
        if (typeof node.properties?.h3PanelH === "number") {
          panel.style.height = node.properties.h3PanelH + "px";
        }
        setCollapsed(!!node.properties?.h3PanelCollapsed, false);

        // Heal combo widgets whose saved value is not one of their options.
        //
        // ComfyUI stores widgets_values positionally, so a workflow saved while the
        // widget list had a different shape hands every value after the change to the
        // wrong input — and a combo then refuses to run with "The value true is not
        // available". The values themselves are unrecoverable at that point; what
        // matters is that the node comes back usable instead of blocking the whole
        // workflow. Runs over every widget regardless of the Settings popup, since
        // hiding a widget's row doesn't change what it holds.
        for (const w of node.widgets || []) {
          const opts = w.options?.values;
          if (!Array.isArray(opts) || opts.length === 0) continue;
          if (opts.includes(w.value)) continue;
          const fallback = w.options?.default ?? opts[0];
          console.warn(
            `[H3PreviewOverride] saved value ${JSON.stringify(w.value)} is not a valid ` +
            `'${w.name}' — this workflow was saved against a different widget layout. ` +
            `Falling back to ${JSON.stringify(fallback)}; check the node's settings.`);
          w.value = fallback;
        }

        // Reconcile the vae socket with the restored show_vae_input value — whatever the
        // base configure() did with node.inputs, this makes the actual socket state
        // match the setting deterministically.
        const restoredShowVae = (node.widgets || []).find((w) => w.name === "show_vae_input");
        if (restoredShowVae) syncVaeInputVisibility(node, restoredShowVae.value);
      });

      // Per-run state. Declared together so helpers/handlers below can close over them.
      let hoverStep = null;
      let lockedStep = null;  // click-locked step; survives mouseleave for inspection
      let lastCurrentStep = -1;
      const history = { stepMs: [], delta: [] };
      let cachedSigmas = null;
      let totalSteps = 0;
      const stepBlobUrls = [];
      const stepVideoFrames = [];  // WebP path: per-step VideoFrame[]
      const stepMp4Urls = [];      // MP4 path: per-step blob URL
      let liveBlobUrl = null;
      let liveMp4Url = null;
      // Global timer never resets between steps — scrub continues at the equivalent elapsed.
      let playbackStartMs = null;
      let videoRafId = null;
      let currentMp4Url = null;

      let timeUnitSeconds = false;
      let lastStepMs = null;
      let lastAvgStepMs = null;
      let lastTotal = null;
      let lastStep = null;
      let lastW = null, lastH = null;

      // Playback rate, derived server-side from the shot's real duration and delivered in
      // the payload. Replaces KJNodes' preview_fps widget lookup: there is no frame-rate
      // control on this node, because "true speed" defines the rate from the clip itself.
      let serverFps = 12;
      function currentFps() {
        return Number.isFinite(serverFps) && serverFps > 0 ? serverFps : 12;
      }
      let bakedFps = null;   // MP4 encode-time fps; scales playbackRate on retime

      // ---- axis --------------------------------------------------------------------
      // Deliberately not gated on cachedSigmas: the σ schedule arrives once, up front, and
      // if that single message is missed scrubbing must still work off whatever did arrive.
      function xStepCount() {
        return Math.max(
          totalSteps || 0,
          cachedSigmas?.length || 0,
          history.delta.length + 1,
          history.stepMs.length + 1,
          lastCurrentStep + 1,
          1);
      }

      function fmtTime(ms) {
        if (ms == null || !Number.isFinite(ms)) return "—";
        return timeUnitSeconds ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(0)}ms`;
      }
      function activeStep() {
        if (hoverStep != null) return hoverStep;
        if (lockedStep != null) return lockedStep;
        return lastCurrentStep;
      }
      function stepPrefix() {
        if (hoverStep != null) return `[${hoverStep}] `;
        if (lockedStep != null) return `🔒[${lockedStep}] `;
        return "";
      }
      // history.stepMs[k-1] = step k's duration.
      function stepTimeForDisplay() {
        const tIdx = activeStep() - 1;
        return (tIdx >= 0 && tIdx < history.stepMs.length) ? history.stepMs[tIdx] : null;
      }
      function renderTime() {
        timeRow.lbl.textContent = timeUnitSeconds ? "step time (s)" : "step time (ms)";
        timeRow.valEl.textContent = stepPrefix() + fmtTime(stepTimeForDisplay());
        if (lastAvgStepMs != null && lastTotal != null && lastStep != null) {
          const eta = Math.max(0, lastTotal - lastStep) * lastAvgStepMs / 1000;
          const avgTxt = timeUnitSeconds
            ? `${(lastAvgStepMs / 1000).toFixed(2)}s/step`
            : `${lastAvgStepMs.toFixed(0)}ms/step`;
          headerSummary.textContent =
            `${lastW}×${lastH} · ${lastStep}/${lastTotal} · ${avgTxt} · ETA ${eta.toFixed(1)}s`;
        }
      }
      function updateSdHeader() {
        const idx = activeStep();
        if (idx < 0) { sdRow.valEl.textContent = ""; return; }
        // σ may legitimately be unavailable (schedule message missed) while Δ is not —
        // fmt() renders that as "—" rather than blanking the whole readout.
        const sig = (cachedSigmas && idx < cachedSigmas.length) ? cachedSigmas[idx] : null;
        const dIdx = idx - 1;
        const d = (dIdx >= 0 && dIdx < history.delta.length) ? history.delta[dIdx] : null;
        sdRow.valEl.textContent = `${stepPrefix()}${fmt(sig, 3)} / ${fmt(d, 3)}`;
      }

      // ---- animated playback -------------------------------------------------------
      // Load into hidden buffer, await its decode(), then opacity-swap. See the note on
      // the imgA/imgB definitions for why decode must run on the soon-to-be-visible one.
      function showLiveFrame(url) {
        const target = pendingImg;
        target.src = url;
        target.decode().then(() => {
          if (hoverStep != null || lockedStep != null) return;
          if (liveBlobUrl !== url) return;
          if (target !== pendingImg) return;  // a newer frame already swapped
          target.style.opacity = "1";
          visibleImg.style.opacity = "0";
          const prev = visibleImg;
          visibleImg = target;
          pendingImg = prev;
        }).catch(() => {});
      }
      // Baked WebP durations are dropped; rate comes live from the server-sent fps.
      async function decodeAnimatedBlob(blob) {
        if (typeof ImageDecoder === "undefined") return null;
        try {
          const decoder = new ImageDecoder({ data: blob.stream(), type: blob.type });
          await decoder.completed;
          const track = decoder.tracks.selectedTrack;
          if (!track || track.frameCount <= 1) { decoder.close?.(); return null; }
          const frames = [];
          for (let i = 0; i < track.frameCount; i++) {
            const r = await decoder.decode({ frameIndex: i });
            frames.push(r.image);
          }
          decoder.close?.();
          return { frames };
        } catch {
          return null;
        }
      }
      function closeStepVideo(stepIdx) {
        const v = stepVideoFrames[stepIdx];
        if (!v) return;
        for (const f of v.frames) { try { f.close(); } catch {} }
        stepVideoFrames[stepIdx] = null;
      }
      function activeStepIdx() {
        if (lockedStep != null) return lockedStep;
        if (hoverStep != null) return hoverStep;
        return lastCurrentStep;
      }

      // Pause state for the WebP path. MP4 uses videoEl.paused natively.
      let isPaused = false;
      let pauseAtMs = 0;
      function elapsedMs() {
        if (playbackStartMs == null) return 0;
        return (isPaused ? pauseAtMs : performance.now()) - playbackStartMs;
      }
      function mp4Active() {
        return currentMp4Url != null && Number.isFinite(visibleVideo.duration)
          && visibleVideo.duration > 0;
      }
      function clipDurationMs() {
        if (mp4Active()) return visibleVideo.duration * 1000;
        const v = stepVideoFrames[activeStepIdx()];
        if (v && v.frames.length > 1) return v.frames.length * (1000 / currentFps());
        return 0;
      }
      function getProgress() {
        if (mp4Active()) return visibleVideo.currentTime / visibleVideo.duration;
        const dur = clipDurationMs();
        if (dur <= 0 || playbackStartMs == null) return 0;
        return (elapsedMs() % dur) / dur;
      }
      function setProgress(pos) {
        pos = Math.max(0, Math.min(1, pos));
        const dur = clipDurationMs();
        if (dur <= 0) return;
        // Sync the global timer so cross-clip scrub picks up here.
        const ref = isPaused ? pauseAtMs : performance.now();
        if (playbackStartMs == null) playbackStartMs = ref;
        playbackStartMs = ref - pos * dur;
        if (mp4Active()) visibleVideo.currentTime = pos * visibleVideo.duration;
      }
      function togglePause() {
        const willPause = !isPaused;
        if (mp4Active()) {
          if (willPause) visibleVideo.pause();
          else visibleVideo.play().catch(() => {});
        }
        if (willPause) {
          pauseAtMs = performance.now();
          isPaused = true;
        } else {
          if (playbackStartMs != null) playbackStartMs += performance.now() - pauseAtMs;
          isPaused = false;
        }
        scrubBar.classList.toggle("h3-pov-paused", isPaused);
      }
      let scrubRafId = null;
      function tickScrub() {
        scrubRafId = requestAnimationFrame(tickScrub);
        const dur = clipDurationMs();
        if (dur > 0) {
          if (scrubBar.style.display === "none") scrubBar.style.display = "block";
          scrubFill.style.width = (getProgress() * 100) + "%";
        } else if (scrubBar.style.display !== "none") {
          scrubBar.style.display = "none";
        }
        if (mp4Active() && bakedFps != null && bakedFps > 0) {
          const rate = currentFps() / bakedFps;
          if (Math.abs(visibleVideo.playbackRate - rate) > 0.001) {
            visibleVideo.playbackRate = rate;
          }
        }
      }
      scrubRafId = requestAnimationFrame(tickScrub);

      function drawCurrentVideoFrame() {
        const idx = activeStepIdx();
        const v = stepVideoFrames[idx];
        if (!v) return false;
        if (playbackStartMs == null) playbackStartMs = performance.now();
        const frameDurMs = 1000 / currentFps();
        const totalMs = v.frames.length * frameDurMs;
        const elapsed = elapsedMs() % totalMs;
        const fIdx = Math.min(v.frames.length - 1, Math.floor(elapsed / frameDurMs));
        const frame = v.frames[fIdx];
        if (videoCanvas.width !== frame.displayWidth || videoCanvas.height !== frame.displayHeight) {
          videoCanvas.width = frame.displayWidth;
          videoCanvas.height = frame.displayHeight;
        }
        videoCtx.drawImage(frame, 0, 0);
        return true;
      }
      function startVideoLoop() {
        if (videoRafId != null) return;
        const tick = () => {
          videoRafId = requestAnimationFrame(tick);
          if (!drawCurrentVideoFrame()) return;
          if (videoCanvas.style.opacity !== "1") {
            videoCanvas.style.opacity = "1";
            imgA.style.opacity = "0";
            imgB.style.opacity = "0";
          }
        };
        videoRafId = requestAnimationFrame(tick);
      }
      function stopVideoLoop() {
        if (videoRafId != null) { cancelAnimationFrame(videoRafId); videoRafId = null; }
        videoCanvas.style.opacity = "0";
        visibleImg.style.opacity = "1";
      }
      // seeked -> rVFC -> double-rAF before hiding the old video, to avoid a black gap.
      function showMp4(url) {
        if (url === currentMp4Url) return;
        currentMp4Url = url;
        if (playbackStartMs == null) playbackStartMs = performance.now();
        const target = pendingVideo;
        target.src = url;
        const promote = () => {
          if (currentMp4Url !== url || target !== pendingVideo) return;
          // Re-snap to NOW — the initial seek drifted during load/paint. Paused-aware.
          try {
            const dur = target.duration;
            if (Number.isFinite(dur) && dur > 0) {
              target.currentTime = (elapsedMs() / 1000) % dur;
            }
          } catch {}
          target.style.opacity = "1";
          requestAnimationFrame(() => requestAnimationFrame(() => {
            if (target !== visibleVideo && currentMp4Url === url) {
              visibleVideo.style.opacity = "0";
              visibleVideo.pause();
              const prev = visibleVideo;
              visibleVideo = target;
              pendingVideo = prev;
              imgA.style.opacity = "0";
              imgB.style.opacity = "0";
              videoCanvas.style.opacity = "0";
              if (isPaused) visibleVideo.pause();
            }
          }));
        };
        const afterSeek = () => {
          if (typeof target.requestVideoFrameCallback === "function") {
            target.requestVideoFrameCallback(() => promote());
          } else {
            promote();
          }
        };
        const onLoaded = () => {
          target.removeEventListener("loadeddata", onLoaded);
          try {
            const dur = target.duration;
            if (Number.isFinite(dur) && dur > 0) {
              target.addEventListener("seeked", afterSeek, { once: true });
              target.currentTime = (elapsedMs() / 1000) % dur;
            } else {
              afterSeek();
            }
            // Must play during load so rVFC fires; we re-pause after promote.
            target.play().catch(() => {});
          } catch { promote(); }
        };
        if (target.readyState >= 2) onLoaded();
        else target.addEventListener("loadeddata", onLoaded, { once: true });
      }
      function hideMp4() {
        visibleVideo.pause();
        visibleVideo.style.opacity = "0";
        pendingVideo.pause();
        pendingVideo.style.opacity = "0";
        visibleImg.style.opacity = "1";
        currentMp4Url = null;
      }
      function setStepBlob(stepIdx, blob) {
        const url = URL.createObjectURL(blob);
        if (placeholder.parentNode) placeholder.remove();
        if (blob.type === "video/mp4") {
          if (stepMp4Urls[stepIdx]) {
            try { URL.revokeObjectURL(stepMp4Urls[stepIdx]); } catch {}
          }
          stepMp4Urls[stepIdx] = url;
          liveMp4Url = url;
          if (hoverStep == null && lockedStep == null) showMp4(url);
          return;
        }
        if (stepBlobUrls[stepIdx]) {
          try { URL.revokeObjectURL(stepBlobUrls[stepIdx]); } catch {}
        }
        stepBlobUrls[stepIdx] = url;
        liveBlobUrl = url;
        if (blob.type === "image/webp") {
          decodeAnimatedBlob(blob).then(v => {
            if (!v) {
              // Single-frame webp — show via the img path.
              if (hoverStep == null && lockedStep == null && liveBlobUrl === url) {
                showLiveFrame(url);
              }
              return;
            }
            closeStepVideo(stepIdx);
            stepVideoFrames[stepIdx] = v;
            startVideoLoop();
          });
        } else if (hoverStep == null && lockedStep == null) {
          showLiveFrame(url);
        }
      }
      function resetFrames() {
        for (const u of stepBlobUrls) { if (u) try { URL.revokeObjectURL(u); } catch {} }
        stepBlobUrls.length = 0;
        liveBlobUrl = null;
        for (let i = 0; i < stepVideoFrames.length; i++) closeStepVideo(i);
        stepVideoFrames.length = 0;
        stopVideoLoop();
        for (const u of stepMp4Urls) { if (u) try { URL.revokeObjectURL(u); } catch {} }
        stepMp4Urls.length = 0;
        liveMp4Url = null;
        hideMp4();
        playbackStartMs = null;
        bakedFps = null;
        isPaused = false;
        scrubBar.classList.remove("h3-pov-paused");
      }

      function redrawSd() {
        const xSteps = xStepCount();
        drawSigmaDeltaGraph(sdRow.canvas, cachedSigmas, history.delta, lastCurrentStep,
          xSteps, hoverStep, lockedStep);
        drawLineGraph(timeRow.canvas, history.stepMs, xSteps, hoverStep, lockedStep);
        updateSdHeader();
        renderTime();
        // Scrub priority: locked > hover > live. Locked persists past mouseleave.
        const displayStep = lockedStep != null ? lockedStep : hoverStep;
        const targetIdx = displayStep != null ? displayStep : lastCurrentStep;
        if (stepMp4Urls[targetIdx]) {
          stopVideoLoop();
          showMp4(stepMp4Urls[targetIdx]);
        } else if (stepVideoFrames[targetIdx]) {
          hideMp4();
          startVideoLoop();
        } else if (displayStep != null && stepBlobUrls[displayStep]) {
          stopVideoLoop();
          hideMp4();
          visibleImg.src = stepBlobUrls[displayStep];
        } else if (liveMp4Url) {
          stopVideoLoop();
          showMp4(liveMp4Url);
        } else if (liveBlobUrl) {
          stopVideoLoop();
          hideMp4();
          showLiveFrame(liveBlobUrl);
        }
      }

      // ---- pointer input -----------------------------------------------------------
      // Document-level CAPTURE phase with manual rect hit-testing, rather than listeners
      // on the elements themselves.
      //
      // Mouse events are not reliably delivered to a DOM widget's own elements: depending
      // on ComfyUI frontend version and zoom level the widget layer can be non-interactive
      // or sit under the graph canvas for hit-testing, and then an element-level
      // "mousemove" simply never fires — hover appears dead and the header keeps showing
      // the live step wherever you point. pointer-events:auto does not fix that, because
      // the problem is delivery rather than CSS. Capture on document fires before anything
      // downstream can swallow it, and getBoundingClientRect gives the box the browser
      // actually laid the element out in.
      function inRect(elm, ev) {
        if (!elm) return false;
        const r = elm.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;   // hidden / collapsed
        return ev.clientX >= r.left && ev.clientX <= r.right
            && ev.clientY >= r.top && ev.clientY <= r.bottom;
      }
      function cellUnderPointer(ev) {
        for (const cell of [sdRow, timeRow]) {
          if (inRect(cell.canvas, ev)) return cell;
        }
        return null;
      }
      function stepFromEvent(ev, canvas) {
        const rect = canvas.getBoundingClientRect();
        // rect carries the graph zoom; GRAPH_PAD_X is in the canvas's own css px. Deriving
        // the scale from rect.width/clientWidth (clientWidth ignores transforms) keeps the
        // mapping honest when zoomed, instead of the cursor drifting off the line.
        const scale = rect.width / Math.max(1, canvas.clientWidth || rect.width);
        const pad = GRAPH_PAD_X * scale;
        const iW = Math.max(1, rect.width - 2 * pad);
        const xSteps = xStepCount();
        const fx = (ev.clientX - rect.left - pad) / iW;
        return Math.max(0, Math.min(xSteps - 1, Math.round(fx * (xSteps - 1))));
      }

      let mouseOverPanel = false;
      let scrubDragging = false;
      const onDocMouseMove = (ev) => {
        // Runs for every mouse move anywhere in the app and getBoundingClientRect forces
        // layout, so test the whole widget once first: almost every event is nowhere near
        // this node and can be dropped for the cost of one rect instead of several.
        if (!inRect(root, ev)) {
          mouseOverPanel = false;
          if (hoverStep != null) { hoverStep = null; redrawSd(); }
          return;
        }
        mouseOverPanel = true;
        const cell = cellUnderPointer(ev);
        if (!cell) {
          if (hoverStep != null) { hoverStep = null; redrawSd(); }
          return;
        }
        const idx = stepFromEvent(ev, cell.canvas);
        if (idx !== hoverStep) { hoverStep = idx; redrawSd(); }
      };

      const onDocMouseDown = (ev) => {
        if (ev.button !== 0 || !inRect(root, ev)) return;

        if (inRect(collapseBtn, ev)) {
          ev.preventDefault();
          ev.stopPropagation();
          setCollapsed(!panel.classList.contains("h3-pov-collapsed"), true);
          return;
        }

        if (inRect(settingsBtn, ev)) {
          ev.preventDefault();
          ev.stopPropagation();
          openSettingsModal(node);
          return;
        }

        if (inRect(scrubBar, ev)) {
          ev.preventDefault();
          ev.stopPropagation();
          scrubDragging = true;
          const rect = scrubBar.getBoundingClientRect();
          const seek = (e) => setProgress((e.clientX - rect.left) / rect.width);
          seek(ev);
          const move = (e) => seek(e);
          const up = () => {
            scrubDragging = false;
            document.removeEventListener("mousemove", move, true);
            document.removeEventListener("mouseup", up, true);
          };
          document.addEventListener("mousemove", move, true);
          document.addEventListener("mouseup", up, true);
          return;
        }

        if (inRect(grip, ev)) {
          ev.preventDefault();
          ev.stopPropagation();
          const startY = ev.clientY;
          const startH = panel.offsetHeight;
          const gripRect = grip.getBoundingClientRect();
          const scale = gripRect.height / Math.max(1, grip.offsetHeight || gripRect.height);
          const move = (e) => {
            const dy = (e.clientY - startY) / (scale || 1);
            const rootRect = root.getBoundingClientRect();
            const maxH = Math.max(MIN_PANEL_H, rootRect.height / (scale || 1) - 80);
            panel.style.height = Math.max(MIN_PANEL_H, Math.min(maxH, startH - dy)) + "px";
            redrawSd();
          };
          const up = () => {
            document.removeEventListener("mousemove", move, true);
            document.removeEventListener("mouseup", up, true);
            node.properties.h3PanelH = panel.offsetHeight;
            node.graph?.change?.();
          };
          document.addEventListener("mousemove", move, true);
          document.addEventListener("mouseup", up, true);
          return;
        }

        // Anywhere over a graph: swallow so LiteGraph doesn't drag the node instead.
        if (cellUnderPointer(ev)) ev.stopPropagation();
      };

      const onDocClick = (ev) => {
        if (!inRect(root, ev)) return;
        if (inRect(collapseBtn, ev) || inRect(settingsBtn, ev) || inRect(scrubBar, ev)) {
          ev.preventDefault();
          ev.stopPropagation();
          return;
        }
        const cell = cellUnderPointer(ev);
        if (cell) {
          ev.preventDefault();
          ev.stopPropagation();
          if (cell === timeRow) {
            timeUnitSeconds = !timeUnitSeconds;
            renderTime();
          } else {
            lockedStep = lockedStep != null ? null : stepFromEvent(ev, cell.canvas);
            redrawSd();
          }
          return;
        }
        // Click on the preview frame toggles pause (animated content only).
        if (inRect(imageArea, ev) && !scrubDragging && clipDurationMs() > 0) {
          ev.preventDefault();
          ev.stopPropagation();
          togglePause();
        }
      };

      // Arrow-key scrub / space to pause, gated on the pointer being over the panel so
      // ComfyUI's global shortcuts aren't shadowed anywhere else on the canvas.
      const onKey = (ev) => {
        if (!mouseOverPanel) return;
        if (ev.key === " " && clipDurationMs() > 0) {
          ev.preventDefault();
          ev.stopPropagation();
          togglePause();
          return;
        }
        if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
        const xSteps = xStepCount();
        const cur = lockedStep != null ? lockedStep
          : (hoverStep != null ? hoverStep : lastCurrentStep);
        ev.preventDefault();
        ev.stopPropagation();
        lockedStep = Math.max(0, Math.min(xSteps - 1,
          Math.max(0, cur) + (ev.key === "ArrowRight" ? 1 : -1)));
        redrawSd();
      };

      document.addEventListener("mousemove", onDocMouseMove, true);
      document.addEventListener("mousedown", onDocMouseDown, true);
      document.addEventListener("click", onDocClick, true);
      document.addEventListener("keydown", onKey, true);

      addWheelPassthrough(root);

      node.addDOMWidget("preview", "h3_preview", root, { serialize: false });
      node.setSize([Math.max(node.size?.[0] ?? 380, 380), Math.max(node.size?.[1] ?? 480, 480)]);

      function resetHistory() {
        history.stepMs.length = 0;
        history.delta.length = 0;
        resetFrames();
        // Clear hover/lock — setStepBlob skips updates while either is set.
        hoverStep = null;
        lockedStep = null;
        lastCurrentStep = -1;
        // Allow the next run to re-pick its time unit from the first sample.
        lastStepMs = null;
      }

      node._h3PreviewHandler = (data) => {
        try {
          if (Array.isArray(data.sigmas) && data.sigmas.length > 1) {
            cachedSigmas = data.sigmas;
            resetHistory();
          }
          // The rate the server derived for this clip; drives both playback paths.
          if (typeof data.fps === "number" && data.fps > 0) {
            serverFps = data.fps;
            bakedFps = data.fps;
          }
          // Indexed by boundary; step 0 = initial noise (image optional).
          if (typeof data.image === "string") {
            const mime = typeof data.mime === "string" ? data.mime : "image/jpeg";
            setStepBlob(data.step, b64ToBlob(data.image, mime));
          }

          totalSteps = data.total || totalSteps;

          if (data.step_ms != null) history.stepMs.push(data.step_ms);
          if (data.delta != null) history.delta.push(data.delta);

          // Auto-pick unit from the first sample so slow steps default to seconds.
          if (lastStepMs == null && data.step_ms != null && data.step_ms > 1500) {
            timeUnitSeconds = true;
          }
          lastStepMs = data.step_ms;
          lastAvgStepMs = data.avg_step_ms;
          lastStep = data.step;
          lastTotal = data.total;
          lastW = data.w;
          lastH = data.h;
          // data.step is 1-based; set BEFORE renderTime so stepTimeForDisplay sees it.
          lastCurrentStep = data.step;
          redrawSd();
        } catch (err) {
          console.warn("[H3PreviewOverride] preview decode failed:", err);
        }
      };

      chainCallback(node, "onRemoved", function () {
        node._h3PreviewHandler = null;
        closeSettingsModal(node);
        document.removeEventListener("mousemove", onDocMouseMove, true);
        document.removeEventListener("mousedown", onDocMouseDown, true);
        document.removeEventListener("click", onDocClick, true);
        document.removeEventListener("keydown", onKey, true);
        resetFrames();
        stopVideoLoop();
        if (scrubRafId != null) cancelAnimationFrame(scrubRafId);
      });

      redrawSd();
    });
  },
});
