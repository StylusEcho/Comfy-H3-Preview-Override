// In-node live preview for MiniMax H3 Preview Plus.
// The Python side sends one animated WebP per rendered preview over the
// "minimax_h3_preview" event, plus (once, up front) the σ schedule for the whole run and
// (on every rendered preview) a Δ/step-time reading — we drop the WebP into a DOM widget
// on the node and the graph data into a pair of interactive, side-by-side canvases below
// it, ported from KJNodes' Preview Override node (github.com/kijai/ComfyUI-KJNodes):
// hover a graph to scrub to any step, click to lock a step for inspection, ←/→ to step
// while locked, click the step-time graph to toggle ms/s, and drag the grip above the
// graphs to resize them vertically — same interaction as KJNodes' own panel-height grip.
// The payload shape is identical for all three decode modes (latent2rgb, tiny vae/taeh3,
// vae) — this file doesn't need to know which one produced a given frame.
//
// All of the node's schema widgets (decode, tiny_vae, preview_target, preview_frames,
// ...) are pulled off the node body and shown in a popup instead, opened from a single
// "⚙ Settings" button. This is a *display* change only: the same widget objects back
// both the popup's controls and (as always) the values ComfyUI serialises into the
// workflow and sends to the Python node — the popup just writes to `widget.value`
// directly rather than the widget drawing its own row on the node. The one exception is
// `show_vae_input`, which also toggles the actual `vae` socket on the node face.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_TYPE = "MiniMaxH3PreviewPlusCS";
const IDLE_TEXT = "waiting for sample…";
const GRAPH_PAD_X = 4;
const GRAPH_PAD_Y = 3;

function fmt(n, d) {
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

// ------------------------------------------------------------------------ graph canvases
// Ported from KJNodes' preview_override.js, minus the SamplerDetailBoost curve overlay
// (that's a different node's feature) and the per-step image-frame cache/scrub (this node
// keeps a single always-current preview image rather than caching one per step).

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
  ctx.strokeStyle = "#242424";
  ctx.lineWidth = 1;
  for (let g = 1; g < 4; g++) {
    const y = Math.round(padY + (g / 4) * (H - 2 * padY)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(W - padX, y);
    ctx.stroke();
  }
}

// σ + Δ overlaid; each series normalised to its own max for comparable shapes.
function drawSigmaDeltaGraph(canvas, sigmas, deltas, step, totalSteps, hoverStep, lockedStep) {
  const { ctx, W, H } = syncCanvasDPR(canvas);
  const padX = GRAPH_PAD_X, padY = GRAPH_PAD_Y;
  const iW = W - 2 * padX, iH = H - 2 * padY;
  ctx.clearRect(0, 0, W, H);
  drawGridlines(ctx, W, H, padX, padY);

  const n = sigmas?.length || 0;
  const xSteps = Math.max(totalSteps || n, n, deltas?.length || 0);
  const xAt = (i) => padX + (i / Math.max(1, xSteps - 1)) * iW;

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
      const px = padX + (i / (n - 1)) * iW;
      const py = sYAt(sigmas[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    const i = Math.max(0, Math.min(n - 1, step));
    const mx = padX + (i / Math.max(1, n - 1)) * iW;
    const my = sYAt(sigmas[i]);
    ctx.fillStyle = "#d0d0d0";
    ctx.beginPath();
    ctx.arc(mx, my, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  let dYAt = null;
  if (deltas && deltas.length >= 1) {
    let dMax = -Infinity;
    for (const v of deltas) if (Number.isFinite(v) && v > dMax) dMax = v;
    const dRange = Math.max(dMax, 1e-6);
    dYAt = (v) => padY + (1 - v / dRange) * iH;

    // delta[i] is plotted at boundary (i+1); flat-extend delta[0] back to boundary 0.
    const lastB = deltas.length;
    ctx.beginPath();
    ctx.moveTo(xAt(0), H - padY);
    ctx.lineTo(xAt(0), dYAt(deltas[0]));
    for (let i = 0; i < deltas.length; i++) ctx.lineTo(xAt(i + 1), dYAt(deltas[i]));
    ctx.lineTo(xAt(lastB), H - padY);
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
    if (sYAt && hoverStep < n) {
      ctx.fillStyle = "#d0d0d0";
      ctx.beginPath();
      ctx.arc(hx - 0.5, sYAt(sigmas[hoverStep]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    // delta[k-1] is plotted at boundary k.
    if (dYAt && hoverStep >= 1 && (hoverStep - 1) < deltas.length) {
      ctx.fillStyle = "#e67e22";
      ctx.beginPath();
      ctx.arc(hx - 0.5, dYAt(deltas[hoverStep - 1]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// totalSteps fixes the x-axis so the line grows left-to-right, not stretching to fill.
function drawLineGraph(canvas, values, totalSteps) {
  const { ctx, W, H } = syncCanvasDPR(canvas);
  const padX = GRAPH_PAD_X, padY = GRAPH_PAD_Y;
  const iW = W - 2 * padX, iH = H - 2 * padY;
  ctx.clearRect(0, 0, W, H);
  drawGridlines(ctx, W, H, padX, padY);
  if (!values || values.length < 1) return;

  let vMax = -Infinity, vMin = Infinity;
  for (const v of values) { if (v > vMax) vMax = v; if (v < vMin) vMin = v; }
  if (vMin > 0) vMin = 0;
  const vRange = Math.max(vMax - vMin, 1e-6);

  const xSteps = Math.max(totalSteps || values.length, values.length);

  ctx.beginPath();
  ctx.moveTo(padX, H - padY);
  for (let i = 0; i < values.length; i++) {
    const px = padX + (i / Math.max(1, xSteps - 1)) * iW;
    const py = padY + (1 - (values[i] - vMin) / vRange) * iH;
    ctx.lineTo(px, py);
  }
  ctx.lineTo(padX + ((values.length - 1) / Math.max(1, xSteps - 1)) * iW, H - padY);
  ctx.closePath();
  ctx.fillStyle = "rgba(230, 126, 34, 0.15)";
  ctx.fill();

  ctx.strokeStyle = "#e67e22";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const px = padX + (i / Math.max(1, xSteps - 1)) * iW;
    const py = padY + (1 - (values[i] - vMin) / vRange) * iH;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

// --------------------------------------------------------------------------- panel + grip
// Layout: image area (flexes to fill leftover space) → status line → a drag grip → a
// fixed-height graphs panel holding the two graph cells SIDE BY SIDE. Dragging the grip
// resizes the graphs panel (shrinking/growing the image area to compensate) and persists
// the chosen height on `node.properties`, exactly like KJNodes' own Preview Override
// panel-height grip (`kjPovPanelH`) — properties round-trip through workflow save/load
// automatically, independent of the widgets_values array.

const DEFAULT_GRAPHS_H = 130;
const MIN_GRAPHS_H = 50;

function buildPanel(node) {
  const root = document.createElement("div");
  Object.assign(root.style, {
    display: "flex", flexDirection: "column", gap: "4px",
    boxSizing: "border-box", width: "100%", height: "100%",
  });

  const frame = document.createElement("div");
  Object.assign(frame.style, {
    position: "relative", width: "100%", minHeight: "100px", flex: "1 1 auto",
    background: "#141414", border: "1px solid #3a3a3a", borderRadius: "6px",
    display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden",
  });

  const img = document.createElement("img");
  Object.assign(img.style, {
    maxWidth: "100%", maxHeight: "100%", objectFit: "contain",
    display: "none", imageRendering: "auto",
  });

  const idle = document.createElement("div");
  idle.textContent = IDLE_TEXT;
  Object.assign(idle.style, { color: "#6a6a6a", fontSize: "11px", fontStyle: "italic" });

  frame.appendChild(img);
  frame.appendChild(idle);

  const status = document.createElement("div");
  Object.assign(status.style, {
    display: "flex", justifyContent: "space-between", gap: "8px", flex: "0 0 auto",
    color: "#8a8a8a", fontSize: "10px", fontFamily: "monospace", padding: "0 2px",
  });
  const left = document.createElement("span");
  const right = document.createElement("span");
  left.textContent = "idle";
  status.appendChild(left);
  status.appendChild(right);

  const grip = document.createElement("div");
  Object.assign(grip.style, {
    flex: "0 0 6px", background: "#242424", cursor: "ns-resize",
    borderRadius: "2px", position: "relative",
  });
  grip.title = "Drag to resize the graphs";
  const gripBar = document.createElement("div");
  Object.assign(gripBar.style, {
    position: "absolute", left: "50%", top: "50%", width: "26px", height: "2px",
    background: "#555", borderRadius: "1px", transform: "translate(-50%, -50%)",
  });
  grip.appendChild(gripBar);
  grip.addEventListener("mouseenter", () => { gripBar.style.background = "#888"; });
  grip.addEventListener("mouseleave", () => { gripBar.style.background = "#555"; });

  const graphsPanel = document.createElement("div");
  Object.assign(graphsPanel.style, {
    flex: "0 0 auto", height: DEFAULT_GRAPHS_H + "px", minHeight: MIN_GRAPHS_H + "px",
    boxSizing: "border-box",
  });

  // Side by side, like KJNodes' `kj-pov-graphs-grid`.
  const graphsRow = document.createElement("div");
  Object.assign(graphsRow.style, {
    display: "flex", flexDirection: "row", gap: "5px", height: "100%", boxSizing: "border-box",
  });
  graphsPanel.appendChild(graphsRow);

  function makeGraphCell(labelHtml, cursor) {
    const cell = document.createElement("div");
    Object.assign(cell.style, {
      flex: "1 1 0", minWidth: "0", display: "flex", flexDirection: "column",
      background: "#0e0e0e", border: "1px solid #242424", borderRadius: "4px",
      overflow: "hidden",
    });
    const head = document.createElement("div");
    Object.assign(head.style, {
      flex: "0 0 auto", display: "flex", justifyContent: "space-between", alignItems: "baseline",
      padding: "2px 6px", background: "#181818", borderBottom: "1px solid #242424",
      fontSize: "10px", fontFamily: "monospace", gap: "6px",
    });
    const lbl = document.createElement("span");
    lbl.innerHTML = labelHtml;
    Object.assign(lbl.style, { color: "#8a8a8a", letterSpacing: "0.03em", flex: "0 0 auto" });
    const val = document.createElement("span");
    Object.assign(val.style, {
      color: "#c0c0c0", flex: "1 1 auto", textAlign: "right",
      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
    });
    head.appendChild(lbl);
    head.appendChild(val);
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
      flex: "1 1 auto", minHeight: "0", display: "block", width: "100%",
      cursor: cursor || "default",
    });
    cell.appendChild(head);
    cell.appendChild(canvas);
    graphsRow.appendChild(cell);
    return { cell, lbl, val, canvas };
  }
  const sdRow = makeGraphCell(
    '<span style="color:#d0d0d0">σ</span> / <span style="color:#e67e22">Δ</span>', "crosshair");
  const timeRow = makeGraphCell("step time (ms)", "pointer");

  node.properties = node.properties || {};
  if (typeof node.properties.h3ppGraphsH === "number") {
    graphsPanel.style.height = node.properties.h3ppGraphsH + "px";
  }
  grip.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const startY = e.clientY;
    const startH = graphsPanel.offsetHeight;
    const move = (ev) => {
      const scale = app.canvas?.ds?.scale || 1;
      const dy = (ev.clientY - startY) / scale;
      const rootRect = root.getBoundingClientRect();
      // Leave the image area at least ~80px so it never fully collapses under the graphs.
      const maxH = Math.max(MIN_GRAPHS_H, rootRect.height / scale - 80);
      const newH = Math.max(MIN_GRAPHS_H, Math.min(maxH, startH - dy));
      graphsPanel.style.height = newH + "px";
    };
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
      node.properties.h3ppGraphsH = graphsPanel.offsetHeight;
      node.graph?.change?.();
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });

  root.appendChild(frame);
  root.appendChild(status);
  root.appendChild(grip);
  root.appendChild(graphsPanel);
  return { root, img, idle, left, right, graphs: graphsRow, graphsPanel, sdRow, timeRow };
}

// ------------------------------------------------------------------------- settings popup

// Hides a widget's own row on the node body without touching its value, its callback,
// or how it serialises — `hidden` is read by LiteGraph's widget layout/draw pass, and
// the zeroed computeSize is a belt-and-suspenders fallback for any pass that only
// consults size. Nothing here changes `widget.type`, so the widget keeps behaving
// exactly like a normal combo/number/toggle widget everywhere except on-canvas.
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
  title.textContent = "MiniMax H3 Preview Plus — Settings";
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

// ------------------------------------------------------------------- graph state per node
// Builds the closures that own one node's graph history/interaction and returns a data
// handler ("minimax_h3_preview" events feed straight in) plus a reset (for a fresh run).
// Ported from KJNodes' preview_override.js hover/lock/arrow-key scrubbing, scoped down to
// what the graphs need (no per-step image cache — this node's image stays single/live).

function setupGraphs(node, panel) {
  let cachedSigmas = null;
  const history = { stepMs: [], delta: [] };
  let totalSteps = 0;
  let lastCurrentStep = -1;
  let hoverStep = null;
  let lockedStep = null;      // click-locked step; survives mouseleave for inspection
  let timeUnitSeconds = false;
  let lastAvgStepMs = null, lastStep = null, lastTotal = null;

  function fmtTime(ms) {
    if (ms == null || !Number.isFinite(ms)) return "—";
    return timeUnitSeconds ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(0)}ms`;
  }
  // history.stepMs[k-1] = the wait before rendered step k.
  function stepTimeForDisplay() {
    let idx = lockedStep != null ? lockedStep : (hoverStep != null ? hoverStep : lastCurrentStep);
    const tIdx = idx - 1;
    return (tIdx >= 0 && tIdx < history.stepMs.length) ? history.stepMs[tIdx] : null;
  }
  function renderTimeHeader() {
    panel.timeRow.lbl.textContent = timeUnitSeconds ? "step time (s)" : "step time (ms)";
    let text = fmtTime(stepTimeForDisplay());
    if (lastAvgStepMs != null && lastTotal != null && lastStep != null) {
      const eta = Math.max(0, lastTotal - lastStep) * lastAvgStepMs / 1000;
      const avgTxt = timeUnitSeconds
        ? `${(lastAvgStepMs / 1000).toFixed(2)}s/render`
        : `${lastAvgStepMs.toFixed(0)}ms/render`;
      text += ` · ${avgTxt} · ETA ${eta.toFixed(1)}s`;
    }
    panel.timeRow.val.textContent = text;
  }
  function updateSdHeader() {
    // Display priority: hover > locked > live. Hover gets [k] prefix, locked gets 🔒[k].
    let idx = null, prefix = "";
    if (hoverStep != null && cachedSigmas && hoverStep < cachedSigmas.length) {
      idx = hoverStep;
      prefix = `[${idx}] `;
    } else if (lockedStep != null && cachedSigmas && lockedStep < cachedSigmas.length) {
      idx = lockedStep;
      prefix = `🔒[${idx}] `;
    } else if (lastCurrentStep >= 0 && cachedSigmas) {
      idx = Math.min(lastCurrentStep, cachedSigmas.length - 1);
    }
    if (idx == null) { panel.sdRow.val.textContent = ""; return; }
    const sig = cachedSigmas[idx];
    const dIdx = idx - 1;
    const d = (dIdx >= 0 && dIdx < history.delta.length) ? history.delta[dIdx] : null;
    panel.sdRow.val.textContent = `${prefix}${fmt(sig, 3)} / ${fmt(d, 3)}`;
  }
  function redraw() {
    drawSigmaDeltaGraph(panel.sdRow.canvas, cachedSigmas, history.delta, lastCurrentStep,
      totalSteps, hoverStep, lockedStep);
    drawLineGraph(panel.timeRow.canvas, history.stepMs, totalSteps);
    updateSdHeader();
    renderTimeHeader();
  }

  function xStepCount() {
    return Math.max(totalSteps || cachedSigmas?.length || 0, cachedSigmas?.length || 0,
      history.delta.length);
  }
  function stepFromEvent(ev, canvas) {
    const rect = canvas.getBoundingClientRect();
    const iW = Math.max(1, rect.width - 2 * GRAPH_PAD_X);
    const xSteps = xStepCount();
    const fx = (ev.clientX - rect.left - GRAPH_PAD_X) / iW;
    return Math.max(0, Math.min(xSteps - 1, Math.round(fx * (xSteps - 1))));
  }

  panel.sdRow.canvas.addEventListener("mousemove", (ev) => {
    if (!cachedSigmas) return;
    const idx = stepFromEvent(ev, panel.sdRow.canvas);
    if (idx !== hoverStep) { hoverStep = idx; redraw(); }
  });
  panel.sdRow.canvas.addEventListener("mouseleave", () => {
    if (hoverStep != null) { hoverStep = null; redraw(); }
  });
  panel.sdRow.canvas.addEventListener("mousedown", (ev) => ev.stopPropagation());
  panel.sdRow.canvas.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    if (lockedStep != null) {
      lockedStep = null;
    } else if (cachedSigmas) {
      lockedStep = stepFromEvent(ev, panel.sdRow.canvas);
    }
    redraw();
  });

  const toggleTimeUnit = (ev) => {
    ev.stopPropagation();
    timeUnitSeconds = !timeUnitSeconds;
    renderTimeHeader();
  };
  panel.timeRow.canvas.addEventListener("click", toggleTimeUnit);
  panel.timeRow.lbl.addEventListener("click", toggleTimeUnit);
  panel.timeRow.val.addEventListener("click", toggleTimeUnit);
  panel.timeRow.lbl.style.cursor = "pointer";
  panel.timeRow.val.style.cursor = "pointer";

  // Arrow-key scrub while locked, gated on hovering the graphs so global ComfyUI
  // shortcuts (e.g. arrow-key panning) aren't shadowed anywhere else on the canvas.
  let mouseOverGraphs = false;
  panel.graphs.addEventListener("mouseenter", () => { mouseOverGraphs = true; });
  panel.graphs.addEventListener("mouseleave", () => { mouseOverGraphs = false; });
  const onKey = (ev) => {
    if (!mouseOverGraphs || !cachedSigmas) return;
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    const xSteps = xStepCount();
    const cur = lockedStep != null ? lockedStep : (hoverStep != null ? hoverStep : lastCurrentStep);
    ev.preventDefault();
    ev.stopPropagation();
    lockedStep = Math.max(0, Math.min(xSteps - 1, cur + (ev.key === "ArrowRight" ? 1 : -1)));
    redraw();
  };
  document.addEventListener("keydown", onKey, true);
  node._mmxKeyHandler = onKey;

  function handleEvent(data) {
    try {
      if (Array.isArray(data.sigmas) && data.sigmas.length > 1) {
        cachedSigmas = data.sigmas;
        history.stepMs.length = 0;
        history.delta.length = 0;
        hoverStep = null;
        lockedStep = null;
      }
      if (data.total_steps != null) { totalSteps = data.total_steps; lastTotal = data.total_steps; }
      if (data.step_ms != null) history.stepMs.push(data.step_ms);
      if (data.delta != null) history.delta.push(data.delta);
      if (data.avg_step_ms != null) lastAvgStepMs = data.avg_step_ms;
      if (data.step != null) { lastStep = data.step; lastCurrentStep = data.step; }
      redraw();
    } catch (err) {
      console.warn("[H3PreviewPlus] graph update failed:", err);
    }
  }
  function reset() {
    cachedSigmas = null;
    history.stepMs.length = 0;
    history.delta.length = 0;
    totalSteps = 0;
    lastCurrentStep = -1;
    hoverStep = null;
    lockedStep = null;
    lastAvgStepMs = null;
    lastStep = null;
    lastTotal = null;
    redraw();
  }
  redraw(); // draw the empty grid immediately, before the first event arrives
  return { handleEvent, reset };
}

app.registerExtension({
  name: "MiniMaxH3PreviewPlusCS",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, arguments);

      // Everything on `this.widgets` right now is a schema-driven value widget (model
      // and vae are sockets, not widgets) — capture that exact list before adding the
      // button/DOM widgets below, then hide each one so it no longer draws a row.
      this._h3SettingsWidgets = (this.widgets || []).slice();
      for (const w of this._h3SettingsWidgets) hideWidget(w);

      // Appended last, after every schema widget, so a workflow saved before this popup
      // existed still maps its widgets_values onto the right inputs by position.
      const settingsBtn = this.addWidget("button", "⚙ Settings", null,
        () => openSettingsModal(this));
      settingsBtn.serialize = false;
      settingsBtn.tooltip = "Open all of this node's settings: decode mode, tiny_vae "
        + "checkpoint, preview target/timing/output, and whether the vae socket shows.";

      const panel = buildPanel(this);
      this._mmxPreview = panel;
      this._mmxGraphs = setupGraphs(this, panel);

      // Reflect the show_vae_input widget's starting value (its default, or a value a
      // saved workflow already restored onto it if this fires after configure).
      const showVaeWidget = this._h3SettingsWidgets.find((w) => w.name === "show_vae_input");
      if (showVaeWidget) syncVaeInputVisibility(this, showVaeWidget.value);

      // Height is declared through the DOM-widget layout API, not through computeSize.
      //
      // computeSize is the wrong hook here: LiteGraph evaluates it to get the node's
      // minimum height, so any height derived from node.size[1] ratchets — the minimum is
      // measured against the size you are trying to leave, and the node can only ever
      // grow. Core's multiline textarea sidesteps that by declaring a minHeight and no
      // maxHeight, which lets the layout engine hand it the leftover space instead. Same
      // deal here: one floor, no ceiling, resizable in both directions. The graphs panel's
      // own resize (the grip) happens *within* this floor — it reallocates space between
      // the image and the graphs, it doesn't grow the node on its own.
      const MIN_PREVIEW_H = 220;
      const widget = this.addDOMWidget("minimax_preview_ui", "minimax_preview_ui", panel.root, {
        getValue: () => "",
        setValue: () => {},
        getMinHeight: () => MIN_PREVIEW_H,
      });
      widget.serialize = false;

      // Everything that used to take a widget row is in the popup now, so the node only
      // needs to fit the button and the preview + graphs panel. Wide enough for two
      // side-by-side graph cells to stay readable.
      if (this.size[0] < 360) this.size[0] = 360;
      if (this.size[1] < 420) this.size[1] = 420;
    };

    // Heal combo widgets whose saved value is not one of their options.
    //
    // ComfyUI stores widgets_values positionally, so a workflow saved while the widget
    // list had a different shape hands every value after the change to the wrong input —
    // and a combo then refuses to run with "The value true is not available". The values
    // themselves are unrecoverable at that point; what matters is that the node comes back
    // usable instead of blocking the whole workflow. Runs over every widget regardless of
    // the settings popup, since hiding a widget's row doesn't change what it holds.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      for (const w of this.widgets || []) {
        const opts = w.options?.values;
        if (!Array.isArray(opts) || opts.length === 0) continue;
        if (opts.includes(w.value)) continue;
        const fallback = w.options?.default ?? opts[0];
        console.warn(
          `[H3PreviewPlus] ${NODE_TYPE}: saved value ${JSON.stringify(w.value)} is not a ` +
          `valid '${w.name}' — this workflow was saved against a different widget layout. ` +
          `Falling back to ${JSON.stringify(fallback)}; check the node's settings.`);
        w.value = fallback;
      }

      // node.properties (including the graphs-panel height) is restored by the base
      // configure() above; re-apply it to the actual DOM element here in case this node
      // built its panel before that restore happened.
      if (typeof this.properties?.h3ppGraphsH === "number" && this._mmxPreview?.graphsPanel) {
        this._mmxPreview.graphsPanel.style.height = this.properties.h3ppGraphsH + "px";
      }

      // Reconcile the vae socket with the restored show_vae_input value — whatever the
      // base configure() did with node.inputs, this makes the actual socket state match
      // the setting deterministically.
      const showVaeWidget = (this.widgets || []).find((w) => w.name === "show_vae_input");
      if (showVaeWidget) syncVaeInputVisibility(this, showVaeWidget.value);

      // A workflow saved by an older version of this node (before settings moved into
      // the popup, or before the graphs existed) may carry a stored size smaller than
      // this node needs today — never shrink it out from under the user, but do grow up
      // to today's minimum so the preview panel, graphs and button aren't left overflowing.
      const min = this.computeSize()[1];
      if (this.size[1] < min) {
        this.size[1] = min;
        // the draw loop only re-arranges widgets for nodes carrying this flag; without it
        // the body would redraw at the new height while the overlay keeps the old one
        this._widgetSlotsDirty = true;
        this.setDirtyCanvas?.(true, true);
      }
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      closeSettingsModal(this);
      if (this._mmxKeyHandler) {
        document.removeEventListener("keydown", this._mmxKeyHandler, true);
        this._mmxKeyHandler = null;
      }
      if (this._mmxPreview?.img?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(this._mmxPreview.img.src);
      }
      this._mmxPreview = null;
      this._mmxGraphs = null;
      return onRemoved?.apply(this, arguments);
    };
  },

  async setup() {
    api.addEventListener("minimax_h3_preview", (event) => {
      const d = event.detail || {};
      // node ids are strings server-side and numbers in the graph — compare loosely
      const node = app.graph?._nodes?.find((n) => String(n.id) === String(d.node_id));
      if (!node) return;

      // Fires on every payload — the σ-schedule init send and every rendered preview —
      // independent of whether this particular one carries an image.
      node._mmxGraphs?.handleEvent(d);

      const panel = node._mmxPreview;
      if (!panel || !d.webp) return;

      panel.img.src = "data:image/webp;base64," + d.webp;
      panel.img.style.display = "block";
      panel.idle.style.display = "none";
      const rate = Number(d.fps);
      const src = Number(d.source_fps);
      // A rate below the shot's own is not a fault, it is the "true speed" trade: the
      // sampled frames are spread over the clip's real length. Say which of the two you
      // are looking at, or the number reads as a setting being ignored.
      const slowed = Number.isFinite(src) && Math.abs(src - rate) > 0.05;
      const fpsText = slowed
        ? `${rate.toFixed(1)}fps of ${src.toFixed(0)} · true speed`
        : `${rate.toFixed(rate % 1 ? 1 : 0)}fps${d.playback === "source fps" ? " · source" : ""}`;
      panel.left.textContent = `step ${d.step}/${d.total_steps} · ${d.frames}f @${fpsText}`;
      // server-side cost of building this preview, not anything the browser spent
      const cost = Number(d.ms) >= 1000 ? `${(d.ms / 1000).toFixed(1)}s` : `${d.ms}ms`;
      panel.right.textContent = `render ${cost} · ${d.mode ? d.mode.split(" ")[0] : ""}`;
      node.setDirtyCanvas?.(true, false);
    });

    // reset the panels back to idle when a new run starts
    api.addEventListener("execution_start", () => {
      for (const n of app.graph?._nodes || []) {
        n._mmxGraphs?.reset();
        const p = n._mmxPreview;
        if (!p) continue;
        p.left.textContent = "waiting…";
        p.right.textContent = "";
      }
    });
  },
});
