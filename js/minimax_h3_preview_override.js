// In-node live preview for MiniMax H3 Preview Override.
// The Python side sends one animated WebP per sampled step over the "minimax_h3_preview"
// event; we drop it into a DOM widget on the node and show a small status line. The
// payload shape is identical for all three decode modes (latent2rgb, tiny vae/taeh3, vae)
// — this file doesn't need to know which one produced a given frame.
//
// All of the node's schema widgets (decode, tiny_vae, preview_target, preview_frames,
// ...) are pulled off the node body and shown in a popup instead, opened from a single
// "⚙ Settings" button. This is a *display* change only: the same widget objects back
// both the popup's controls and (as always) the values ComfyUI serialises into the
// workflow and sends to the Python node — the popup just writes to `widget.value`
// directly rather than the widget drawing its own row on the node.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_TYPE = "MiniMaxH3PreviewOverrideCS";
const IDLE_TEXT = "waiting for sample…";

function buildPanel() {
  const root = document.createElement("div");
  Object.assign(root.style, {
    display: "flex", flexDirection: "column", gap: "4px",
    boxSizing: "border-box", width: "100%", height: "100%",
  });

  const frame = document.createElement("div");
  Object.assign(frame.style, {
    position: "relative", width: "100%", minHeight: "160px", flex: "1",
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
    display: "flex", justifyContent: "space-between", gap: "8px",
    color: "#8a8a8a", fontSize: "10px", fontFamily: "monospace", padding: "0 2px",
  });
  const left = document.createElement("span");
  const right = document.createElement("span");
  left.textContent = "idle";
  status.appendChild(left);
  status.appendChild(right);

  root.appendChild(frame);
  root.appendChild(status);
  return { root, img, idle, left, right };
}

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

// One label + control (+ optional description) row per setting, built fresh each time
// the popup opens so it always reflects the widget's current value.
function buildSettingsRow(node, widget) {
  const row = document.createElement("div");
  Object.assign(row.style, {
    display: "flex", flexDirection: "column", gap: "3px",
    padding: "7px 0", borderBottom: "1px solid #2c2c2c",
  });

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
  labelRow.appendChild(control);
  row.appendChild(labelRow);

  const tip = widgetTooltip(widget);
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
  title.textContent = "MiniMax H3 Preview Override — Settings";
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
  name: "MiniMaxH3PreviewOverrideCS",

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

      const panel = buildPanel();
      this._mmxPreview = panel;

      // Height is declared through the DOM-widget layout API, not through computeSize.
      //
      // computeSize is the wrong hook here: LiteGraph evaluates it to get the node's
      // minimum height, so any height derived from node.size[1] ratchets — the minimum is
      // measured against the size you are trying to leave, and the node can only ever
      // grow. Core's multiline textarea sidesteps that by declaring a minHeight and no
      // maxHeight, which lets the layout engine hand it the leftover space instead. Same
      // deal here: one floor, no ceiling, resizable in both directions.
      const MIN_PREVIEW_H = 140;
      const widget = this.addDOMWidget("minimax_preview_ui", "minimax_preview_ui", panel.root, {
        getValue: () => "",
        setValue: () => {},
        getMinHeight: () => MIN_PREVIEW_H,
      });
      widget.serialize = false;

      // Everything that used to take a widget row is in the popup now, so the node only
      // needs to fit the button and the preview panel.
      if (this.size[0] < 300) this.size[0] = 300;
      if (this.size[1] < 260) this.size[1] = 260;
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
          `[H3PreviewOverride] ${NODE_TYPE}: saved value ${JSON.stringify(w.value)} is not a ` +
          `valid '${w.name}' — this workflow was saved against a different widget layout. ` +
          `Falling back to ${JSON.stringify(fallback)}; check the node's settings.`);
        w.value = fallback;
      }

      // A workflow saved by an older version of this node (before settings moved into
      // the popup, or before some widget existed) may carry a stored size taller than
      // this node needs today — never shrink it out from under the user, but do grow up
      // to today's minimum so the preview panel and button aren't left overflowing.
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
      if (this._mmxPreview?.img?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(this._mmxPreview.img.src);
      }
      this._mmxPreview = null;
      return onRemoved?.apply(this, arguments);
    };
  },

  async setup() {
    api.addEventListener("minimax_h3_preview", (event) => {
      const d = event.detail || {};
      // node ids are strings server-side and numbers in the graph — compare loosely
      const node = app.graph?._nodes?.find((n) => String(n.id) === String(d.node_id));
      const panel = node?._mmxPreview;
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
        const p = n._mmxPreview;
        if (!p) continue;
        p.left.textContent = "waiting…";
        p.right.textContent = "";
      }
    });
  },
});
