# Comfy-H3-Preview-Override

A single ComfyUI custom node: **MiniMax H3 Preview Plus**, combining

* the H3-aware live sampling preview from
  [ComfyUI-MiniMaxH3-Director](https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director)
  (unpacks MiniMax H3's packed audio+video latent so the preview shows the whole shot,
  not just the first latent frame), with
* the **taeh3** tiny-VAE decode path and the **interactive σ/Δ and step-time graphs**
  from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)' Preview Override
  node (a small trained temporal decoder for near-VAE colour accuracy at close to
  latent2rgb speed, plus a live, scrubbable read on the sampler's noise schedule and how
  much the shot is still changing).

Wire it between your model and the sampler and watch the shot denoise on the node
itself, with a choice of three decode modes and a pair of interactive graphs underneath.
All of the decode/timing/output settings live behind a single **⚙ Settings** button on
the node rather than as a stack of widget rows, so the node itself stays small — just
the button, the live preview, and the graphs.

## Why this exists

ComfyUI core ships `latent_rgb_factors` for MiniMax H3, so a preview isn't *missing* —
but core's own previewer renders `x0[0, :, 0]`, the first latent frame only, so you
watch one frozen still while a five-second shot samples.

KJNodes' **Model Preview Override** already solves the "show the whole clip" problem
well, including an optional taeh3/taehv tiny-VAE decode — but its video paths are gated
on LTX-specific checks (`_is_ltx_latent_format`, `_is_ltx2_diffusion_model`), and nothing
there unpacks H3's packed AV latent. On MiniMax it falls through to the same single
frame.

ComfyUI-MiniMaxH3-Director's **MiniMax H3 Preview Override** node fixes the unpacking,
but only offers `latent2rgb` and a full `vae` decode — nothing in between.

This node does both: H3's packed-latent unpacking, plus all three decode modes — under
its own name and node ID (**MiniMax H3 Preview Plus** / `MiniMaxH3PreviewPlusCS`), distinct
from the Director's own Preview Override node so the two packages can be installed
side by side.

## Decode modes

| `decode` | Cost | Quality | Notes |
|---|---|---|---|
| `latent2rgb (fast)` | ~free (one matmul) | rough colours | Default. Always available, no extra files. |
| `tiny vae (taeh3)` | small, roughly latent2rgb speed | near-VAE colours | Needs a taehv-format checkpoint sized for H3 (24 latent channels, patch size 2 — commonly named `taeh3`) in `models/vae_approx`, picked via the `tiny_vae` widget. See [Getting a taeh3 checkpoint](#getting-a-taeh3-checkpoint) below. |
| `vae (quality)` | real decode cost, tens of seconds per preview at high res | true colours | Needs `minimax_h3_video_vae` wired into `vae`. |

`tiny vae (taeh3)` behaves a little differently from the other two when previewing the
whole clip: because the decoder carries state frame-to-frame (it's a temporal model with
memory blocks, not a per-frame matmul), asking for the full latent length triggers an
H3-aware chunked decode that comes back at the *exact* real pixel-frame count H3 will
actually output — better timing accuracy than either of the other two modes, for
roughly the cost of `latent2rgb`. Asking for fewer frames than the full latent length
(via `preview_frames`) decodes a chronological *prefix* of that length rather than an
even thin across the clip, for the same reason.

## Requirements

* **ComfyUI ≥ 0.30.0** — H3 support, `comfy_api.latest` and the packed AV latent all
  landed in 0.30.
* For `decode='tiny vae (taeh3)'`: a ComfyUI core build that ships `comfy.taesd.taehv`
  (the same recent releases that added TAEHV/LTX2 video previews to core). If it's
  missing, the node logs a warning and falls back to `latent2rgb` at run time — it does
  not error out.
* **Python 3.10+** (ComfyUI's own environment).
* **No extra pip packages.** Everything imported ships with ComfyUI already.

## Installation

Clone into your `custom_nodes` folder and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StylusEcho/Comfy-H3-Preview-Override.git
```

### Compatibility with ComfyUI-MiniMaxH3-Director

This node registers under its **own node ID and display name**
(`MiniMaxH3PreviewPlusCS` / "MiniMax H3 Preview Plus"), separate from the
`MiniMaxH3PreviewOverrideCS` node shipped in ComfyUI-MiniMaxH3-Director — so you can
install **both** packages together without a duplicate-node-name conflict. Use the
Director for the Director, Retake Stitch, Enhance Prompt and Save Last Frame nodes, and
drop this node in wherever you'd have used its Preview Override.

> Earlier commits on this repo's PR reused the Director's node ID as a drop-in
> replacement. If you built a workflow against one of those, the node ID has since
> changed to `MiniMaxH3PreviewPlusCS` — delete the old node and re-add "MiniMax H3
> Preview Plus" from the node list; the widget values aren't recoverable automatically.

## Settings

Click **⚙ Settings** on the node to open a popup with every widget below — the node body
itself only ever shows the live preview panel, the graphs and the button. The popup edits
the same underlying widget values ComfyUI always serialised (nothing about how a
workflow saves or loads changed), so old saved workflows load their settings exactly as
before; you just no longer scroll a tall stack of rows to see or change them. Hover any
row (or its control) for a mouseover caption explaining that setting — the same text is
also shown underneath the row.

| Setting | What it does |
|---|---|
| `decode` | `latent2rgb (fast)`, `tiny vae (taeh3)` or `vae (quality)` — see the table above. |
| `tiny_vae` | Which `models/vae_approx` checkpoint to use for `decode='tiny vae (taeh3)'`. `none` by default. |
| `preview_target` | `node` shows it on this node — always available. `sampler (VHS)` puts it in the sampler's usual preview slot and needs [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) installed; `both` does both. |
| `preview_frames` | Cap on **latent** frames used. Thinned evenly for `latent2rgb`/`vae`; taken as a chronological prefix for `tiny vae (taeh3)` (see above). The main cost knob. |
| `preview_fps` | The shot's frame rate. A FLOAT, so the Director's `fps` output wires straight in. |
| `playback` | `true speed` (default) spreads the sampled frames across the shot's real length, so the preview lasts as long as the finished clip. `source fps` plays them at `preview_fps` flat, like ComfyUI's own preview. |
| `max_resolution` | Long edge of the preview image, as a **target**. |
| `webp_quality` | Quality of the animation sent to the browser. |
| `every_n_steps` | Never preview more often than every N sampler steps. |
| `max_preview_overhead` | Share of render time previews may use, in percent (default 25). After a preview costing C seconds the next waits `C·(100/P − 1)` s. 0 disables. |
| `suppress_default_preview` | Hides ComfyUI's built-in single-frame preview. |
| `show_vae_input` | Shows or hides the `vae` socket on the node face. On by default; turn it off to declutter when you're only using `latent2rgb` or `tiny vae (taeh3)`. Turning it off **disconnects** any wired VAE (sockets don't have a "hidden but still linked" state), so switch it back on and rewire before using `decode='vae (quality)'`. |

`vae` itself isn't a Settings-popup row — it's a socket (`minimax_h3_video_vae`), shown or
hidden on the node face by `show_vae_input` above. Only needed for `decode='vae (quality)'`.

## Interactive graphs

Below the live preview image, two small graphs sit **side by side** (ported from
KJNodes' Preview Override, whose own panel lays them out the same way) and update
alongside every rendered preview:

* **σ / Δ** — the sampler's noise schedule (σ, dotted grey line, drawn in full up front)
  overlaid with how much the shot is still changing (Δ, orange fill — the average
  per-element magnitude of change in the previewed latent since the last rendered
  preview). A flattening Δ against a still-high σ is often a sign a shot has converged
  early; a Δ that's still spiking near the end of the schedule is worth a longer look.
* **step time** — wall-clock time between rendered previews, with a click-to-toggle
  ms/s label showing the average and an ETA for the rest of the run.

Drag the grip directly above the graphs to resize them **vertically** — the image area
above shrinks or grows to compensate, and the chosen height is remembered on the node
(same mechanism as KJNodes' own panel-height grip), so it survives saving and reloading
the workflow.

Both graphs are interactive:

| Action | Effect |
|---|---|
| Hover the σ/Δ graph | Scrub to any step; the header shows that step's exact σ/Δ values. |
| Click the σ/Δ graph | Lock the hovered step so it survives moving the mouse away; click again (or click elsewhere on the graph) to unlock. |
| ← / → while hovering the graphs | Step the locked position one step at a time. |
| Click the step-time graph (or its label/value) | Toggle the time readout between ms and s. |
| Drag the grip above the graphs | Resize the graphs panel vertically; persists across saves. |

The graphs update on the same cadence the preview image does — `every_n_steps` and
`max_preview_overhead` throttle both together, so a heavily throttled run's graphs show
gaps between rendered steps rather than every single sampler step. At the default
`every_n_steps=1` with no throttling, they're as fine-grained as the sampler itself.

## Getting a taeh3 checkpoint

`tiny vae (taeh3)` needs a **taehv-format** decoder checkpoint whose latent shape
matches MiniMax H3: 24 latent channels, patch size 2. This is the same
[madebyollin/taehv](https://github.com/madebyollin/taehv) family of tiny temporal
decoders that ComfyUI core and KJNodes already use for LTX previews, trained instead for
H3's latent. Drop the checkpoint file into `ComfyUI/models/vae_approx/` and select it
from the `tiny_vae` widget — it'll show up alongside any other `vae_approx` files you
already have. A checkpoint whose channel count doesn't match H3's is rejected at run
time (a warning is logged) and the node falls back to `latent2rgb` for that run.

## Attribution

This node is a merge of two GPL-3.0 projects:

* **H3 packed-latent unpacking, preview scheduling, playback-rate math and the node's
  JS panel** — from `minimax_preview.py` / `js/minimax_preview.js` in
  [ComfyUI-MiniMaxH3-Director](https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director).
* **The taeh3/tiny-VAE decoder loader** (`tiny_vae.py`) — ported near-verbatim from
  `nodes/tiny_vae.py` in [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes).
* **The interactive σ/Δ and step-time graphs** (the canvas drawing, hover/lock/arrow-key
  scrubbing, side-by-side layout, and the panel-height resize grip, all in
  `js/minimax_h3_preview_override.js`) — ported from
  `web/js/preview_override/preview_override.js` in
  [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), with the SamplerDetailBoost
  curve overlay and per-step image-frame cache/scrub left out (out of scope for this node;
  it keeps a single always-current preview image instead).

Both are licensed GNU GPLv3; this repository is too — see [LICENSE](LICENSE).
