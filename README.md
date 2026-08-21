# Comfy-H3-Preview-Override

**H3 Preview Override** — [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)'
Preview Override node, ported to MiniMax H3.

Wire it between your model and the sampler and watch the *whole shot* denoise on the node
itself: a scrubbable per-step frame history, live σ/Δ and step-time graphs, three decode
modes, and playback that runs at the clip's real speed. Every setting lives behind a
single **⚙** button in the panel header, so the node body stays down to just the preview
image and the graph panel.

## Why this exists

ComfyUI core ships `latent_rgb_factors` for MiniMax H3, so a preview isn't *missing* —
but core's own previewer renders `x0[0, :, 0]`, the first latent frame only, so you watch
one frozen still while a five-second shot samples.

KJNodes' **Model Preview Override** solves the "show the whole clip" problem properly,
including a taeh3/taehv tiny-VAE decode path — but its video paths are gated on
LTX-specific checks (`_is_ltx_latent_format`, `_is_ltx2_diffusion_model`), and nothing in
it unpacks H3's packed audio+video latent. On MiniMax it falls through to the same single
frame.

This is that node with the H3 gaps filled in. Kijai's structure and behaviour are kept
wholesale; see [Attribution](#attribution) for exactly what changed.

## Decode modes

| `decode` | Cost | Quality | Notes |
|---|---|---|---|
| `latent2rgb (fast)` | ~free (one matmul) | rough colours | Default. Always available, no extra files. |
| `tiny vae (taeh3)` | small, roughly latent2rgb speed | near-VAE colours | Needs a taehv-format checkpoint sized for H3 (24 latent channels, patch size 2 — commonly named `taeh3`) in `models/vae_approx`, picked with `tiny_vae`. See [Getting a taeh3 checkpoint](#getting-a-taeh3-checkpoint). |
| `vae (quality)` | real decode cost, tens of seconds per preview at high res | true colours | Needs `minimax_h3_video_vae` wired into `vae`. |

**All three modes span the whole shot.** `preview_frames` is how many frames to show,
spread evenly from the first frame of the clip to the last — never a truncated opening
section.

For `latent2rgb` and `vae` the latent is thinned *before* decoding, so `preview_frames` is
also the main cost knob. `tiny vae (taeh3)` can't work that way: the decoder chains state
frame to frame (memory blocks), so a mid-clip frame can't be produced without decoding
everything before it. Kijai's node decodes a chronological prefix there to bound per-step
cost; this one decodes the full clip and subsamples the output instead, because a prefix
shows the opening fraction of the shot and calls it the preview. So for that mode
`preview_frames` caps transfer size, not decode time — use `every_n_steps` and
`max_preview_overhead` as the cost knobs.

## True speed

There is no frame-rate widget. The playback rate is derived from the shot's real duration,
ported from
[ComfyUI-MiniMaxH3-Director](https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director)'s
preview node:

* **`true speed`** (default) — the previewed frames are spread across the clip's real
  length, so the preview lasts exactly as long as the finished video. H3 compresses time
  ~3.35× in the latent (17 pixel frames per 5 latent tokens), so with `latent2rgb` — one
  image per latent frame — this caps at about 24 / 3.35 ≈ 7.2 fps. That is not a setting
  being ignored; it's what honest timing looks like at that frame budget. Raise
  `preview_frames`, or use `tiny vae (taeh3)`, to push it up.
* **`source fps`** — plays at H3's native 24 flat, like ComfyUI's own preview and most
  other packs: motion reads at normal speed, but the clip is over in a third of the time.

Judge timing with the first, movement with the second.

## The panel

| Action | Effect |
|---|---|
| Hover either graph | Scrub to any step; both headers show that step's values, and the preview jumps to that step's cached frames. |
| Click the σ/Δ graph | Lock the hovered step so it survives moving the mouse away; click again to unlock. |
| ← / → over the panel | Step the locked position one step at a time. |
| Click the step-time graph | Toggle the readout between ms and s. |
| Click the preview image | Pause/resume playback (animated content only). |
| Space over the panel | Same as clicking the image. |
| Drag the scrub bar | Seek within the clip. |
| Drag the grip above the panel | Resize the graph panel; persists across saves. |
| **▾ / ▸** in the panel header | Collapse the graphs to just the header, handing the space back to the preview. Persists across saves. |
| **⚙** in the panel header | Open the Settings popup — see below. |

The graphs:

* **σ / Δ** — the sampler's noise schedule (σ, dotted grey, drawn in full up front)
  overlaid with how much the shot is still changing (Δ, orange — the average per-element
  change in the previewed latent since the last rendered preview). A flattening Δ against
  a still-high σ often means the shot converged early.
* **step time** — wall-clock time between rendered previews, with the average and an ETA
  in the panel header.

Scrubbing works off whatever data has arrived; it doesn't depend on the one-shot σ
schedule message, and a step with no known σ shows `—` rather than disabling the readout.

## Settings

Click **⚙** in the panel header to open a popup with every widget below — the node body
itself only ever shows the preview image and the graph panel. The popup edits the same
underlying widget values ComfyUI always serialised (nothing about how a workflow saves or
loads changed), so old saved workflows load their settings exactly as before; you just no
longer scroll a stack of rows above the preview to see or change them. Hover any row (or
its control) for a mouseover caption explaining that setting — the same text is also
shown underneath the row.

| Setting | What it does |
|---|---|
| `decode` | `latent2rgb (fast)`, `tiny vae (taeh3)` or `vae (quality)` — see above. |
| `playback` | `true speed` (default) or `source fps` — see above. |
| `preview_target` | `node` shows it here — always available. `sampler (VHS)` puts it in the sampler's usual preview slot and needs [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite); `both` does both. |
| `preview_frames` | How many frames to show, spread evenly across the whole clip. |
| `max_resolution` | Max preview side in pixels. `0` = full decoded resolution, no downscale. |
| `jpeg_quality` | Quality for the preview transport (JPEG for a single frame, WebP for an animation). Ignored when NVENC MP4 encoding is used. |
| `every_n_steps` | Never preview more often than every N sampler steps. |
| `max_preview_overhead` | Share of render time previews may use, in percent (default 25). After a preview costing C seconds the next waits `C·(100/P − 1)` s. `0` disables. |
| `suppress_default_preview` | Suppress ComfyUI's built-in single-frame preview while this runs. |
| `tiny_vae` | Which `models/vae_approx` checkpoint to use for `decode='tiny vae (taeh3)'`. |
| `show_vae_input` | Shows or hides the `vae` socket on the node face. On by default; turn it off to declutter when you're only using `latent2rgb` or `tiny vae (taeh3)`. Turning it off **disconnects** any wired VAE (a socket has no "hidden but still linked" state), so switch it back on and rewire before using `decode='vae (quality)'`. |

`vae` itself isn't a Settings-popup row — it's a socket (`minimax_h3_video_vae`), shown or
hidden by `show_vae_input` above. Only needed for `decode='vae (quality)'`.

**Get H3 Preview Frames** is a second node that returns everything captured during the
last run as an IMAGE batch — one frame per rendered preview, so it reads as a timelapse of
the shot converging. Wire `model` from H3 Preview Override and `after_sample` from
anything after the sampler to force the execution order.

## Requirements

* **ComfyUI ≥ 0.30.0** — H3 support, `comfy_api.latest` and the packed AV latent all
  landed in 0.30.
* For `decode='tiny vae (taeh3)'`: a core build shipping `comfy.taesd.taehv` (the same
  releases that added TAEHV/LTX2 video previews). If it's missing the node logs a warning
  and falls back to `latent2rgb` — it does not error out.
* **Python 3.10+** (ComfyUI's own environment).
* **No extra pip packages.** PyAV is optional: if it's present *and* built with NVENC,
  previews are encoded as MP4 (roughly 8× faster and 5× smaller); otherwise animated WebP
  is used, which is always available through Pillow.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StylusEcho/Comfy-H3-Preview-Override.git
```

Then restart ComfyUI and reload the browser tab.

This registers `H3PreviewOverride` and `GetH3PreviewFrames`, which don't collide with
either KJNodes' `ModelPreviewOverrideKJ` or ComfyUI-MiniMaxH3-Director's
`MiniMaxH3PreviewOverrideCS`, so all three packs can be installed together.

## Getting a taeh3 checkpoint

`tiny vae (taeh3)` needs a **taehv-format** decoder checkpoint whose latent shape matches
MiniMax H3: 24 latent channels, patch size 2. This is the same
[madebyollin/taehv](https://github.com/madebyollin/taehv) family of tiny temporal decoders
ComfyUI core and KJNodes already use for LTX previews, trained for H3's latent instead.
Drop it in `ComfyUI/models/vae_approx/` and pick it from `tiny_vae`. A checkpoint whose
channel count doesn't match is rejected at run time (with a warning) and the node falls
back to `latent2rgb` for that run.

> ComfyUI core has since gained native `taeh3` support
> ([Comfy-Org/ComfyUI#15695](https://github.com/Comfy-Org/ComfyUI/pull/15695)), which
> registers `taeh3` in `VAELoader` and sizes `TAEHV` for H3 directly. On a core build new
> enough to include it you can load a taeh3 checkpoint with the stock **VAELoader** and
> wire it into the `vae` socket instead. The bundled `tiny_vae.py` loader is kept for
> older builds, where core cannot size that decoder at all.

## Attribution

This pack is a port of two GPL-3.0 projects:

* **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** — `h3_preview_override.py`
  is a port of `nodes/preview_override_node.py`, `js/h3_preview_override.js` of
  `web/js/preview_override/preview_override.js`, `js/h3_preview_override.css` of
  `web/js/preview_override/preview_override.css`, and `tiny_vae.py` of `nodes/tiny_vae.py`.
  Kijai's structure is kept: the OUTER_SAMPLE wrapper, off-thread encoder, NVENC MP4 with
  WebP fallback, the per-step frame cache and playback scrub bar, and the σ/Δ + step-time
  graphs.
* **[ComfyUI-MiniMaxH3-Director](https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director)**
  — the H3 packed-latent unpacking, the latent-to-pixel frame-count mapping
  (`pixel_frames_from_latent_t`), the `true speed` playback derivation, and the
  preview-overhead throttle.

Changes from Kijai's original, beyond the H3 adaptation:

* **Packed audio+video latent.** `CFGGuider.sample` packs the video and audio streams into
  one flat tensor and only then wraps the callback with the nested view — and that wrapper
  sits *behind* an OUTER_SAMPLE wrapper, so what reaches the callback is the flat pack.
  `_video_stream` unpacks it with core's own `unpack_latents`.
* **LTX gating removed**, along with the `WrappedPreviewer` routing it fed.
* **Frames span the whole clip** in every decode mode (see [Decode modes](#decode-modes)).
* **No frame-rate widget** — rate is derived server-side (see [True speed](#true-speed)).
* **Collapsible graph panel.**
* **A Settings popup** replaces the stack of widget rows Kijai's node draws above the
  preview — see [Settings](#settings). Adds one new widget, `show_vae_input`, to toggle
  the `vae` socket's visibility.
* **The SamplerDetailBoost curve overlay is not ported** — it reads `extra_options` only
  KJNodes' own sampler node sets, and this pack doesn't ship one.
* **Pointer input uses document-level capture-phase listeners with rect hit-testing**
  rather than listeners on the elements. A DOM widget's own mouse events are not reliably
  delivered across ComfyUI frontend versions and zoom levels; when they aren't, hover
  scrubbing silently does nothing.

Both upstreams are GNU GPLv3; so is this — see [LICENSE](LICENSE).
