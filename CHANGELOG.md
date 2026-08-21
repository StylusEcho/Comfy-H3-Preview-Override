# Changelog

## 0.5.1

- All of the node's widgets (`decode`, `playback`, `preview_target`, `preview_frames`,
  `max_resolution`, `jpeg_quality`, `every_n_steps`, `max_preview_overhead`,
  `suppress_default_preview`, `tiny_vae`) now live in a popup opened by a **⚙** button in
  the panel header, instead of each drawing its own row above the preview. Purely a
  display change — the same widget objects back the popup's controls, so saved workflows
  load their values exactly as before.
- Added a **`show_vae_input`** setting that shows or hides the `vae` socket on the node
  face — a declutter toggle for when you're only using `latent2rgb` or `tiny vae
  (taeh3)`. Turning it off disconnects any wired VAE, since a socket has no "hidden but
  still linked" state.

## 0.5.0

**Rewritten as a port of KJNodes' Preview Override.** The previous node
(`MiniMaxH3PreviewPlusCS`, built up from the Director's preview node) is removed and
replaced by `H3PreviewOverride`, ported from `nodes/preview_override_node.py` and
`web/js/preview_override/preview_override.js` in ComfyUI-KJNodes with the H3 gaps filled
in. Workflows using the old node need it deleted and re-added.

What the port brings over the old node: the per-step frame cache with hover/lock
scrubbing (the preview jumps to whatever step you point at, not just the latest), the
playback scrub bar with click-to-pause, NVENC MP4 encoding with animated-WebP fallback,
and the off-thread encoder so the sampler never blocks on a preview. A second node,
**Get H3 Preview Frames**, returns the whole run's captured frames as an IMAGE batch.

Plus the four requested changes:

- **Pack renamed to `Comfy-H3-Preview-Override`** (`comfy-h3-preview-override`), matching
  the repository.
- **The graph panel collapses** to just its header via the ▾/▸ button, handing the space
  back to the preview image. Persists on the node alongside the panel height.
- **`preview_frames` spreads frames across the whole video** in every decode mode — first
  frame to last, never a truncated opening section. For `tiny vae (taeh3)` that means the
  whole clip is decoded and the output subsampled, where Kijai's node decodes a prefix to
  bound cost; `preview_frames` there caps transfer size rather than decode time.
- **The frame-rate widget is gone.** Playback rate is derived server-side from the shot's
  real duration — the Director's `true speed`, now the default — with `source fps` (H3's
  native 24 flat) as the alternative.

Not ported: the SamplerDetailBoost curve overlay, which reads `extra_options` only
KJNodes' own sampler sets.

## 0.4.1

Bug fixes to the taeh3 preview span and to graph scrubbing, plus a graphs hide button.

- **`decode='tiny vae (taeh3)'` previewed only the opening of the shot.** With the
  default `preview_frames=24` against a ~37-frame latent it decoded `range(24)` — a
  chronological prefix, i.e. the first ~65% of the clip — inheriting KJNodes'
  cost-bounding behaviour. It compounded: `pixel_frames` is measured off the *full*
  latent, so those prefix-only frames were then spread across the whole clip's duration,
  making the preview both truncated and too slow. It now decodes the whole clip and
  evenly subsamples the output, so the preview spans the finished shot like the other two
  modes. `preview_frames` no longer bounds decode time for this mode (only transfer
  size); use `every_n_steps` / `max_preview_overhead` instead.
- **Hovering the graphs did nothing.** Scrubbing was gated on `cachedSigmas`, which is
  only ever set by the single up-front σ-schedule message — miss it and hover was dead
  for the whole run. It also hit-tested against a different x-axis than the draw code
  derived internally, so the cursor line could land away from the pointer. The axis is
  now computed once and passed to both, scrubbing is bound to *both* canvases sharing one
  cursor, and it no longer depends on the σ message (an unknown σ renders as `—`).
- Added a **graphs ▾** button on the status line that hides/shows the graphs; the state
  persists on the node like the panel height does.

## 0.4.0

**Breaking:** renamed the node and pack from "MiniMax H3 Preview Override" to
**"MiniMax H3 Preview Plus"** — node ID changed `MiniMaxH3PreviewOverrideCS` →
`MiniMaxH3PreviewPlusCS`, and the pack's registry name changed to
`minimaxh3-preview-plus`. Workflows built against the old node ID need that node
deleted and re-added under its new name; widget values aren't carried over
automatically. The upside: this node no longer shares an ID with
ComfyUI-MiniMaxH3-Director's own Preview Override node, so the two packages can now be
installed side by side without a duplicate-node-name conflict.

Other changes:
- The σ/Δ and step-time graphs now sit **side by side** instead of stacked, and a drag
  grip above them resizes the graphs panel vertically (the image area shrinks/grows to
  compensate) — same interaction as KJNodes' own Preview Override panel grip. The chosen
  height persists on the node (`node.properties`), surviving save/reload.
- Added a **`show_vae_input`** setting that shows or hides the `vae` socket on the node
  face — a declutter toggle for when you're only using `latent2rgb` or `tiny vae
  (taeh3)`. Turning it off disconnects any wired VAE, since a socket has no "hidden but
  still linked" state.
- Settings-popup rows now show a native mouseover tooltip (in addition to the existing
  always-visible description line) explaining each setting.

## 0.3.0

Added the interactive σ/Δ and step-time graphs from KJNodes' Preview Override node,
under the live preview image. The Python side now sends the sampler's σ schedule once
up front and, on every rendered preview, a Δ (change-magnitude) reading and step timing
alongside the image; the JS side draws both as scrubbable canvases — hover to inspect
any step, click to lock it, arrow keys to step while locked, click the step-time graph
to toggle ms/s. The SamplerDetailBoost curve overlay and KJNodes' per-step image-frame
cache/scrub were left out as out of scope; this node still shows one always-current
preview image.

## 0.2.0

All of the node's widgets (`decode`, `tiny_vae`, `preview_target`, `preview_frames`,
`preview_fps`, `playback`, `max_resolution`, `webp_quality`, `every_n_steps`,
`max_preview_overhead`, `suppress_default_preview`) now live in a popup opened by a
single `⚙ Settings` button on the node, instead of each drawing its own row. The node
body shows only the button and the live preview panel. Purely a display change — the
same widget objects back the popup's controls, so saved workflows load their values
exactly as before.

## 0.1.0

Initial release. Combines ComfyUI-MiniMaxH3-Director's `MiniMax H3 Preview Override`
node (H3 packed-latent unpacking, in-node live preview) with ComfyUI-KJNodes' taeh3/
tiny-VAE decode path, adding a `decode='tiny vae (taeh3)'` mode alongside the existing
`latent2rgb (fast)` and `vae (quality)` modes, plus a `tiny_vae` widget to pick the
`models/vae_approx` checkpoint it uses.
