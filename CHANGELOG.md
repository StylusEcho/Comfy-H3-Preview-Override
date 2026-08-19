# Changelog

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
