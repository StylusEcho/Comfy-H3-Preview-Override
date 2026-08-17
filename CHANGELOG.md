# Changelog

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
