# Comfy-H3-Preview-Override

A single ComfyUI custom node: **MiniMax H3 Preview Override**, combining

* the H3-aware live sampling preview from
  [ComfyUI-MiniMaxH3-Director](https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director)
  (unpacks MiniMax H3's packed audio+video latent so the preview shows the whole shot,
  not just the first latent frame), with
* the **taeh3** tiny-VAE decode path from
  [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)' Preview Override node
  (a small trained temporal decoder for near-VAE colour accuracy at close to
  latent2rgb speed).

Wire it between your model and the sampler and watch the shot denoise on the node
itself, with a choice of three decode modes. All of those decode/timing/output settings
live behind a single **⚙ Settings** button on the node rather than as a stack of widget
rows, so the node itself stays small — just the button and the live preview.

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

This node does both: H3's packed-latent unpacking, plus all three decode modes.

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

This node registers under the **same node ID and display name**
(`MiniMaxH3PreviewOverrideCS` / "MiniMax H3 Preview Override") as the Preview Override
node shipped in ComfyUI-MiniMaxH3-Director, since it's meant as a drop-in replacement for
it — existing workflows keep working, they just gain the `tiny_vae` decode option.

If you install **both** packages, ComfyUI will load whichever one registers the class
last and print a duplicate-node-name warning for the other; only one will actually run.
Either disable one of the two packages in ComfyUI Manager, or (if you rely on the rest of
the Director suite — the Director, Retake Stitch, Enhance Prompt and Save Last Frame
nodes) delete `minimax_preview.py` and its `js/minimax_preview.js` counterpart from your
ComfyUI-MiniMaxH3-Director checkout so only this package's version of the Preview
Override node is registered.

## Settings

Click **⚙ Settings** on the node to open a popup with every widget below — the node body
itself only ever shows the live preview panel and the button. The popup edits the same
underlying widget values ComfyUI always serialised (nothing about how a workflow saves
or loads changed), so old saved workflows load their settings exactly as before; you
just no longer scroll a tall stack of rows to see or change them.

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
| `vae` | `minimax_h3_video_vae`. Only needed for `decode='vae (quality)'`. |

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

Both are licensed GNU GPLv3; this repository is too — see [LICENSE](LICENSE).
