"""Live sampling preview for MiniMax H3, ported from KJNodes' Preview Override.

Provenance
----------
This is a port of `nodes/preview_override_node.py` from ComfyUI-KJNodes
(github.com/kijai/ComfyUI-KJNodes, GPL-3.0), adapted for MiniMax H3. Kijai's structure is
kept: the OUTER_SAMPLE wrapper, the off-thread encoder, NVENC MP4 with a WebP fallback,
the per-step frame cache the JS side scrubs through, and the σ/Δ + step-time graph feed.

What changed for H3
-------------------
1. **Packed audio+video latent.** `CFGGuider.sample` packs the video and audio streams
   into ONE flat tensor and only then wraps the callback with the nested view. That
   wrapper sits *behind* an OUTER_SAMPLE wrapper in the call chain, so what reaches this
   callback is the flat pack, not the NestedTensor. `_video_stream` unpacks it with core's
   own `unpack_latents`. KJNodes' `_normalize_packed_x0` does not handle this shape, which
   is why its node falls through to a single frame on MiniMax.

2. **LTX gating removed.** KJNodes' video paths are gated on `_is_ltx_latent_format` /
   `_is_ltx2_diffusion_model` and route through the LTX node's `WrappedPreviewer`. None of
   that applies here, so the decode paths are: latent2rgb, tiny VAE (taeh3), full VAE.

3. **Frames span the whole clip.** Every decode path spreads `preview_frames` evenly
   across the entire shot. Kijai's tiny-VAE path decodes a chronological *prefix* to bound
   per-step cost — see `_tiny_vae_decode_to_pil` for why that cannot be kept here.

4. **True speed instead of a frame-rate widget.** There is no `preview_fps`; playback rate
   is derived server-side from H3's real output length, ported from ComfyUI-MiniMaxH3-
   Director's preview node. See `_playback_rate`.

The SamplerDetailBoost curve overlay is not ported — it reads `extra_options` that only
KJNodes' own sampler node sets, and this pack does not ship one.
"""

import base64
import io as pyio
import logging
import queue
import struct
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import folder_paths
import latent_preview
from comfy_api.latest import io

from .tiny_vae import load_tiny_vae_decoder

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

try:
    from protocol import BinaryEventTypes
except ImportError:
    BinaryEventTypes = None

log = logging.getLogger(__name__)

EVENT = "h3_preview_override"
WRAPPER_KEY = "h3_preview_override"

DECODE_FAST = "latent2rgb (fast)"
DECODE_TAEH3 = "tiny vae (taeh3)"
DECODE_VAE = "vae (quality)"

PLAYBACK_TRUE = "true speed"
PLAYBACK_SOURCE = "source fps"

TARGET_NODE = "node"
TARGET_SAMPLER = "sampler (VHS)"
TARGET_BOTH = "both"

MODEL_FPS = 24.0            # H3's native output rate


# --------------------------------------------------------------------------- H3 latent maths

def _video_stream(x0, latent_shapes=None):
    """Pull the [B, C, T, h, w] video latent out of whatever the sampler handed us."""
    if x0 is None:
        return None
    if getattr(x0, "is_nested", False):
        return x0.tensors[0]
    if x0.ndim == 5:
        return x0
    if latent_shapes and len(latent_shapes) > 1:
        try:
            return comfy.utils.unpack_latents(x0, list(latent_shapes))[0]
        except Exception as e:
            log.warning("[H3PreviewOverride] could not unpack the packed latent: %s", e)
            return None
    return None


def pixel_frames_from_latent_t(latent_t):
    """How many output frames one H3 video latent covers.

    The latent is compressed ~3.35x in time (core's `video_latent_t`: 17k+5 pixel frames
    become 5k+2 latent frames), so a preview that plays latent frames at the video's fps
    runs more than three times too fast. Inverting that mapping is what lets the preview
    be honest about the shot's real duration.
    """
    if latent_t <= 2:
        return 5
    k, remainder = divmod(int(latent_t) - 2, 5)
    if remainder == 0:
        return 17 * k + 5
    return max(1, int(round(latent_t * 17.0 / 5.0)))   # off-grid latent: approximate


def _playback_rate(playback, shown, pixel_frames):
    """Frames per second to play `shown` images at. Ported from the Director's node.

    Two honest answers, and this picks between them rather than taking a frame rate:

    `true speed` spreads the images across the shot's real duration, so the preview lasts
    exactly as long as the finished clip. With latent2rgb that caps out at
    MODEL_FPS / 3.35 — one image per latent frame, and H3 compresses time by that much.

    `source fps` plays them at H3's native rate flat. That is what ComfyUI's own preview
    and the other packs do: motion reads at normal speed, but the clip is over in a third
    of the time. Useful for judging movement, misleading about timing.
    """
    if playback == PLAYBACK_SOURCE:
        return MODEL_FPS
    return max(0.1, MODEL_FPS * max(1, shown) / max(1, pixel_frames))


def _even_indices(total, count):
    """`count` indices spread evenly across [0, total), first and last included."""
    if count is None or count <= 0 or count >= total:
        return list(range(total))
    return sorted(set(np.linspace(0, total - 1, count).round().astype(int).tolist()))


def throttle_gap(cost_seconds, max_overhead_percent):
    """How long to wait after a preview that took `cost_seconds`.

    A full VAE decode of a 1344x768 shot can cost tens of seconds — once per step that is
    minutes of pure overhead. Rather than guess a step interval, hold previews to a share
    of wall-clock: to spend at most P percent of the time previewing, a render costing C
    must be followed by C*(100/P - 1) seconds of actual sampling.
    """
    if max_overhead_percent <= 0 or cost_seconds <= 0:
        return 0.0
    return cost_seconds * (100.0 / float(max_overhead_percent) - 1.0)


# ------------------------------------------------------------------------- encoder plumbing

def _suppressed_preview_image(self_, preview_format, x0):
    return None


class _AsyncPreviewEncoder:
    """Off-thread encoder. Bounded FIFO drops-on-full so the sampler never blocks on us."""

    _STOP = object()

    def __init__(self, max_in_flight=2):
        self.q = queue.Queue(maxsize=max_in_flight)
        self.thread = threading.Thread(target=self._run, name="h3_preview_encoder", daemon=True)
        self.thread.start()

    def submit(self, fn):
        try:
            self.q.put_nowait(fn)
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is self._STOP:
                return
            try:
                item()
            except Exception:
                logging.exception("[H3PreviewOverride] async encoder error")

    def shutdown(self, drain_timeout=5.0):
        try:
            self.q.put(self._STOP, timeout=drain_timeout)
        except queue.Full:
            pass
        self.thread.join(timeout=drain_timeout)


# PyPI PyAV wheels typically lack NVENC; probe once at import.
def _probe_nvenc():
    try:
        import av  # noqa
        av.Codec("h264_nvenc", "w")
        return True
    except Exception:
        return False


# --disable-api-nodes installs a CSP with no media-src
def _csp_blocks_video():
    try:
        from comfy.cli_args import args as _args
        return bool(getattr(_args, "disable_api_nodes", False))
    except Exception:
        return False


_HAS_NVENC = _probe_nvenc()
_NVENC_AVAILABLE = _HAS_NVENC and not _csp_blocks_video()
if _HAS_NVENC and not _NVENC_AVAILABLE:
    logging.info("[H3PreviewOverride] --disable-api-nodes blocks blob: video, using WebP.")

# NVENC H.264 rejects sub-145x49 inputs at avcodec_open2 — fall back to WebP for small frames.
_NVENC_MIN_W = 145
_NVENC_MIN_H = 49

_nvenc_warned = False


def _encode_mp4_nvenc(frames, fps, max_res):
    # Fragmented MP4 so the browser can decode mid-download. Returns (None, 0, 0) on
    # failure (including too-small-for-NVENC), so caller falls through to WebP.
    global _nvenc_warned
    if not frames:
        return None, 0, 0
    try:
        import av
    except Exception:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = f if f.mode == "RGB" else f.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    # yuv420p requires even dimensions.
    w0, h0 = pil_frames[0].width, pil_frames[0].height
    out_w, out_h = w0 & ~1, h0 & ~1
    if (out_w, out_h) != (w0, h0):
        pil_frames = [pf.resize((out_w, out_h), Image.LANCZOS) for pf in pil_frames]
    if out_w < _NVENC_MIN_W or out_h < _NVENC_MIN_H:
        return None, 0, 0
    # Driver/GPU varies what option combos are accepted; bare preset always works.
    option_candidates = [
        {"preset": "p1", "rc": "vbr", "cq": "23"},
        {"preset": "p1"},
    ]
    last_err = None
    for opts in option_candidates:
        buf = pyio.BytesIO()
        try:
            container = av.open(
                buf, mode="w", format="mp4",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            stream = container.add_stream("h264_nvenc", rate=int(max(1, round(fps))))
            stream.width = out_w
            stream.height = out_h
            stream.pix_fmt = "yuv420p"
            stream.options = opts
            for pf in pil_frames:
                for pkt in stream.encode(av.VideoFrame.from_image(pf)):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            return base64.b64encode(buf.getvalue()).decode("ascii"), out_w, out_h
        except Exception as e:
            last_err = e
            continue
    if not _nvenc_warned:
        _nvenc_warned = True
        logging.warning("[H3PreviewOverride] NVENC MP4 encode failed, using WebP: %s", last_err)
    return None, 0, 0


def _encode_animated_webp(frames, fps, quality, max_res):
    if not frames:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = f if f.mode == "RGB" else f.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)
    duration_ms = max(1, int(round(1000 / max(0.1, fps))))
    buf = pyio.BytesIO()
    try:
        pil_frames[0].save(
            buf, format="WEBP", save_all=True, append_images=pil_frames[1:],
            duration=duration_ms, loop=0, quality=quality, method=4,
        )
    except Exception as e:
        logging.warning("[H3PreviewOverride] animated WebP encode failed: %s", e)
        return None, 0, 0
    return base64.b64encode(buf.getvalue()).decode("ascii"), pil_frames[0].width, pil_frames[0].height


# ----------------------------------------------------------------------------- decode paths

def _get_core_previewer(load_device, latent_format):
    # Walk past custom-node hooks on get_previewer to reach the unwrapped core function.
    fn = latent_preview.get_previewer
    seen = set()
    while hasattr(fn, "__wrapped__") and id(fn) not in seen:
        seen.add(id(fn))
        fn = fn.__wrapped__
    return fn(load_device, latent_format)


def _decode_video_frames_l2rgb(x0, latent_format, max_frames):
    """latent2rgb over a [B, C, T, h, w] latent, thinned evenly across the whole clip."""
    # Bulk-blocking GPU->CPU copy (not per-frame non_blocking) avoids torn frames at high res.
    if x0.ndim != 5:
        return []
    rgb_factors = getattr(latent_format, "latent_rgb_factors", None)
    if rgb_factors is None:
        return []
    try:
        reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
        if reshape is not None:
            x0 = reshape(x0)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        factors = torch.tensor(rgb_factors, device=x0.device, dtype=x0.dtype).transpose(0, 1)
        bias_t = torch.tensor(bias, device=x0.device, dtype=x0.dtype) if bias is not None else None
        x = x0[0]
        indices = _even_indices(x.shape[1], max_frames)
        x = x[:, indices]
        x = x.movedim(0, -1)
        rgb = F.linear(x, factors, bias=bias_t)
        rgb.add_(1.0).mul_(127.5).clamp_(0, 255)
        rgb_cpu = rgb.to(torch.uint8).cpu().numpy()
        return [Image.fromarray(rgb_cpu[i]) for i in range(rgb_cpu.shape[0])]
    except Exception as e:
        logging.warning("[H3PreviewOverride] latent2rgb decode failed: %s", e)
        return []


def _full_vae_decode_to_pil(vae, x0, max_frames=None):
    """Full-quality decode. Thins the latent first — the real VAE is the expensive one."""
    # vae.decode handles device + tiling. Output shape varies by VAE; accept
    # (B, T, H, W, C) or (T, H, W, C) and normalise.
    if vae is None or x0.ndim != 5:
        return []
    indices = _even_indices(x0.shape[2], max_frames)
    x0 = x0[:, :, indices]
    try:
        images = vae.decode(x0)
    except Exception as e:
        logging.warning("[H3PreviewOverride] VAE decode failed: %s", e)
        return []
    if images.ndim == 5:
        images = images[0]
    if images.ndim != 4:
        return []
    u8 = (images.float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


def _tiny_vae_decode_to_pil(decoder, x0, max_frames=None):
    """Tiny VAE (taehv/taeh3). Decodes the WHOLE clip, then thins the output frames.

    Kijai's version decodes only the first `max_frames` latent frames, because the decoder
    chains state frame to frame (MemBlocks) and a mid-clip frame cannot be produced without
    everything before it — taking a prefix is the cheap way to bound per-step cost.

    That cannot be kept here. A prefix shows the opening fraction of the shot and calls it
    the preview, and it compounds: the playback rate is derived from the *full* latent's
    real duration, so those prefix-only frames get stretched across the whole clip and the
    preview reads as both truncated and too slow. So this pays for the full decode — taeh3
    is cheap enough that it is the right trade — and subsamples afterwards.

    Consequence worth knowing: for this mode `preview_frames` bounds encode/transfer size
    but not decode time. `every_n_steps` and `max_preview_overhead` are the cost knobs.

    Raises on failure so the caller can disable the decoder instead of retrying every step.
    """
    if x0.ndim == 4:
        rgb = decoder.decode(x0[:1])[0].movedim(0, -1).unsqueeze(0).contiguous()
    elif x0.ndim == 5:
        rgb = decoder.decode_video(x0[:1], frame_indices=None)
    else:
        return []
    if max_frames and 0 < max_frames < rgb.shape[0]:
        rgb = rgb[torch.as_tensor(_even_indices(rgb.shape[0], max_frames), device=rgb.device)]
    u8 = rgb.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


class _VHSStreamer:
    """Streams individual frames to VideoHelperSuite's animated latent-preview player."""

    def __init__(self, rate):
        self.rate = max(1, int(rate))
        self.first = True
        self.last_time = 0.0
        self.cursor = 0

    def send(self, pil_frames, rate=None):
        if PromptServer is None or BinaryEventTypes is None:
            return 0
        srv = PromptServer.instance
        total = len(pil_frames)
        if total == 0:
            return 0
        if rate and self.first:
            # locked in with the handshake — the player is told the rate exactly once
            self.rate = max(1, int(round(rate)))
        now = time.time()
        count = int((now - self.last_time) * self.rate)
        self.last_time += count / self.rate
        if count > total:
            count = total
        elif count <= 0:
            return 0
        if self.first:
            self.first = False
            srv.send_sync("VHS_latentpreview",
                          {"length": total, "rate": self.rate, "id": srv.last_node_id})
            self.last_time = now + 1.0 / self.rate

        order = [(self.cursor + i) % total for i in range(count)]
        node_id = (srv.last_node_id or "").encode("ascii")
        for i in order:
            message = pyio.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(i.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", node_id))
            pil_frames[i].save(message, format="JPEG", quality=95)
            srv.send_sync(BinaryEventTypes.PREVIEW_IMAGE, message.getvalue(), srv.client_id)
        self.cursor = (self.cursor + count) % total
        return count


# -------------------------------------------------------------------------------- wrapper

class _PreviewOverrideWrapper:
    def __init__(self, node_id, decode_mode, max_resolution, jpeg_quality, suppress_default,
                 preview_frames, playback, target, every_n_steps, max_overhead,
                 vae=None, tiny_vae="none"):
        self.node_id = str(node_id) if node_id is not None else None
        self.decode_mode = decode_mode
        self.max_resolution = int(max_resolution)
        self.jpeg_quality = int(jpeg_quality)
        self.suppress_default = bool(suppress_default)
        self.preview_frames = int(preview_frames)
        self.playback = playback or PLAYBACK_TRUE
        self.target = target
        self.every_n_steps = max(1, int(every_n_steps))
        self.max_overhead = max(0, min(100, int(max_overhead)))
        self.vae = vae
        self.tiny_vae = tiny_vae or "none"
        self.frames = []          # consumed by GetH3PreviewFrames

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, **kwargs):
        guider = executor.class_obj
        model_patcher = guider.model_patcher
        latent_shapes = kwargs.get("latent_shapes")
        latent_format = model_patcher.model.latent_format

        to_node = self.target in (TARGET_NODE, TARGET_BOTH)
        vhs = _VHSStreamer(MODEL_FPS) if self.target in (TARGET_SAMPLER, TARGET_BOTH) else None

        # Tiny VAE from models/vae_approx
        tiny_vae = None
        if self.decode_mode == DECODE_TAEH3 and self.tiny_vae != "none":
            tiny_vae = load_tiny_vae_decoder(self.tiny_vae)
            if tiny_vae is not None and latent_shapes and len(latent_shapes[0]) >= 2:
                channels = int(latent_shapes[0][1])
                if channels != tiny_vae.latent_channels:
                    logging.warning(
                        "[H3PreviewOverride] '%s' decodes %d-channel latents but this "
                        "model's are %d-channel; ignoring it.",
                        self.tiny_vae, tiny_vae.latent_channels, channels)
                    tiny_vae = None

        previewer = _get_core_previewer(model_patcher.load_device, latent_format)
        # Latent2RGB fallback — used when the active previewer returns a non-PIL result
        # (e.g. TAEHV/TAESD on a 5D latent), and for anything that isn't a video latent.
        fallback_previewer = None
        try:
            rgb_factors = getattr(latent_format, "latent_rgb_factors", None)
            if rgb_factors is not None:
                fallback_previewer = latent_preview.Latent2RGBPreviewer(
                    rgb_factors,
                    getattr(latent_format, "latent_rgb_factors_bias", None),
                    getattr(latent_format, "latent_rgb_factors_reshape", None),
                )
        except Exception:
            pass

        original_callback = callback
        node_id = self.node_id
        max_res = self.max_resolution
        quality = self.jpeg_quality
        self.frames = []

        # N+1 boundaries for N steps: keep them all so the step marker advances through each.
        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        total_steps_init = max(0, len(sigmas_list) - 1)

        # Pre-seed so step 1 has a measurable delta (the model's first move off noise).
        initial_seed_cpu = None
        try:
            if sigmas is not None and len(sigmas) > 0:
                # sigmas often lives on CPU while noise is on CUDA — align before the multiply.
                s0 = sigmas[0].to(noise.device) if hasattr(sigmas[0], "to") else sigmas[0]
                seeded = _video_stream(noise * s0, latent_shapes)
                if seeded is not None:
                    initial_seed_cpu = seeded.detach().float().cpu()
        except Exception as e:
            logging.warning("[H3PreviewOverride] initial delta pre-fill failed: %s", e)

        state = {"last_x0_cpu": initial_seed_cpu, "last_time": None, "step_ms_window": [],
                 "cost": 0.0, "finished": 0.0, "warned": False, "throttle_logged": False}

        # Boundary-0 message: sigmas (required by the JS hover handler) plus an optional
        # preview of the initial noise.
        if node_id is not None and PromptServer is not None:
            init_payload = {
                "node_id": node_id, "step": 0, "total": total_steps_init,
                "sigma": sigmas_list[0] if sigmas_list else None, "sigmas": sigmas_list,
            }
            try:
                init_latent = _video_stream(
                    noise * (sigmas[0].to(noise.device) if sigmas is not None and len(sigmas) > 0
                             else 1.0),
                    latent_shapes)
                pil_init = None
                if init_latent is not None and init_latent.ndim == 5:
                    frames = _decode_video_frames_l2rgb(init_latent, latent_format, 1)
                    pil_init = frames[0] if frames else None
                if pil_init is not None:
                    if pil_init.mode != "RGB":
                        pil_init = pil_init.convert("RGB")
                    if max_res and max_res > 0 and (pil_init.width > max_res or pil_init.height > max_res):
                        pil_init = ImageOps.contain(pil_init, (max_res, max_res), Image.LANCZOS)
                    ibuf = pyio.BytesIO()
                    pil_init.save(ibuf, format="JPEG", quality=quality)
                    init_payload["image"] = base64.b64encode(ibuf.getvalue()).decode("ascii")
                    init_payload["mime"] = "image/jpeg"
                    init_payload["w"] = pil_init.width
                    init_payload["h"] = pil_init.height
            except Exception as e:
                logging.warning("[H3PreviewOverride] initial noise preview failed "
                                "(sigmas still sent): %s", e)
            PromptServer.instance.send_sync(EVENT, init_payload, PromptServer.instance.client_id)

        encoder = _AsyncPreviewEncoder()

        def _should_skip(now):
            gap = throttle_gap(state["cost"], self.max_overhead)
            return gap > 0 and (now - state["finished"]) < gap

        def new_callback(step, x0, x, total_steps_):
            nonlocal tiny_vae
            do_preview = (x0 is not None and step % self.every_n_steps == 0
                          and not _should_skip(time.time()))
            if do_preview:
                t0 = time.time()
                try:
                    # NEVER rebind x0 — the sampler reuses the same tensor downstream
                    # (unpack_latents reshapes it). Preview work stays on x0_view.
                    x0_view = _video_stream(x0, latent_shapes)
                    if x0_view is None:
                        x0_view = x0

                    # Measured off the FULL latent, before any thinning: this is the shot's
                    # real length, and it is what makes 'true speed' mean anything.
                    pixel_frames = (pixel_frames_from_latent_t(int(x0_view.shape[2]))
                                    if x0_view.ndim == 5 else 1)

                    pil_frames = []
                    if tiny_vae is not None:
                        try:
                            pil_frames = _tiny_vae_decode_to_pil(tiny_vae, x0_view,
                                                                 self.preview_frames)
                        except Exception as e:
                            # OOM at 16x upscale is the likely cause — drop to the cheap
                            # paths for good rather than retrying every step.
                            logging.warning("[H3PreviewOverride] tiny VAE decode failed, "
                                            "falling back: %s", e)
                            tiny_vae = None
                    if not pil_frames and self.decode_mode == DECODE_VAE and self.vae is not None \
                            and x0_view.ndim == 5:
                        pil_frames = _full_vae_decode_to_pil(self.vae, x0_view, self.preview_frames)
                    if not pil_frames and x0_view.ndim == 5:
                        pil_frames = _decode_video_frames_l2rgb(x0_view, latent_format,
                                                                self.preview_frames)

                    if not pil_frames:
                        # Not a video latent (or nothing above could handle it) — fall back
                        # to whatever previewer core would have used.
                        for prev in (previewer, fallback_previewer):
                            if prev is None:
                                continue
                            try:
                                out = prev.decode_latent_to_preview(x0_view)
                            except Exception:
                                continue
                            if isinstance(out, Image.Image):
                                pil_frames = [out]
                                break

                    if not pil_frames:
                        if original_callback is not None:
                            original_callback(step, x0, x, total_steps_)
                        return

                    pil_first = pil_frames[0]
                    if pil_first.mode != "RGB":
                        pil_first = pil_first.convert("RGB")
                        pil_frames[0] = pil_first
                    self.frames.append(pil_first)

                    rate = _playback_rate(self.playback, len(pil_frames), pixel_frames)

                    if vhs is not None:
                        vhs.send(pil_frames, rate)

                    if node_id is not None and PromptServer is not None and to_node:
                        # x0_view (not x0) so the audio half of the pack can't dampen the norm.
                        x0_cpu_now = x0_view.detach().float().cpu()
                        prev_x0_cpu = state["last_x0_cpu"]
                        state["last_x0_cpu"] = x0_cpu_now

                        now = time.perf_counter()
                        step_ms = None
                        if state["last_time"] is not None:
                            step_ms = (now - state["last_time"]) * 1000.0
                            w = state["step_ms_window"]
                            w.append(step_ms)
                            if len(w) > 8:
                                w.pop(0)
                        state["last_time"] = now
                        avg_step_ms = (sum(state["step_ms_window"]) / len(state["step_ms_window"])) \
                            if state["step_ms_window"] else None
                        sigma_val = sigmas_list[step] if 0 <= step < len(sigmas_list) else None
                        sent_step = step + 1

                        def _encode_and_send(
                            pil_frames=pil_frames, x0_cpu_now=x0_cpu_now, prev_x0_cpu=prev_x0_cpu,
                            step_ms=step_ms, avg_step_ms=avg_step_ms, sigma_val=sigma_val,
                            sent_step=sent_step, total_steps_=total_steps_, rate=rate,
                        ):
                            if len(pil_frames) > 1:
                                # NVENC ~8x faster + ~5x smaller than PIL WebP when available.
                                b64, w_, h_, mime = None, 0, 0, None
                                if _NVENC_AVAILABLE:
                                    b64, w_, h_ = _encode_mp4_nvenc(pil_frames, rate, max_res)
                                    if b64:
                                        mime = "video/mp4"
                                if not b64:
                                    b64, w_, h_ = _encode_animated_webp(pil_frames, rate,
                                                                        quality, max_res)
                                    mime = "image/webp"
                            else:
                                pil_send = pil_frames[0]
                                if max_res and max_res > 0 and (pil_send.width > max_res
                                                                or pil_send.height > max_res):
                                    pil_send = ImageOps.contain(pil_send, (max_res, max_res),
                                                                Image.LANCZOS)
                                buf = pyio.BytesIO()
                                pil_send.save(buf, format="JPEG", quality=quality)
                                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                                w_, h_ = pil_send.width, pil_send.height
                                mime = "image/jpeg"

                            if not b64:
                                return

                            delta_v = None
                            if prev_x0_cpu is not None and prev_x0_cpu.shape == x0_cpu_now.shape:
                                diff = x0_cpu_now - prev_x0_cpu
                                delta_v = (diff.norm() / max(1, diff.numel()) ** 0.5).item()

                            PromptServer.instance.send_sync(
                                EVENT,
                                {
                                    "node_id": node_id, "image": b64, "mime": mime,
                                    "w": w_, "h": h_, "step": sent_step, "total": total_steps_,
                                    "sigma": sigma_val, "sigmas": None, "delta": delta_v,
                                    "step_ms": step_ms, "avg_step_ms": avg_step_ms,
                                    # Derived server-side from the shot's real length; the
                                    # browser plays at this rate rather than reading a widget.
                                    "fps": round(float(rate), 3),
                                    "source_fps": MODEL_FPS,
                                    "playback": self.playback,
                                    "frames": len(pil_frames),
                                    "mode": self.decode_mode,
                                },
                                PromptServer.instance.client_id,
                            )

                        encoder.submit(_encode_and_send)

                    state["cost"] = time.time() - t0
                    state["finished"] = time.time()
                    if self.max_overhead > 0 and state["cost"] > 1.0 \
                            and not state["throttle_logged"]:
                        state["throttle_logged"] = True
                        logging.info(
                            "[H3PreviewOverride] a preview costs %.1fs; holding it to %d%% of "
                            "the render, so previews will be spaced ~%.0fs apart. Lower "
                            "preview_frames or max_resolution for more of them.",
                            state["cost"], self.max_overhead,
                            state["cost"] * (100.0 / self.max_overhead - 1.0))
                except Exception as e:
                    # never take the generation down over a preview
                    if not state["warned"]:
                        state["warned"] = True
                        logging.warning("[H3PreviewOverride] preview failed, continuing "
                                        "without it: %r", e, exc_info=True)
            if original_callback is not None:
                original_callback(step, x0, x, total_steps_)

        # Patch every concrete decode_latent_to_preview_image — subclasses like VHS's
        # WrappedPreviewer override it and would otherwise still emit previews of their own.
        prev_methods = []
        if self.suppress_default:
            targets = [latent_preview.LatentPreviewer]
            stack = list(latent_preview.LatentPreviewer.__subclasses__())
            while stack:
                cls = stack.pop()
                targets.append(cls)
                stack.extend(cls.__subclasses__())
            for cls in targets:
                if "decode_latent_to_preview_image" in cls.__dict__:
                    prev_methods.append((cls, cls.__dict__["decode_latent_to_preview_image"]))
                    cls.decode_latent_to_preview_image = _suppressed_preview_image
        try:
            # Seeds step 1's duration measurement (sampling-start -> end of step 1).
            state["last_time"] = time.perf_counter()
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            new_callback, disable_pbar, seed, **kwargs)
        finally:
            encoder.shutdown(drain_timeout=5.0)
            for cls, prev in prev_methods:
                cls.decode_latent_to_preview_image = prev


def _register(model, wrapper):
    """Attach the wrapper where the sampler will actually look for it.

    CFGGuider reads `get_all_wrappers(OUTER_SAMPLE, self.model_options, is_model_options=True)`,
    so the model_options side is the one that matters on current builds; `clone()` deep-copies
    model_options, so this cannot leak back into the source patcher. Older builds read the
    patcher-side dict instead — checking before adding there avoids registering twice and
    firing the preview for every step twice over.
    """
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        WRAPPER_KEY, wrapper, model.model_options, is_model_options=True)
    registered = comfy.patcher_extension.get_all_wrappers(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, model.model_options,
        is_model_options=True)
    if wrapper not in registered and hasattr(model, "add_wrapper_with_key"):
        logging.info("[H3PreviewOverride] using ModelPatcher-side wrapper registration.")
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   WRAPPER_KEY, wrapper)


def _find_wrappers(model):
    """Both registration sides, since `_register` may have used either."""
    found = []
    try:
        for w in comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                model.model_options, is_model_options=True) or []:
            if isinstance(w, _PreviewOverrideWrapper):
                found.append(w)
    except Exception:
        pass
    if not found and hasattr(model, "get_wrappers"):
        try:
            found = [w for w in (model.get_wrappers(
                comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY) or [])
                if isinstance(w, _PreviewOverrideWrapper)]
        except Exception:
            pass
    return found


# ---------------------------------------------------------------------------------- nodes

class H3PreviewOverride(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3PreviewOverride",
            display_name="H3 Preview Override",
            category="MiniMax H3",
            description=(
                "Live preview of the whole shot while it denoises, shown on this node, with "
                "sigma/delta and step-time graphs. Ported from KJNodes' Preview Override and "
                "adapted for MiniMax H3: unpacks H3's packed audio+video latent (which the "
                "LTX-gated original does not), spreads the previewed frames across the whole "
                "clip, and derives playback rate from the shot's real duration instead of "
                "taking a frame rate. Wire between the model and the sampler."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Model to attach the preview to."),
                io.Vae.Input("vae", optional=True,
                             tooltip="minimax_h3_video_vae. Only needed for decode='vae (quality)'."),
                io.Combo.Input("decode", options=[DECODE_FAST, DECODE_TAEH3, DECODE_VAE],
                               default=DECODE_FAST,
                               tooltip="latent2rgb is a single matmul — effectively free, rough "
                                       "colours. tiny vae (taeh3) runs a small trained decoder "
                                       "picked in 'tiny_vae' — near-VAE colour at close to "
                                       "latent2rgb speed, and the exact real pixel-frame count, "
                                       "so timing reads true. vae is the real decoder: true "
                                       "colours, real time per preview — raise every_n_steps."),
                io.Combo.Input("playback", options=[PLAYBACK_TRUE, PLAYBACK_SOURCE],
                               default=PLAYBACK_TRUE,
                               tooltip="'true speed' (default) spreads the previewed frames "
                                       "across the shot's real duration, so the preview lasts "
                                       "exactly as long as the finished clip. With latent2rgb "
                                       "that caps at 24 / 3.35 fps, because there is one image "
                                       "per latent frame and H3 compresses time by that much. "
                                       "'source fps' plays them at H3's native 24 flat: motion "
                                       "reads at normal speed, the clip ends early. Judge timing "
                                       "with the first, movement with the second."),
                io.Combo.Input("preview_target", options=[TARGET_NODE, TARGET_SAMPLER, TARGET_BOTH],
                               default=TARGET_NODE,
                               tooltip="Where the preview appears: on this node, in the sampler's "
                                       "usual preview slot (needs VideoHelperSuite), or both."),
                io.Int.Input("preview_frames", default=24, min=1, max=512, step=1,
                             tooltip="How many frames to show, spread evenly across the WHOLE "
                                     "clip — never a truncated opening section. For latent2rgb "
                                     "and vae the latent is thinned before decoding, so this is "
                                     "also the main cost knob; for tiny vae (taeh3) the whole "
                                     "clip is decoded regardless (the decoder chains state frame "
                                     "to frame), so there it caps transfer size, not decode time."),
                io.Int.Input("max_resolution", default=512, min=0, max=8192, step=8,
                             tooltip="Max preview side in pixels. 0 = full decoded resolution, "
                                     "no downscale."),
                io.Int.Input("jpeg_quality", default=80, min=30, max=100, step=1,
                             tooltip="Quality for the preview transport (JPEG for a single "
                                     "frame, WebP for an animation). Ignored when NVENC MP4 "
                                     "encoding is available and used."),
                io.Int.Input("every_n_steps", default=1, min=1, max=50, step=1, optional=True,
                             tooltip="Never preview more often than every N sampler steps."),
                io.Int.Input("max_preview_overhead", default=25, min=0, max=100, step=5,
                             optional=True,
                             tooltip="Cap on how much of the render time previews may use, in "
                                     "percent. A full VAE decode can cost tens of seconds per "
                                     "preview; this spaces them out automatically instead of "
                                     "stalling the run. 0 disables the cap."),
                io.Boolean.Input("suppress_default_preview", default=True, optional=True,
                                 tooltip="Suppress ComfyUI's built-in single-frame preview while "
                                         "this runs, so only this node's frame updates. The "
                                         "progress bar still advances normally."),
                # NEW WIDGETS GO LAST. ComfyUI serialises widget values positionally, so
                # inserting one in the middle hands every value after it to the wrong input
                # in workflows that were saved before it existed.
                io.Combo.Input("tiny_vae",
                               options=["none"] + folder_paths.get_filename_list("vae_approx"),
                               default="none", optional=True,
                               tooltip="Tiny temporal VAE (taehv format) from models/vae_approx, "
                                       "used when decode='tiny vae (taeh3)'. A checkpoint trained "
                                       "for H3's 24-channel, patch-size-2 latent — commonly named "
                                       "'taeh3'. A checkpoint with the wrong channel count is "
                                       "rejected at run time and the node falls back to latent2rgb."),
                io.Boolean.Input("show_vae_input", default=True, optional=True,
                                 tooltip="Show the 'vae' socket on the node face. Purely a "
                                         "JS-side declutter toggle — turning it off hides (and "
                                         "disconnects) the socket for when you're only using "
                                         "'latent2rgb (fast)' or 'tiny vae (taeh3)'; turn it back "
                                         "on and rewire before switching to "
                                         "decode='vae (quality)'."),
            ],
            outputs=[io.Model.Output(tooltip="Model with the preview attached.")],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, decode=DECODE_FAST, playback=PLAYBACK_TRUE,
                preview_target=TARGET_NODE, preview_frames=24, max_resolution=512,
                jpeg_quality=80, every_n_steps=1, max_preview_overhead=25,
                suppress_default_preview=True, vae=None, tiny_vae="none",
                show_vae_input=True) -> io.NodeOutput:
        # show_vae_input is JS-only (it toggles the socket's visibility on the node face) —
        # accepted here only because ComfyUI passes every schema input by keyword.
        del show_vae_input
        if decode == DECODE_VAE and vae is None:
            raise ValueError(
                "H3 Preview Override: decode is set to 'vae (quality)' but no VAE is "
                "connected. Wire minimax_h3_video_vae into 'vae', or switch decode back to "
                "'latent2rgb (fast)'."
            )
        if decode == DECODE_TAEH3 and (tiny_vae is None or tiny_vae == "none"):
            raise ValueError(
                "H3 Preview Override: decode is set to 'tiny vae (taeh3)' but no tiny VAE is "
                "selected. Drop a taehv checkpoint trained for H3's 24-channel latent "
                "(commonly named 'taeh3') into models/vae_approx and pick it from 'tiny_vae', "
                "or switch decode back to 'latent2rgb (fast)'."
            )

        m = model.clone()
        _register(m, _PreviewOverrideWrapper(
            str(cls.hidden.unique_id), decode, max_resolution, jpeg_quality,
            suppress_default_preview, preview_frames, playback, preview_target,
            every_n_steps, max_preview_overhead, vae, tiny_vae))
        return io.NodeOutput(m)


class GetH3PreviewFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetH3PreviewFrames",
            display_name="Get H3 Preview Frames",
            category="MiniMax H3",
            description=(
                "Returns the frames captured by H3 Preview Override during the most recent "
                "sampling — one image per rendered preview, so it reads as a timelapse of the "
                "shot converging. Wire 'model' from H3 Preview Override (the same one feeding "
                "the sampler) and 'after_sample' from after the sampler to force the order."
            ),
            inputs=[
                io.Model.Input("model",
                               tooltip="The model output by H3 Preview Override (used to locate "
                                       "the captured frames)."),
                io.MultiType.Input(
                    "after_sample", [io.Latent, io.Image],
                    tooltip="Anything from after the sampler (LATENT or IMAGE). The value is "
                            "ignored — it just forces this node to run after sampling."),
            ],
            outputs=[io.Image.Output(display_name="frames")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, after_sample) -> io.NodeOutput:
        wrappers = _find_wrappers(model)
        if not wrappers:
            raise RuntimeError(
                "Get H3 Preview Frames: no H3 Preview Override wrapper found on this model.")
        frames = wrappers[-1].frames
        if not frames:
            raise RuntimeError(
                "Get H3 Preview Frames: no frames captured. Ensure the sampler ran with this "
                "model, and that preview_target is 'node' or 'both'.")
        tensors = []
        for pil in frames:
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        # Frames can differ in size if max_resolution changed mid-run; stack needs one shape.
        if len({t.shape for t in tensors}) > 1:
            h, w = tensors[0].shape[0], tensors[0].shape[1]
            tensors = [t if t.shape[:2] == (h, w) else torch.from_numpy(
                np.asarray(Image.fromarray((t.numpy() * 255).astype(np.uint8)).resize(
                    (w, h), Image.LANCZOS), dtype=np.float32) / 255.0) for t in tensors]
        return io.NodeOutput(torch.stack(tensors, dim=0))
