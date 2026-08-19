"""Live sampling preview for MiniMax H3, with an optional taeh3 tiny-VAE decode path.

Why this exists
---------------
ComfyUI core does ship `latent_rgb_factors` for MiniMaxH3Video, so previews are not
missing — but `Latent2RGBPreviewer` renders `x0[0, :, 0]`, i.e. the first latent frame
only. You watch a single still while a five-second shot is being sampled.

KJNodes' Preview Override (github.com/kijai/ComfyUI-KJNodes) does the good version of
this — in-node preview, optional full VAE decode, and a fast temporal tiny-VAE ("taehv")
decode path — but its video paths are gated on `_is_ltx_latent_format` /
`_is_ltx2_diffusion_model`, and nothing there unpacks H3's packed AV latent, so on
MiniMax it falls through to the same single frame.

This node is the H3 equivalent of KJNodes' Preview Override: it unpacks H3's flat
audio+video latent pack the way `MiniMax H3 Preview Override`
(github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director) does, and adds KJNodes'
`tiny_vae` (taehv/taeh3) decode path on top of that node's existing `latent2rgb` and
`vae` modes — a `decode='tiny vae (taeh3)'` option that runs a small trained decoder
(`tiny_vae.py`, ported from KJNodes) instead of either the single free matmul or the
full, slow VAE. A taehv checkpoint trained for H3's 24-channel, patch-size-2 latent (a
"taeh3" checkpoint) gets near-VAE colour accuracy at close to latent2rgb speed, and the
exact pixel-frame count H3 will actually output, because `TAEHVDecoder` in `tiny_vae.py`
detects that shape and switches to H3's own 17-pixel-frames-per-5-latent-tokens chunking.
Note that this node decodes the *whole* clip in that mode and subsamples afterwards,
where KJNodes decodes only a prefix to bound its cost — see `_tiny_vae_decode`.

The one non-obvious detail
--------------------------
`CFGGuider.sample` packs the video and audio streams into ONE flat tensor and only then
wraps the callback with the nested view. That wrapper sits *behind* an OUTER_SAMPLE
wrapper in the call chain, so what reaches this callback is the flat pack, not the
NestedTensor — `_video_stream` unpacks it with core's own `unpack_latents`.
"""

import base64
import io as _io
import logging
import struct
import time

import torch
import torch.nn.functional as F
from PIL import Image

import comfy.patcher_extension
import comfy.utils
import folder_paths
import latent_preview
import server
from comfy_api.latest import io
from protocol import BinaryEventTypes

from .tiny_vae import load_tiny_vae_decoder

log = logging.getLogger(__name__)

EVENT = "minimax_h3_preview"
DECODE_FAST = "latent2rgb (fast)"
DECODE_TAEH3 = "tiny vae (taeh3)"
DECODE_VAE = "vae (quality)"
PLAYBACK_TRUE = "true speed"
PLAYBACK_SOURCE = "source fps"
MODEL_FPS = 24.0            # H3's native output rate
TARGET_NODE = "node"
TARGET_SAMPLER = "sampler (VHS)"
TARGET_BOTH = "both"


def _video_stream(x0, latent_shapes=None):
    """Pull the [B, C, T, h, w] video latent out of whatever the sampler handed us."""
    if x0 is None:
        return None
    if getattr(x0, "is_nested", False):
        return x0.tensors[0]
    if x0.ndim == 5:
        return x0
    if latent_shapes and len(latent_shapes) > 1:
        return comfy.utils.unpack_latents(x0, list(latent_shapes))[0]
    return None


def _pick_frames(video, max_frames):
    """Evenly thin [B, C, T, h, w] down to at most max_frames along T."""
    t = video.shape[2]
    if max_frames <= 0 or t <= max_frames:
        return video
    idx = torch.linspace(0, t - 1, max_frames).round().long().unique()
    return video[:, :, idx]


def pixel_frames_from_latent_t(latent_t):
    """How many output frames one H3 video latent covers.

    The latent is compressed ~3.35x in time (core's `video_latent_t`: 17k+5 pixel frames
    become 5k+2 latent frames), so a preview that plays latent frames at the video's fps
    runs more than three times too fast. Inverting that mapping is what keeps the preview
    honest about the shot's real speed.
    """
    if latent_t <= 2:
        return 5
    k, remainder = divmod(int(latent_t) - 2, 5)
    if remainder == 0:
        return 17 * k + 5
    return max(1, int(round(latent_t * 17.0 / 5.0)))   # off-grid latent: approximate


class _RGBFactors:
    def __init__(self, latent_format):
        factors = getattr(latent_format, "latent_rgb_factors", None)
        if factors is None:
            raise ValueError("latent format has no latent_rgb_factors")
        # stored as an F.linear weight: [out=3, in=C] -> channel count is shape[1]
        self.w = torch.tensor(factors, device="cpu").transpose(0, 1)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        self.b = torch.tensor(bias, device="cpu") if bias is not None else None

    def __call__(self, video):
        """[B, C, T, h, w] -> [N, h, w, 3] in 0..1"""
        chans = self.w.shape[1]
        moved = video.movedim(2, 1)                       # [B, T, C, h, w]
        # flatten batch-major; take the shape AFTER the movedim, not the caller's
        x = moved.reshape((-1,) + tuple(moved.shape[-3:]))[:, :chans].to(torch.float32)
        w = self.w.to(dtype=x.dtype, device=x.device)
        b = self.b.to(dtype=x.dtype, device=x.device) if self.b is not None else None
        return ((F.linear(x.movedim(1, -1), w, bias=b) + 1.0) / 2.0).clamp(0, 1)


def _vae_decode(vae, video):
    """Full-quality decode of [B, C, T, h, w] -> [N, h, w, 3] in 0..1."""
    images = vae.decode(video)
    if images.ndim == 5:                       # [B, T, h, w, 3]
        images = images.reshape(-1, *images.shape[-3:])
    return images.clamp(0, 1).to(torch.float32).cpu()


def _tiny_vae_decode(decoder, video, preview_frames):
    """Tiny-VAE (taehv/taeh3) decode of [B, C, T, h, w] -> [N, h, w, 3] in 0..1.

    Always decodes the WHOLE clip, then evenly thins the resulting pixel frames — it
    never decodes a prefix of the latent.

    The distinction matters and is the whole reason this node exists. The decoder chains
    state frame to frame (MemBlocks), so a mid-clip frame cannot be decoded without
    everything before it having been decoded first. The cheap way out is to decode only
    the first `preview_frames` latent frames, which is what KJNodes' node does to bound
    its per-step cost — but that shows the opening fraction of the shot and calls it the
    preview. Worse here: `pixel_frames` is measured off the full latent, so those
    prefix-only frames then get spread across the whole clip's duration, and the preview
    reads as both truncated and too slow.

    So this pays for the full decode instead — taeh3 is cheap enough that it is the right
    trade — and subsamples afterwards. `preview_frames` therefore caps how many *images*
    you get, spread across the entire shot, exactly as it does for the other two decode
    modes; it no longer bounds the decode cost for this mode.
    """
    images = decoder.decode_video(video, frame_indices=None)
    total = int(images.shape[0])
    if preview_frames > 0 and total > preview_frames:
        idx = torch.linspace(0, total - 1, preview_frames).round().long().unique()
        images = images[idx]
    return images.clamp(0, 1).to(torch.float32).cpu()


def _to_pil(images, max_res):
    """Render frames at `max_res` on the long edge — a target, not just a ceiling.

    latent2rgb frames arrive at latent resolution (a 1344x768 shot is an 84x48 grid), so
    without an upscale the preview is a postage stamp; with a nearest-neighbour upscale it
    is a mosaic. Smooth interpolation reads as "approximate", which is what it is — switch
    decode to 'vae (quality)' (or try 'tiny vae (taeh3)') for real detail.
    """
    out = []
    for frame in images:
        arr = (frame * 255.0).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(arr)
        if max_res > 0:
            longest = max(img.width, img.height)
            if longest != max_res and longest > 0:
                scale = max_res / float(longest)
                size = (max(1, int(round(img.width * scale))),
                        max(1, int(round(img.height * scale))))
                img = img.resize(size, Image.LANCZOS if scale < 1.0 else Image.BICUBIC)
        out.append(img)
    return out


def _encode_animated_webp(frames, fps, quality):
    if not frames:
        return None
    buf = _io.BytesIO()
    try:
        frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:],
                       duration=max(1, int(round(1000 / max(1, fps)))), loop=0,
                       quality=quality, method=0)
    except Exception as e:
        log.warning("[H3PreviewPlus] animated WebP encode failed: %s", e)
        return None
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


class _VHSStreamer:
    """Streams individual frames to VideoHelperSuite's animated latent-preview player."""

    def __init__(self, rate):
        self.rate = max(1, int(rate))
        self.first = True
        self.last_time = 0.0
        self.cursor = 0

    def send(self, pil_frames, rate=None):
        srv = server.PromptServer.instance
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
            message = _io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(i.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", node_id))
            pil_frames[i].save(message, format="JPEG", quality=95)
            srv.send_sync(BinaryEventTypes.PREVIEW_IMAGE, message.getvalue(), srv.client_id)
        self.cursor = (self.cursor + count) % total
        return count


class _OuterSampleWrapper:
    def __init__(self, node_id, decode_mode, vae, max_resolution, preview_frames,
                 preview_fps, webp_quality, every_n_steps, suppress_default, target,
                 max_overhead=25, playback=None, tiny_vae_name="none"):
        self.node_id = node_id
        self.decode_mode = decode_mode
        self.vae = vae
        self.max_resolution = int(max_resolution)
        self.preview_frames = int(preview_frames)
        self.preview_fps = float(preview_fps)
        self.webp_quality = int(webp_quality)
        self.every_n_steps = max(1, int(every_n_steps))
        self.suppress_default = bool(suppress_default)
        self.target = target
        self.max_overhead = max(0, min(100, int(max_overhead)))
        self.playback = playback or PLAYBACK_TRUE
        self.tiny_vae_name = tiny_vae_name or "none"

    def _rate_for(self, shown, pixel_frames):
        """Frames per second to play `shown` images at.

        Two honest answers, and the node used to pick one for you:

        `true speed` spreads the images across the shot's real duration, so the preview
        lasts as long as the finished clip. With latent2rgb that caps out at
        preview_fps / 3.35 — one image per latent frame, and H3 compresses time by that
        much — which looks like a setting being ignored if nobody says so.

        `source fps` plays them at preview_fps flat. That is what ComfyUI's own preview and
        the other packs do, and it is why they show a round 24: the motion reads at normal
        speed but the clip is over in a third of the time. Useful for judging movement,
        misleading about timing.
        """
        if self.playback == PLAYBACK_SOURCE:
            return max(0.1, self.preview_fps)
        return max(0.1, self.preview_fps * shown / max(1, pixel_frames))

    def _send_to_node(self, b64, n_frames, step, total_steps, ms, rate,
                       sigma=None, delta=None, step_ms=None, avg_step_ms=None):
        server.PromptServer.instance.send_sync(EVENT, {
            "node_id": self.node_id, "webp": b64, "frames": n_frames,
            "fps": round(float(rate), 2), "source_fps": round(float(self.preview_fps), 2),
            "step": step, "total_steps": total_steps, "ms": ms, "mode": self.decode_mode,
            "playback": self.playback,
            # Graph data for the node's interactive σ/Δ and step-time panels (ported from
            # KJNodes' Preview Override). sigma/delta/step_ms ride along with the image, so
            # they update on the same cadence every_n_steps/max_preview_overhead allow —
            # slower than raw sampler steps if either is throttling this run.
            "sigma": sigma, "delta": delta, "step_ms": step_ms, "avg_step_ms": avg_step_ms,
        })

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, **kwargs):
        guider = executor.class_obj
        latent_shapes = kwargs.get("latent_shapes")
        latent_format = guider.model_patcher.model.latent_format
        to_node = self.target in (TARGET_NODE, TARGET_BOTH)

        # The full σ schedule for this run, sent once up front so the node's graph can draw
        # its x-axis before the first preview lands. Also doubles as the per-step sigma
        # lookup below (KJNodes indexes the same list the same way).
        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        if to_node and sigmas_list:
            server.PromptServer.instance.send_sync(EVENT, {
                "node_id": self.node_id, "step": 0,
                "total_steps": max(0, len(sigmas_list) - 1), "sigmas": sigmas_list,
            })

        # Pre-seed the Δ baseline with the model's own first transformation (noise * sigma0)
        # so the very first rendered preview already has a real delta instead of a blank one.
        last_x0_cpu = None
        try:
            if sigmas_list:
                s0 = sigmas[0].to(noise.device) if hasattr(sigmas[0], "to") else sigmas[0]
                seeded = _video_stream(noise * s0, latent_shapes)
                if seeded is not None and seeded.ndim == 5:
                    seeded = _pick_frames(seeded, self.preview_frames)
                    last_x0_cpu = seeded.detach().float().cpu()
        except Exception as e:
            log.warning("[H3PreviewPlus] initial delta seed failed: %s", e)

        to_rgb = None
        if self.decode_mode != DECODE_VAE:
            try:
                to_rgb = _RGBFactors(latent_format)
            except Exception as e:
                log.warning("[H3PreviewPlus] preview unavailable: %s", e)

        tiny_vae_decoder = None
        if self.decode_mode == DECODE_TAEH3:
            tiny_vae_decoder = load_tiny_vae_decoder(self.tiny_vae_name)
            if tiny_vae_decoder is not None and latent_shapes and len(latent_shapes[0]) >= 2:
                channels = int(latent_shapes[0][1])
                if channels != tiny_vae_decoder.latent_channels:
                    log.warning(
                        "[H3PreviewPlus] tiny_vae '%s' decodes %d-channel latents but "
                        "this model's are %d-channel; falling back to latent2rgb.",
                        self.tiny_vae_name, tiny_vae_decoder.latent_channels, channels)
                    tiny_vae_decoder = None
            if tiny_vae_decoder is None and to_rgb is None:
                try:
                    to_rgb = _RGBFactors(latent_format)
                except Exception as e:
                    log.warning("[H3PreviewPlus] preview unavailable: %s", e)

        vhs = _VHSStreamer(self.preview_fps) if self.target in (TARGET_SAMPLER, TARGET_BOTH) else None

        # Core's previewer is built before we are reached, so suppression has to happen on
        # the class it goes through. Restored in the finally below, always.
        original_decode = latent_preview.LatentPreviewer.decode_latent_to_preview_image
        if self.suppress_default:
            latent_preview.LatentPreviewer.decode_latent_to_preview_image = \
                lambda self_, preview_format, x0: None

        original_cb = callback
        state = {"warned": False, "sent": 0, "cost": 0.0, "finished": 0.0, "anim": 0.0,
                 "throttle_logged": False, "cap_logged": False, "last_time": None,
                 "step_ms_window": [], "last_x0_cpu": last_x0_cpu}
        log.info("[H3PreviewPlus] preview: %s, target=%s, <=%d frames @%d fps, max %dpx.",
                 self.decode_mode, self.target, self.preview_frames, self.preview_fps,
                 self.max_resolution)

        def _should_skip(now):
            gap = throttle_gap(state["cost"], self.max_overhead)
            if to_node:
                # Replacing the <img> restarts the animation from frame one. Send a new one
                # every step and a five-second loop never gets past its first second — it
                # reads as a stuck, crawling preview. Let each animation play through.
                gap = max(gap, state["anim"])
            return gap > 0 and (now - state["finished"]) < gap

        def combined(step, x0, x, total_steps):
            if (to_rgb is not None or self.vae is not None or tiny_vae_decoder is not None) \
                    and x0 is not None and step % self.every_n_steps == 0 \
                    and not _should_skip(time.time()):
                now_t = time.time()
                # Time since the last *rendered* preview, not the raw sampler step — with
                # every_n_steps or max_preview_overhead throttling, several sampler steps can
                # elapse between two renders, and this graphs exactly that gap, honestly.
                step_ms = None
                if state["last_time"] is not None:
                    step_ms = (now_t - state["last_time"]) * 1000.0
                    window = state["step_ms_window"]
                    window.append(step_ms)
                    if len(window) > 8:
                        window.pop(0)
                state["last_time"] = now_t
                avg_step_ms = (sum(state["step_ms_window"]) / len(state["step_ms_window"])) \
                    if state["step_ms_window"] else None
                t0 = now_t
                try:
                    video = _video_stream(x0, latent_shapes)
                    if video is not None and video.ndim == 5:
                        pixel_frames = pixel_frames_from_latent_t(int(video.shape[2]))
                        thinned = _pick_frames(video, self.preview_frames)
                        if self.decode_mode == DECODE_TAEH3 and tiny_vae_decoder is not None:
                            images = _tiny_vae_decode(tiny_vae_decoder, video, self.preview_frames)
                        elif self.decode_mode == DECODE_VAE and self.vae is not None:
                            images = _vae_decode(self.vae, thinned)
                        else:
                            images = to_rgb(thinned)
                        frames = _to_pil(images, self.max_resolution)

                        # Δ: mean per-element magnitude of change since the last rendered
                        # preview, on the same thinned tensor regardless of decode mode so
                        # shapes line up step to step. A rough "how much is still moving"
                        # signal for the graph, not tied to any one decode's own scale.
                        delta = None
                        x0_cpu_now = thinned.detach().float().cpu()
                        prev_x0_cpu = state["last_x0_cpu"]
                        if prev_x0_cpu is not None and prev_x0_cpu.shape == x0_cpu_now.shape:
                            diff = x0_cpu_now - prev_x0_cpu
                            delta = (diff.norm() / max(1, diff.numel()) ** 0.5).item()
                        state["last_x0_cpu"] = x0_cpu_now
                        # The shot lasts pixel_frames / fps seconds. Spread however many
                        # frames we ended up with across exactly that long. Counting them
                        # after the decode matters: latent2rgb yields one image per latent
                        # frame, the VAE and taeh3 whole-clip decode expand each latent
                        # frame into ~3.35 of them.
                        rate = self._rate_for(len(frames), pixel_frames)
                        if not state["cap_logged"] and self.playback == PLAYBACK_TRUE \
                                and rate < self.preview_fps - 0.05:
                            state["cap_logged"] = True
                            log.info("[H3PreviewPlus] preview plays at %.1f fps, not %.0f: "
                                     "%d frame(s) spread over the shot's %.2fs so it lasts as "
                                     "long as the finished clip. %s Switch playback to '%s' to "
                                     "play them at %.0f fps instead — the motion reads normally, "
                                     "the clip ends early.",
                                     rate, self.preview_fps, len(frames),
                                     pixel_frames / MODEL_FPS,
                                     "latent2rgb has one image per latent frame, so it cannot "
                                     "exceed %.1f fps here." % (self.preview_fps * 5.0 / 17.0)
                                     if self.decode_mode == DECODE_FAST else
                                     "Raising preview_frames raises it.",
                                     PLAYBACK_SOURCE, self.preview_fps)
                        if to_node:
                            b64 = _encode_animated_webp(frames, rate, self.webp_quality)
                            if b64:
                                sigma_val = sigmas_list[step] if 0 <= step < len(sigmas_list) else None
                                self._send_to_node(b64, len(frames), step + 1, total_steps,
                                                   int((time.time() - t0) * 1000), rate,
                                                   sigma_val, delta, step_ms, avg_step_ms)
                        if vhs is not None:
                            vhs.send(frames, rate)
                        state["sent"] += len(frames)
                        state["cost"] = time.time() - t0
                        state["finished"] = time.time()
                        state["anim"] = len(frames) / max(0.1, rate) if to_node else 0.0
                        if self.max_overhead > 0 and state["cost"] > 1.0 \
                                and not state["throttle_logged"]:
                            state["throttle_logged"] = True
                            log.info("[H3PreviewPlus] a preview costs %.1fs; holding it to "
                                     "%d%% of the render, so previews will be spaced ~%.0fs "
                                     "apart. Lower preview_frames or max_resolution for more "
                                     "of them.", state["cost"], self.max_overhead,
                                     state["cost"] * (100.0 / self.max_overhead - 1.0))
                except Exception as e:
                    # never take the generation down over a preview
                    if not state["warned"]:
                        state["warned"] = True
                        log.warning("[H3PreviewPlus] preview failed, continuing without "
                                    "it: %r", e, exc_info=True)
            if original_cb is not None:
                original_cb(step, x0, x, total_steps)

        try:
            out = executor(noise, latent_image, sampler, sigmas, denoise_mask, combined,
                           disable_pbar, seed, **kwargs)
        finally:
            latent_preview.LatentPreviewer.decode_latent_to_preview_image = original_decode
        log.info("[H3PreviewPlus] preview rendered %d frames.", state["sent"])
        return out


class MiniMaxH3PreviewPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PreviewPlusCS",
            display_name="MiniMax H3 Preview Plus",
            category="MiniMax H3",
            description=(
                "Live preview of the whole shot while it denoises, shown on this node. "
                "Core previews only ever draw the first latent frame; MiniMax H3's packed "
                "audio+video latent also has to be unpacked first, which the LTX preview "
                "nodes do not do. Combines that H3 unpacking with KJNodes' taeh3/tiny-VAE "
                "decode path, so 'decode' has a fast, a near-VAE-quality, and a "
                "full-quality option. Wire between the Director's model output and the "
                "sampler."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Model to attach the preview to."),
                io.Vae.Input("vae", optional=True,
                             tooltip="minimax_h3_video_vae. Only needed for decode='vae (quality)'."),
                io.Combo.Input("decode", options=[DECODE_FAST, DECODE_TAEH3, DECODE_VAE], default=DECODE_FAST,
                               tooltip="latent2rgb is a single matmul — effectively free, rough colours. "
                                       "tiny vae (taeh3) runs a small trained decoder, picked in "
                                       "'tiny_vae' — near-VAE colour accuracy at close to latent2rgb "
                                       "speed, and the exact real pixel-frame count, so timing reads "
                                       "true. vae is the real decoder: true colours, but it costs real "
                                       "time per preview, so raise every_n_steps with it."),
                io.Combo.Input("preview_target", options=[TARGET_NODE, TARGET_SAMPLER, TARGET_BOTH],
                               default=TARGET_NODE,
                               tooltip="Where the preview appears: on this node, in the sampler's usual "
                                       "preview slot (needs VideoHelperSuite), or both."),
                io.Int.Input("max_resolution", default=512, min=64, max=2048, step=32,
                             tooltip="Long edge of the preview image. With latent2rgb the source "
                                     "is latent-sized (a 1344x768 shot is an 84x48 grid), so this "
                                     "upscales — smooth, but soft. Use decode='vae (quality)' or "
                                     "'tiny vae (taeh3)' when you need to judge detail."),
                io.Int.Input("preview_frames", default=24, min=1, max=512, step=1,
                             tooltip="How many frames of the shot to show, thinned evenly across "
                                     "the whole clip so the preview always spans the finished "
                                     "shot. For latent2rgb and vae this also caps the decode cost, "
                                     "because the latent is thinned before decoding. For tiny vae "
                                     "(taeh3) it does not: that decoder chains state frame to "
                                     "frame, so the whole clip is decoded and then subsampled — "
                                     "lower this to cut encode/transfer size, not decode time."),
                io.Float.Input("preview_fps", default=24.0, min=1.0, max=60.0, step=1.0,
                               tooltip="The shot's own frame rate — 24 for H3. FLOAT so the "
                                       "Director's 'fps' output can be wired straight in. "
                                       "Whether the preview actually plays at this rate "
                                       "depends on 'playback'; with 'true speed' it is a "
                                       "ceiling, not a promise."),
                io.Int.Input("webp_quality", default=80, min=1, max=100, step=1, optional=True,
                             tooltip="WebP quality of the animation sent to the node."),
                io.Int.Input("every_n_steps", default=1, min=1, max=50, step=1, optional=True,
                             tooltip="Never preview more often than every N sampler steps."),
                io.Int.Input("max_preview_overhead", default=25, min=0, max=100, step=5, optional=True,
                             tooltip="Cap on how much of the render time previews may use, in "
                                     "percent. A full VAE decode can cost tens of seconds per "
                                     "preview; this spaces them out automatically instead of "
                                     "stalling the run. 0 disables the cap."),
                io.Boolean.Input("suppress_default_preview", default=True, optional=True,
                                 tooltip="Hide ComfyUI's built-in single-frame preview while this runs."),
                # NEW WIDGETS GO LAST. ComfyUI serialises widget values positionally, so
                # inserting one in the middle shifts every value after it in workflows that
                # were saved before it existed — this one first landed between preview_fps
                # and webp_quality and handed a saved 80 to a combo that has no such option.
                io.Combo.Input("playback", options=[PLAYBACK_TRUE, PLAYBACK_SOURCE],
                               default=PLAYBACK_TRUE, optional=True,
                               tooltip="'true speed' spreads the sampled frames across the "
                                       "shot's real duration, so the preview lasts exactly as "
                                       "long as the finished clip — but with latent2rgb that "
                                       "caps at preview_fps / 3.35, because there is one image "
                                       "per latent frame and H3 compresses time by that much. "
                                       "'source fps' plays them at preview_fps flat, like "
                                       "ComfyUI's own preview: motion reads at normal speed, "
                                       "the clip ends early. Judge timing with the first, "
                                       "movement with the second."),
                io.Combo.Input("tiny_vae", options=["none"] + folder_paths.get_filename_list("vae_approx"),
                               default="none", optional=True,
                               tooltip="Tiny temporal VAE (taehv format) from models/vae_approx, used "
                                       "when decode='tiny vae (taeh3)'. A checkpoint trained for H3's "
                                       "24-channel, patch-size-2 latent — commonly named 'taeh3' — "
                                       "gets near-VAE colour accuracy at close to latent2rgb speed. A "
                                       "checkpoint with the wrong channel count is rejected at run "
                                       "time and the node falls back to latent2rgb."),
                io.Boolean.Input("show_vae_input", default=True, optional=True,
                                 tooltip="Show the 'vae' socket on the node face. Purely a JS-side "
                                         "declutter toggle — turning it off hides (and disconnects) "
                                         "the socket for when you're only using 'latent2rgb (fast)' "
                                         "or 'tiny vae (taeh3)'; turn it back on and rewire before "
                                         "switching to decode='vae (quality)'."),
            ],
            outputs=[io.Model.Output(tooltip="Model with the preview attached.")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, decode=DECODE_FAST, preview_target=TARGET_NODE, max_resolution=512,
                preview_frames=24, preview_fps=24.0, playback=PLAYBACK_TRUE, webp_quality=80,
                every_n_steps=1, max_preview_overhead=25, suppress_default_preview=True,
                vae=None, tiny_vae="none", show_vae_input=True) -> io.NodeOutput:
        # show_vae_input is JS-only (it toggles the socket's visibility on the node face) —
        # accepted here only because ComfyUI passes every schema input by keyword.
        del show_vae_input
        if decode == DECODE_VAE and vae is None:
            raise ValueError(
                "MiniMax H3 Preview Plus: decode is set to 'vae (quality)' but no VAE is "
                "connected. Wire minimax_h3_video_vae into 'vae', or switch decode back to "
                "'latent2rgb (fast)'."
            )
        if decode == DECODE_TAEH3 and (tiny_vae is None or tiny_vae == "none"):
            raise ValueError(
                "MiniMax H3 Preview Plus: decode is set to 'tiny vae (taeh3)' but no tiny "
                "VAE is selected. Drop a taehv checkpoint trained for H3's 24-channel latent "
                "(commonly named 'taeh3') into models/vae_approx and pick it from 'tiny_vae', "
                "or switch decode back to 'latent2rgb (fast)'."
            )

        m = model.clone()
        wrapper = _OuterSampleWrapper(
            str(cls.hidden.unique_id), decode, vae, max_resolution, preview_frames,
            preview_fps, webp_quality, every_n_steps, suppress_default_preview,
            preview_target, max_preview_overhead, playback, tiny_vae)

        # Register where the sampler actually looks: CFGGuider reads
        #   get_all_wrappers(OUTER_SAMPLE, self.model_options, is_model_options=True)
        # `clone()` deep-copies model_options, so this cannot leak into the source patcher.
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "minimax_h3_preview", wrapper, m.model_options, is_model_options=True)

        registered = comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, m.model_options,
            is_model_options=True)
        if wrapper not in registered and hasattr(m, "add_wrapper_with_key"):
            # a build that still reads the patcher-side dict; checking first avoids
            # registering twice and firing the preview for every step twice over
            log.info("[H3PreviewPlus] using ModelPatcher-side wrapper registration.")
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   "minimax_h3_preview", wrapper)
        return io.NodeOutput(m)
