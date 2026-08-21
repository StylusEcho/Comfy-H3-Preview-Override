"""Offline checks for h3_preview_override.py — the half that needs ComfyUI on the path
(comfy.patcher_extension, comfy.utils, comfy.model_management, latent_preview, server,
comfy_api.latest, folder_paths, comfy.taesd.taesd/taehv). Run it with the same interpreter
ComfyUI uses, from inside `custom_nodes/Comfy-H3-Preview-Override`:

    python test_node.py

Nothing here samples, decodes or talks to a network.

Two things make the import work at all, and both are easy to trip over:

* the folder name has hyphens, so the package is loaded under a synthetic module name;
* the node file registers no server routes at import time, but `server.PromptServer`
  still needs an `.instance` attribute for the module-level `import server` to resolve
  cleanly against a real ComfyUI checkout, so it's stubbed the same way regardless.
"""
import importlib.util
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.dirname(os.path.dirname(HERE))      # custom_nodes/<pack> -> ComfyUI

for path in (COMFY_ROOT, os.path.dirname(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _stub_prompt_server():
    import server

    if getattr(server.PromptServer, "instance", None) is not None:
        return

    class _Routes:
        def post(self, *_a, **_k):
            return lambda fn: fn

        def get(self, *_a, **_k):
            return lambda fn: fn

    server.PromptServer.instance = types.SimpleNamespace(routes=_Routes())


def _load_package():
    _stub_prompt_server()
    name = "h3_preview_override_undertest"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, "__init__.py"),
        submodule_search_locations=[HERE])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return sys.modules[name + ".h3_preview_override"]


preview = _load_package()
_results = []


def check(name, got, want):
    _results.append((got == want, name, got, want))


def check_raises(name, fn, needle):
    try:
        fn()
    except Exception as e:                                     # noqa: BLE001 - that is the check
        ok = needle in str(e)
        _results.append((ok, name, "raised %r" % str(e)[:90] if not ok else "raised", "raised"))
        return
    _results.append((False, name, "did not raise", "raised"))


# -------------------------------------------------- schema vs execute() signature
# ComfyUI passes every input by keyword, so an input declared in the schema with no
# matching parameter is a TypeError at run time and nowhere earlier — the whole graph
# dies on the node, after the model has loaded. Cheap to catch here instead.
def _schema_inputs(node_cls):
    return [i.id for i in node_cls.define_schema().inputs]


for node_cls in (preview.H3PreviewOverride, preview.GetH3PreviewFrames):
    params = inspect.signature(node_cls.execute.__func__).parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = [] if accepts_kwargs else [
        n for n in _schema_inputs(node_cls) if n not in params]
    check("%s: every schema input has an execute() parameter" % node_cls.__name__, missing, [])

# show_vae_input was added last (after tiny_vae) and is a widget, not a socket older
# workflows already carry a value for — it has to stay last, or a workflow saved before
# it existed hands its values to the wrong widgets.
check("show_vae_input is the last widget (positional-serialisation trap)",
      _schema_inputs(preview.H3PreviewOverride)[-1], "show_vae_input")

# `options`/`default` are ComfyUI-internal attribute names on the schema objects; read
# them defensively so a rename upstream fails this one check instead of the whole file.
def _input(node_cls, input_id):
    return next(i for i in node_cls.define_schema().inputs if i.id == input_id)


def _attr(obj, name, fallback="<no such attribute>"):
    return getattr(obj, name, fallback)


# The frame-rate widget is gone; playback mode replaces it and defaults to true speed.
check("preview_fps is not an input any more",
      "preview_fps" in _schema_inputs(preview.H3PreviewOverride), False)
playback_in = _input(preview.H3PreviewOverride, "playback")
check("playback offers both modes", list(_attr(playback_in, "options", [])),
      [preview.PLAYBACK_TRUE, preview.PLAYBACK_SOURCE])
check("playback defaults to true speed", _attr(playback_in, "default"), preview.PLAYBACK_TRUE)

check("decode offers all three modes",
      list(_attr(_input(preview.H3PreviewOverride, "decode"), "options", [])),
      [preview.DECODE_FAST, preview.DECODE_TAEH3, preview.DECODE_VAE])


# ------------------------------------------------------------------ pixel_frames_from_latent_t
pf = preview.pixel_frames_from_latent_t
check("2 or fewer latent frames is one 5-frame chunk", pf(2), 5)
check("1 latent frame (degenerate) is still one chunk", pf(1), 5)
check("on-grid: 7 latent frames -> 22 pixel frames (17*1+5)", pf(7), 22)
check("on-grid: 12 latent frames -> 39 pixel frames (17*2+5)", pf(12), 39)
check("off-grid latent frame counts still return something positive", pf(9) > 0, True)

# ------------------------------------------------------------------------- _even_indices
# This is what makes "frames spread across the whole video" true rather than a prefix:
# whatever the count, the first and last frame of the clip are always included.
ei = preview._even_indices
check("asking for more than there is returns everything", ei(5, 20), [0, 1, 2, 3, 4])
check("asking for none returns everything", ei(4, 0), [0, 1, 2, 3])
check("a thinned selection spans the whole clip", ei(37, 5), [0, 9, 18, 27, 36])
check("the last frame is always included", ei(100, 7)[-1], 99)
check("the first frame is always included", ei(100, 7)[0], 0)
check("no duplicate indices when the count is dense", len(ei(10, 9)), len(set(ei(10, 9))))

# --------------------------------------------------------------------------- _playback_rate
# true speed: the shown frames are spread across the shot's real duration, so N frames of
# a `pixel_frames`-long clip play at 24*N/pixel_frames.
pr = preview._playback_rate
check("true speed spreads frames over the clip's real length",
      round(pr(preview.PLAYBACK_TRUE, 24, 124), 4), round(24.0 * 24 / 124, 4))
check("true speed with one image per output frame is the native rate",
      pr(preview.PLAYBACK_TRUE, 124, 124), 24.0)
check("source fps ignores the frame count and plays at H3's native rate",
      pr(preview.PLAYBACK_SOURCE, 24, 124), 24.0)
check("a rate is never zero or negative", pr(preview.PLAYBACK_TRUE, 0, 10000) > 0, True)

# ------------------------------------------------------------------------------ throttle_gap
tg = preview.throttle_gap
check("no overhead cap means no gap", tg(5.0, 0), 0.0)
check("zero-cost preview means no gap", tg(0.0, 25), 0.0)
check("25% cap after a 1s preview waits 3s", tg(1.0, 25), 3.0)
check("50% cap after a 1s preview waits 1s", tg(1.0, 50), 1.0)

# ------------------------------------------------------------------------- execute() validation
node = preview.H3PreviewOverride
check_raises("decode='vae (quality)' without a VAE is refused by name",
             lambda: node.execute.__func__(node, model=None, decode=preview.DECODE_VAE),
             "no VAE is")
check_raises("decode='tiny vae (taeh3)' without a tiny_vae is refused by name",
             lambda: node.execute.__func__(node, model=None, decode=preview.DECODE_TAEH3),
             "no tiny VAE is")


failed = [r for r in _results if not r[0]]
for ok, name, got, want in _results:
    print(("ok  " if ok else "FAIL") + " - " + name + ("" if ok else f" (got {got!r}, want {want!r})"))
print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed.")
if failed:
    sys.exit(1)
