"""Offline checks for minimax_h3_preview_override.py — the half that needs ComfyUI on the
path (comfy.patcher_extension, comfy.utils, latent_preview, server, comfy_api.latest,
protocol, folder_paths, comfy.taesd.taesd/taehv). Run it with the same interpreter
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
    return sys.modules[name + ".minimax_h3_preview_override"]


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
    schema = node_cls.define_schema()
    return [i.id for i in schema.inputs]


node_cls = preview.MiniMaxH3PreviewPlus
params = inspect.signature(node_cls.execute.__func__).parameters
accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
missing = [] if accepts_kwargs else [
    name for name in _schema_inputs(node_cls) if name not in params]
check("MiniMaxH3PreviewPlus: every schema input has an execute() parameter", missing, [])

# show_vae_input was added last (after tiny_vae) and is a widget, not a socket that older
# saved workflows already carry a value for — it has to stay last, or a workflow saved
# before it existed hands its saved values to the wrong widgets.
check("show_vae_input is the last widget (positional-serialisation trap)",
      _schema_inputs(node_cls)[-1], "show_vae_input")
decode_input = next(i for i in node_cls.define_schema().inputs if i.id == "decode")
check("decode offers all three modes",
      list(decode_input.options),
      [preview.DECODE_FAST, preview.DECODE_TAEH3, preview.DECODE_VAE])


# ------------------------------------------------------------------ pixel_frames_from_latent_t
pf = preview.pixel_frames_from_latent_t
check("2 or fewer latent frames is one 5-frame chunk", pf(2), 5)
check("1 latent frame (degenerate) is still one chunk", pf(1), 5)
check("on-grid: 7 latent frames -> 22 pixel frames (17*1+5)", pf(7), 22)
check("on-grid: 12 latent frames -> 39 pixel frames (17*2+5)", pf(12), 39)
check("off-grid latent frame counts still return something positive", pf(9) > 0, True)

# ------------------------------------------------------------------------------ throttle_gap
tg = preview.throttle_gap
check("no overhead cap means no gap", tg(5.0, 0), 0.0)
check("zero-cost preview means no gap", tg(0.0, 25), 0.0)
check("25% cap after a 1s preview waits 3s", tg(1.0, 25), 3.0)
check("50% cap after a 1s preview waits 1s", tg(1.0, 50), 1.0)

# ------------------------------------------------------------------------- execute() validation
check_raises("decode='vae (quality)' without a VAE is refused by name",
             lambda: node_cls.execute.__func__(node_cls, model=None, decode="vae (quality)"),
             "no VAE is")
check_raises("decode='tiny vae (taeh3)' without a tiny_vae is refused by name",
             lambda: node_cls.execute.__func__(node_cls, model=None, decode="tiny vae (taeh3)"),
             "no tiny")


failed = [r for r in _results if not r[0]]
for ok, name, got, want in _results:
    print(("ok  " if ok else "FAIL") + " - " + name + ("" if ok else f" (got {got!r}, want {want!r})"))
print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed.")
if failed:
    sys.exit(1)
