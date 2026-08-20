from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .h3_preview_override import GetH3PreviewFrames, H3PreviewOverride


class H3PreviewOverrideExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [H3PreviewOverride, GetH3PreviewFrames]


async def comfy_entrypoint() -> H3PreviewOverrideExtension:
    return H3PreviewOverrideExtension()


NODE_CLASS_MAPPINGS = {
    "H3PreviewOverride": H3PreviewOverride,
    "GetH3PreviewFrames": GetH3PreviewFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PreviewOverride": "H3 Preview Override",
    "GetH3PreviewFrames": "Get H3 Preview Frames",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
