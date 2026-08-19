from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .minimax_h3_preview_override import MiniMaxH3PreviewPlus


class H3PreviewPlusExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MiniMaxH3PreviewPlus]


async def comfy_entrypoint() -> H3PreviewPlusExtension:
    return H3PreviewPlusExtension()


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PreviewPlusCS": MiniMaxH3PreviewPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PreviewPlusCS": "MiniMax H3 Preview Plus",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
