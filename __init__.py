from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .minimax_h3_preview_override import MiniMaxH3PreviewOverride


class H3PreviewOverrideExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MiniMaxH3PreviewOverride]


async def comfy_entrypoint() -> H3PreviewOverrideExtension:
    return H3PreviewOverrideExtension()


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PreviewOverrideCS": MiniMaxH3PreviewOverride,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PreviewOverrideCS": "MiniMax H3 Preview Override",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
