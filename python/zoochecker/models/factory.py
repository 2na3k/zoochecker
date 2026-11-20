from pathlib import Path

from mlx_lm import load
from mlx_lm.models.qwen3_5 import Model as MLXQwen35
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_config, load_tokenizer
from mlx_vlm.utils import get_model_path
from safetensors import safe_open

from zoochecker.server_args import ServerArgs

from ._base import BaseModel
from .qwen35 import Qwen35


class ModelFactory:
    @staticmethod
    def load(args: ServerArgs) -> tuple[BaseModel, TokenizerWrapper]:
        path = get_model_path(args.model_path or args.model_name)
        config = load_config(path)
        if (
            config.get("vision_config")
            and config.get("model_type") in ("qwen3_5", "qwen3_5_moe")
            and _has_vision_weights(path)
        ):
            from mlx_vlm import load as load_vlm

            upstream, processor = load_vlm(str(path))
            model = Qwen35(upstream, processor=processor)
            codec = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
        else:
            upstream, codec = load(str(path))
            if not isinstance(upstream, MLXQwen35):
                raise ValueError(
                    f"Unsupported checkpoint model type: {config.get('model_type')!r}"
                )
            model = Qwen35(upstream)

        eos = config.get("text_config", {}).get("eos_token_id")
        if eos is not None:
            codec.eos_token_ids |= {eos} if isinstance(eos, int) else set(eos)
        return model, codec


def _has_vision_weights(path: Path) -> bool:
    for file in path.glob("model*.safetensors"):
        with safe_open(file, framework="np") as weights:
            if any(
                key.startswith(
                    ("vision_tower.", "model.visual.", "model.language_model.visual.")
                )
                for key in weights.keys()
            ):
                return True
    return False
