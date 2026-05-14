"""Vision analysis tool — send images to a multimodal LLM for description.

Supports both base64-encoded images and file paths.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from pico.tools.registry import ToolRegistry
from pico.tools.utils import _error, _success

logger = logging.getLogger(__name__)


def _load_image_base64(image: str) -> tuple[str, str]:
    """Load an image and return (base64_data, media_type).

    Args:
        image: Either a file path or a base64-encoded string.

    Returns:
        Tuple of (base64_encoded_bytes, media_type).
    """
    # If it looks like a file path, read the file
    if Path(image).expanduser().exists():
        path = Path(image).expanduser().resolve()
        data = path.read_bytes()
        suffix = path.suffix.lower()
        media_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        media_type = media_map.get(suffix, "image/png")
        return base64.b64encode(data).decode("ascii"), media_type

    # Otherwise treat as base64
    # Detect media type from magic bytes
    try:
        raw = base64.b64decode(image[:32])
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return image, "image/png"
        elif raw[:2] == b"\xff\xd8":
            return image, "image/jpeg"
        elif raw[:4] == b"GIF8":
            return image, "image/gif"
        elif raw[:4] == b"RIFF":
            return image, "image/webp"
    except Exception:
        pass

    return image, "image/png"


def vision_analyze(
    image: str,
    prompt: str = "Describe this image in detail.",
    model: str | None = None,
) -> str:
    """Analyze an image using a multimodal LLM.

    Sends the image (as base64) with a text prompt to the configured
    vision-capable model and returns the description.

    Args:
        image: Path to an image file or base64-encoded image data.
        prompt: Text prompt for what to analyze (default: "Describe this image in detail.").
        model: Optional model override (e.g. "gpt-4o"). If not provided,
               uses the configured model.

    Returns:
        JSON with keys: success, description, model_used.
    """
    try:
        b64_data, media_type = _load_image_base64(image)
    except Exception as e:
        return _error(f"Failed to load image: {e}")

    # Default vision model
    vision_model = model
    if not vision_model:
        try:
            from pico.config import get_config
            cfg = get_config()
            vision_model = cfg.model
        except Exception:
            vision_model = "gpt-4o-mini"

    try:
        from openai import OpenAI
        from pico.config import get_config

        cfg = get_config()
        api_key = cfg.api_key or os.environ.get("PICO_API_KEY", "")
        base_url = cfg.base_url or os.environ.get("PICO_BASE_URL", "https://api.openai.com/v1")

        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64_data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )

        description = response.choices[0].message.content or ""

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info("vision_analyze: %d chars description", len(description))

        return _success({
            "description": description,
            "model_used": vision_model,
            "usage": usage,
        })

    except ImportError:
        return _error("openai package not installed — cannot run vision analysis")
    except Exception as e:
        logger.error("vision_analyze failed: %s", e)
        return _error(f"Vision analysis failed: {e}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _check_available() -> bool:
    """Check if openai is installed (required for vision)."""
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def register(registry: ToolRegistry) -> None:
    """Register vision tools with the given registry."""
    registry.register(
        name="vision_analyze",
        toolset="vision",
        description=(
            "Analyze an image using a multimodal LLM. "
            "Provide an image path or base64-encoded image data, and a text prompt "
            "describing what to analyze."
        ),
        parameters={
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Path to an image file or base64-encoded image data."},
                "prompt": {
                    "type": "string",
                    "description": "Text prompt for analysis (default: 'Describe this image in detail.').",
                    "default": "Describe this image in detail.",
                },
                "model": {
                    "type": "string",
                    "description": "Vision model to use (e.g. 'gpt-4o'). Defaults to gpt-4o-mini.",
                },
            },
            "required": ["image"],
        },
        handler=vision_analyze,
        check_fn=_check_available,
    )
