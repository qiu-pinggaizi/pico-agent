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


def _vision_analyze_openai(
    b64_data: str,
    media_type: str,
    prompt: str,
    vision_model: str,
    cfg: Any,
) -> str:
    """Vision analysis using OpenAI-compatible API."""
    from openai import OpenAI

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

    logger.info("vision_analyze (openai): %d chars description", len(description))

    return _success({
        "description": description,
        "model_used": vision_model,
        "usage": usage,
    })


def _vision_analyze_anthropic(
    b64_data: str,
    media_type: str,
    prompt: str,
    vision_model: str,
    cfg: Any,
) -> str:
    """Vision analysis using Anthropic native API."""
    import anthropic

    api_key = cfg.api_key or os.environ.get("PICO_API_KEY", "")
    base_url = cfg.base_url or None

    client = anthropic.Anthropic(api_key=api_key, base_url=base_url if base_url else None)

    response = client.messages.create(
        model=vision_model,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    description = ""
    for block in response.content:
        if block.type == "text":
            description += block.text

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

    logger.info("vision_analyze (anthropic): %d chars description", len(description))

    return _success({
        "description": description,
        "model_used": vision_model,
        "usage": usage,
    })


def vision_analyze(
    image: str,
    prompt: str = "Describe this image in detail.",
    model: str | None = None,
) -> str:
    """Analyze an image using a multimodal LLM."""
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
        from pico.config import get_config

        cfg = get_config()
        provider = cfg.provider.lower()

        if provider == "anthropic":
            return _vision_analyze_anthropic(b64_data, media_type, prompt, vision_model, cfg)
        else:
            return _vision_analyze_openai(b64_data, media_type, prompt, vision_model, cfg)

    except ImportError as e:
        pkg = "anthropic" if "anthropic" in str(e) else "openai"
        return _error(f"{pkg} package not installed — cannot run vision analysis: {e}")
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
