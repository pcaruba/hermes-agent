"""Minimax image generation backend.

Wraps the Minimax ``/v1/image_generation`` endpoint as an
:class:`ImageGenProvider` implementation. Supports the ``image-01`` model
with the eight aspect ratios the Minimax API exposes.

Selection precedence for the model (first hit wins):

1. ``MINIMAX_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.minimax.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our model IDs)
4. :data:`DEFAULT_MODEL` — ``image-01``

Output is downloaded from the hosted ``image_urls`` returned by the API and
saved under ``$HERMES_HOME/cache/images/`` — same pattern as the ``xai``
plugin. The URLs are temporary, so we materialise the bytes locally at
tool-completion time so downstream consumers (Telegram ``send_photo``,
browser fetch) never hit a dead link.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default API host. Matches the upstream Minimax SDK / MCP default. Users
# with a regional deployment (e.g. CN endpoint) can override via the
# ``MINIMAX_API_HOST`` env var, mirroring the existing MCP server convention.
DEFAULT_API_HOST = "https://api.minimax.io"

# ``MM-API-Source`` header is required alongside the bearer token — without
# it the API returns ``status_code: 1004 login fail`` even with a valid key.
# Identified by inspecting the open-source ``minimax-mcp`` client.
MM_API_SOURCE = "Minimax-MCP"

IMAGE_GENERATION_ENDPOINT = "/v1/image_generation"


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

API_MODEL = "image-01"

_MODELS: Dict[str, Dict[str, Any]] = {
    "image-01": {
        "display": "Minimax image-01",
        "speed": "~6-15s",
        "strengths": "General-purpose text-to-image with prompt optimizer",
    },
}

DEFAULT_MODEL = "image-01"

# Aspect ratios supported by the Minimax API. ``resolve_aspect_ratio()`` will
# clamp any of the three canonical agent values (``landscape`` / ``square`` /
# ``portrait``) onto the closest Minimax enum.
_VALID_MINIMAX_RATIOS = {
    "1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9",
}

_ASPECT_RATIO_MAP: Dict[str, str] = {
    # Agent canonical -> Minimax canonical. ``landscape`` is the default
    # (matches ImageGenProvider.DEFAULT_ASPECT_RATIO).
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


def _minimax_aspect_ratio(canonical: str) -> str:
    """Translate an agent canonical aspect ratio into a Minimax enum value.

    Falls back to ``16:9`` for any value we don't recognise. The Minimax API
    rejects unknown ratios with a 4xx error, so we always translate rather
    than pass through.
    """
    return _ASPECT_RATIO_MAP.get(canonical, "16:9")


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001 — defensive; picker must not crash
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``.

    Mirrors the precedence used by the upstream ``openai`` plugin so the
    picker UX is consistent across backends.
    """
    env_override = os.environ.get("MINIMAX_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_gen_config()
    minimax_cfg = cfg.get("minimax") if isinstance(cfg.get("minimax"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(minimax_cfg, dict):
        value = minimax_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_api_host() -> str:
    """Return the Minimax API host, honouring ``MINIMAX_API_HOST`` if set."""
    return os.environ.get("MINIMAX_API_HOST", DEFAULT_API_HOST).rstrip("/")


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MinimaxImageGenProvider(ImageGenProvider):
    """Minimax ``/v1/image_generation`` backend — ``image-01`` model."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "Minimax"

    def is_available(self) -> bool:
        # Provider requires an API key. The python ``requests`` package is a
        # baseline Hermes dependency so we don't need to probe for it.
        return bool(os.environ.get("MINIMAX_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Minimax",
            "badge": "paid",
            "tag": "Minimax image-01 text-to-image via the /v1/image_generation endpoint",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "Minimax API key",
                    "url": "https://platform.minimaxi.com/user-center/basic-information",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # Minimax's image_generation endpoint is text-to-image only. The
        # platform offers separate edit endpoints that this provider does
        # not wrap yet — return text-only so the dynamic tool schema is
        # honest about what the backend accepts.
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        canonical = resolve_aspect_ratio(aspect_ratio)
        minimax_ratio = _minimax_aspect_ratio(canonical)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="minimax",
                aspect_ratio=canonical,
            )

        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "MINIMAX_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → Minimax to configure, or `hermes setup` "
                    "to add the key. Get one at "
                    "https://platform.minimaxi.com/user-center/basic-information"
                ),
                error_type="auth_required",
                provider="minimax",
                aspect_ratio=canonical,
            )

        # The plugin does not wrap an edit endpoint yet. Refuse cleanly so
        # the agent gets a useful error rather than a confusing 4xx from
        # the upstream API.
        if image_url or reference_image_urls:
            return error_response(
                error=(
                    "The Minimax plugin is text-to-image only. Image-to-image "
                    "editing is not supported by /v1/image_generation — pick "
                    "a backend that advertises image modality (e.g. FAL.ai, "
                    "OpenAI, xAI, Krea)."
                ),
                error_type="invalid_argument",
                provider="minimax",
                aspect_ratio=canonical,
            )

        model_id, _meta = _resolve_model()
        host = _resolve_api_host()
        url = f"{host}{IMAGE_GENERATION_ENDPOINT}"

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "aspect_ratio": minimax_ratio,
            "n": 1,
            "prompt_optimizer": True,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "MM-API-Source": MM_API_SOURCE,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=120,
            )
        except requests.RequestException as exc:
            logger.debug("Minimax image generation request failed", exc_info=True)
            return error_response(
                error=f"Minimax request failed: {exc}",
                error_type="network_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=canonical,
            )

        # Parse the JSON envelope first — Minimax returns 200 with an error
        # body for auth/quota/exhaustion cases.
        try:
            data = response.json()
        except ValueError:
            return error_response(
                error=(
                    f"Minimax returned a non-JSON response "
                    f"(HTTP {response.status_code}): {response.text[:200]!r}"
                ),
                error_type="invalid_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=canonical,
            )

        base_resp = data.get("base_resp") if isinstance(data, dict) else None
        if isinstance(base_resp, dict) and base_resp.get("status_code", 0) != 0:
            status_code = base_resp.get("status_code")
            status_msg = base_resp.get("status_msg", "unknown error")
            # 1004 = auth/bearer rejected, 2038 = real-name verification missing.
            error_type = (
                "auth_required" if status_code in (1004, 1001) else "api_error"
            )
            return error_response(
                error=(
                    f"Minimax API error {status_code}: {status_msg}. "
                    f"Verify MINIMAX_API_KEY is set, has image-generation "
                    f"scope, and that real-name verification is complete at "
                    f"https://platform.minimaxi.com/user-center/basic-information"
                ),
                error_type=error_type,
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=canonical,
            )

        # Success — extract hosted URLs and materialise locally.
        image_urls: List[str] = []
        if isinstance(data, dict):
            inner = data.get("data") or {}
            if isinstance(inner, dict):
                urls = inner.get("image_urls") or []
                if isinstance(urls, list):
                    image_urls = [u for u in urls if isinstance(u, str)]

        if not image_urls:
            return error_response(
                error="Minimax returned no image_urls in the response",
                error_type="empty_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=canonical,
            )

        first_url = image_urls[0]
        try:
            saved_path = save_url_image(first_url, prefix=f"minimax_{model_id}")
            image_ref = str(saved_path)
        except Exception as exc:  # noqa: BLE001
            # Fall back to the bare URL with a clear warning — same defensive
            # pattern as the xai plugin.
            logger.warning(
                "Minimax image URL %s could not be cached (%s); falling back to bare URL.",
                first_url,
                exc,
            )
            image_ref = first_url

        extra: Dict[str, Any] = {
            "size": minimax_ratio,
            "aspect_ratio_minimax": minimax_ratio,
            "n": 1,
            "prompt_optimizer": True,
        }
        if len(image_urls) > 1:
            extra["additional_urls"] = image_urls[1:]

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=canonical,
            provider="minimax",
            modality="text",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MinimaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MinimaxImageGenProvider())