"""
LLM initialisation helper.

Reads provider config from config.py and API keys from the environment.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from config import DEFAULT_LLM, LLM_PROVIDERS, LLMProvider

logger = logging.getLogger(__name__)


def build_llm(provider: LLMProvider, streaming: bool = False) -> ChatOpenAI:
    """Create a ChatOpenAI instance from a provider descriptor."""
    kwargs = dict(
        base_url=provider.base_url,
        api_key=provider.api_key,      # read from env via the property
        model=provider.model,
        timeout=provider.timeout,
        max_retries=provider.max_retries,
        streaming=streaming,
    )
    # A negative temperature is our sentinel for "omit temperature" (reasoning
    # models like gpt-5-mini determine their own temperature).
    if provider.temperature >= 0:
        kwargs["temperature"] = provider.temperature
    return ChatOpenAI(**kwargs)


def get_llm(name: str = DEFAULT_LLM) -> ChatOpenAI:
    """
    Return a ready-to-use ChatOpenAI for the given provider name.

    Falls back to the default provider when the requested one is unavailable.
    """
    if name not in LLM_PROVIDERS:
        available = ", ".join(LLM_PROVIDERS)
        raise ValueError(
            f"Unknown LLM provider '{name}'. Available: {available}"
        )

    try:
        llm = build_llm(LLM_PROVIDERS[name])
        logger.info("Initialised LLM: %s (%s)", name, LLM_PROVIDERS[name].model)
        return llm
    except Exception:
        if name != DEFAULT_LLM:
            logger.warning(
                "Failed to initialise '%s', falling back to '%s'.",
                name, DEFAULT_LLM,
            )
            return build_llm(LLM_PROVIDERS[DEFAULT_LLM])
        raise
