"""
Centralized configuration for the CPG2PVG pipeline.

All tunable parameters live here so you never need to grep through source code
to change a number.  API keys are loaded from a .env file (see .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM provider definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMProvider:
    """Immutable descriptor for one LLM endpoint."""
    base_url: str
    model: str
    api_key_env: str          # name of the env-var that holds the secret
    temperature: float = 0.0
    timeout: int = 300
    max_retries: int = 2

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise EnvironmentError(
                f"Missing environment variable: {self.api_key_env}. "
                f"Set it in your .env file."
            )
        return key


LLM_PROVIDERS: Dict[str, LLMProvider] = {
    "deepseek": LLMProvider(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "gpt-4o-mini": LLMProvider(
        base_url="https://api.ai-gaochao.cn/v1",
        model="gpt-4o-mini",
        api_key_env="GAOCHAO_API_KEY",
    ),
    "gpt-5-mini": LLMProvider(
        base_url="https://api.ai-gaochao.cn/v1",
        model="gpt-5-mini",
        api_key_env="GAOCHAO_API_KEY",
        temperature=-1.0,  # sentinel: means "omit temperature" (reasoning model)
    ),
    "claude-small": LLMProvider(
        base_url="https://api.ai-gaochao.cn/v1",
        model="claude-3-5-haiku-20241022",
        api_key_env="GAOCHAO_API_KEY",
    ),
    "qwen-long": LLMProvider(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-long",
        api_key_env="QWEN_API_KEY",
    ),
}

DEFAULT_LLM = "deepseek"

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------

SPLITTER_CHUNK_SIZE_LIMIT = 2000

HIERARCHY_BATCH_SIZE = 80
HIERARCHY_RECENT_NON_ROOTS = 20

EXTRACTOR_MAX_WORKERS = 8
EXTRACT_PASSAGE_TOKEN_LIMIT = 50000

GENERATE_MAX_WORKERS = 4

TREE_BIN_TOKEN_LIMIT = 12000

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8012

# ---------------------------------------------------------------------------
# Debug / IO
# ---------------------------------------------------------------------------

DEBUG_SNAPSHOTS_DIR = "debug_snapshots"
