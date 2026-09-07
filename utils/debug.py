"""
Debug snapshot helpers.

All file-system I/O for debugging is isolated here so the core pipeline
stays pure (no side-effects beyond LLM calls).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from config import DEBUG_SNAPSHOTS_DIR

logger = logging.getLogger(__name__)


def _snapshot_dir(prefix: str) -> str:
    """返回该 prefix 专属的快照子目录路径，并确保目录存在。"""
    directory = os.path.join(DEBUG_SNAPSHOTS_DIR, prefix)
    os.makedirs(directory, exist_ok=True)
    return directory


def save_json(prefix: str, suffix: str, data: Any) -> str:
    """Write *data* as pretty-printed JSON and return the file path."""
    path = os.path.join(_snapshot_dir(prefix), f"{suffix}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info("Debug snapshot saved: %s", path)
    return path


def save_markdown(prefix: str, suffix: str, text: str) -> str:
    """Write *text* as a Markdown file and return the file path."""
    path = os.path.join(_snapshot_dir(prefix), f"{suffix}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Debug snapshot saved: %s", path)
    return path


def load_json(prefix: str, suffix: str) -> Any:
    """Load a previously-saved JSON snapshot.  Returns None on failure."""
    path = os.path.join(DEBUG_SNAPSHOTS_DIR, prefix, f"{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Invoke-trace logger (records every LLM call payload + output)
# ---------------------------------------------------------------------------

class InvokeTracer:
    """
    Append-only Markdown log that records the full payload sent to the LLM
    and the raw output received, one entry per generate call.

    Usage::

        tracer = InvokeTracer("bone_cancer")
        tracer.log_invoke(index=1, context="Treatment > Surgery",
                          prompt_preview=prompt_text, payload=payload_dict,
                          output=llm_output)
    """

    def __init__(self, prefix: str) -> None:
        self.path = os.path.join(_snapshot_dir(prefix), "Invoke_Trace.md")
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(f"# Invoke Trace Log: {prefix}\n\n")

    def log_invoke(
        self,
        index: int,
        context: str,
        prompt_preview: str,
        payload: Dict[str, Any],
        output: str,
    ) -> None:
        facts_text = payload.get("facts", "")
        content_text = payload.get("content", "")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"## Invoke #{index}: {context}\n")
            fh.write(f"**Facts Token Est**: {len(facts_text) // 4}\n")
            fh.write(f"**Content Token Est**: {len(content_text) // 4}\n\n")
            fh.write("### System Prompt\n```markdown\n")
            fh.write(prompt_preview)
            fh.write("\n```\n\n")
            fh.write("### CPG Source Content (Human Message)\n```markdown\n")
            fh.write(content_text)
            fh.write("\n```\n\n")
            fh.write("### LLM Output\n")
            fh.write(f"**Chars**: {len(output)}  |  **Token Est**: {len(output) // 4}\n\n")
            fh.write("```markdown\n")
            fh.write(output)
            fh.write("\n```\n\n---\n\n")
