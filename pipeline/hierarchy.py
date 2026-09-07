"""
TitleHierarchyAgent — classify headings into semantic labels (A–E) and
reconstruct the parent-child tree via batched LLM calls with global
accumulation.

This is the first LLM-calling stage of the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from config import HIERARCHY_BATCH_SIZE, HIERARCHY_RECENT_NON_ROOTS
from models import HeadingCandidate
from prompts import LABEL_KEYWORDS, TITLE_HIERARCHY_SYSTEM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal model for one classified title
# ---------------------------------------------------------------------------

class TitleItem(BaseModel):
    id: int = Field(..., description="Heading chunk ID")
    parent_id: Optional[int] = Field(None, description="Logical parent ID")
    label: str = Field(..., description="Semantic label A–E")
    title: str = Field("", description="Cleaned heading text")
    keep: bool = Field(True, description="Whether to keep this heading")


# ---------------------------------------------------------------------------
# Keyword-based fallback
# ---------------------------------------------------------------------------

def infer_label_from_text(text: str, default: str = "B") -> str:
    """Heuristic label assignment using keyword matching (no LLM)."""
    lower = text.lower()
    for label, keywords in LABEL_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return label
    return default


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TitleHierarchyAgent:
    """
    Classify heading candidates into A–E labels and assign parent IDs.

    The agent processes headings in batches of ``HIERARCHY_BATCH_SIZE``.
    A *global navigation map* accumulates across batches so that later batches
    can reference root chapters discovered earlier — this solves the "batch
    boundary" problem where a sub-heading appears in a different batch from
    its parent.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", TITLE_HIERARCHY_SYSTEM),
            ("human",
             "### CURRENT BATCH HEADINGS:\n{candidates}\n\n"
             "Reconstruct the logical hierarchy."),
        ])
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.chain = self.prompt | self.llm_json | StrOutputParser()

    # ----- public API -----

    def classify(
        self, candidates: List[HeadingCandidate],
    ) -> Dict[int, TitleItem]:
        """
        Return ``{heading_id: TitleItem}`` for every candidate.

        Headings that fail LLM classification fall back to keyword heuristics.
        """
        if not candidates:
            return {}

        results: Dict[int, TitleItem] = {}
        lookup = {c.heading_id: c for c in candidates}
        history: List[Dict[str, Any]] = []

        for start in range(0, len(candidates), HIERARCHY_BATCH_SIZE):
            batch = candidates[start : start + HIERARCHY_BATCH_SIZE]
            payload = [
                {"id": c.heading_id, "text": c.text, "preview": c.preview}
                for c in batch
            ]

            # Build the navigation map from accumulated history
            roots = [h for h in history if h.get("is_root")]
            recent = [h for h in history if not h.get("is_root")][
                -HIERARCHY_RECENT_NON_ROOTS:
            ]
            nav_items = sorted(roots + recent, key=lambda x: x["id"])
            history_str = "\n".join(
                f'[ID: {h["id"]}] {h["text"]} (Label: {h["label"]})'
                for h in nav_items
            ) or "None (First batch)"

            try:
                items = self._invoke(payload, history_str)
                for entry in items:
                    hid = entry.get("id")
                    if hid not in lookup:
                        continue
                    entry.setdefault("title", lookup[hid].text)
                    item = TitleItem(**entry)
                    results[hid] = item

                    # Register anchors for next batches
                    is_root = item.parent_id is None
                    if is_root or item.label in ("C", "D"):
                        history.append({
                            "id": hid,
                            "text": item.title,
                            "label": item.label,
                            "is_root": is_root,
                        })
            except Exception as exc:
                logger.warning("Hierarchy batch failed, using fallback: %s", exc)

        # Fallback for any candidates the LLM missed
        for c in candidates:
            if c.heading_id not in results:
                results[c.heading_id] = TitleItem(
                    id=c.heading_id,
                    parent_id=None,
                    label=infer_label_from_text(c.text),
                    title=c.text,
                    keep=True,
                )
        return results

    # ----- internals -----

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _invoke(
        self, payload: List[Dict[str, Any]], history_map: str,
    ) -> List[Dict[str, Any]]:
        raw = self.chain.invoke({
            "candidates": json.dumps(payload, ensure_ascii=False),
            "history_map": history_map,
        })
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned).get("items", [])
