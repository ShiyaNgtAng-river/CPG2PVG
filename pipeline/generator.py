"""
PVG generation — convert binned source content into patient-friendly Markdown.

Two classes:

* ``PVGWriter``  – wraps a single LLM call to rewrite one bin.
* ``PVGRunner``  – orchestrates extraction + per-bin generation + stitching.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import GENERATE_MAX_WORKERS
from models import ClassifiedTextBlock
from prompts import STREAM_BIN_PROMPT
from utils.debug import InvokeTracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level writer (one bin → one Markdown section)
# ---------------------------------------------------------------------------

class PVGWriter:
    """Invoke the LLM once to rewrite a single chapter bin."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", STREAM_BIN_PROMPT),
                ("human", "### CHAPTER CONTENT:\n{content}"),
            ])
            | llm
            | StrOutputParser()
        )

    def write_bin(
        self,
        facts: Dict[str, str],
        context_hint: str,
        content: str,
    ) -> str:
        """Return the generated PVG Markdown for one bin."""
        facts_text = json.dumps(facts, ensure_ascii=False, indent=2)
        try:
            return self._chain.invoke({
                "context_hint": context_hint,
                "facts": facts_text,
                "content": content,
            })
        except Exception as exc:
            logger.error("Generation failed for '%s': %s", context_hint, exc)
            return ""


# ---------------------------------------------------------------------------
# Fact-sidebar filter ("SmartTree Dehydration")
# ---------------------------------------------------------------------------

def _filter_facts(
    all_facts: Dict[str, str],
    context_hint: str,
    siblings: List[str],
) -> Dict[str, str]:
    """
    Return a subset of *all_facts* relevant to the current bin.

    Strategy (layered):
      1. **Base** — A (metadata) and E (terms) are always included.
      2. **Smart relevance** — B (disease basics) and C (treatment) are
         included based on keyword matching against the bin's breadcrumb
         path and sibling titles.
      3. **Safety net** — if neither B nor C was selected, include both
         (the bin is probably a generic chapter where context matters).
    """
    filtered: Dict[str, str] = {}

    if "A" in all_facts:
        filtered["A"] = all_facts["A"]

    combined = (context_hint + " " + " ".join(siblings)).lower()

    need_b = any(
        kw in combined
        for kw in ("stage", "staging", "risk", "grade", "diagnosis",
                    "intro", "what is", "about", "treat")
    )
    need_c = any(
        kw in combined
        for kw in ("treat", "surgery", "drug", "therapy", "management",
                    "care", "follow", "support", "effect")
    )

    if need_b and "B" in all_facts:
        filtered["B"] = all_facts["B"]
    if need_c and "C" in all_facts:
        filtered["C"] = all_facts["C"]

    # Safety net
    if "B" not in filtered and "C" not in filtered:
        if "B" in all_facts:
            filtered["B"] = all_facts["B"]
        if "C" in all_facts:
            filtered["C"] = all_facts["C"]

    return filtered


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

class PVGRunner:
    """
    Orchestrate the generation of a full PVG from pre-computed facts and
    chapter bins.

    Usage::

        runner = PVGRunner(generating_llm)
        pvg_markdown = runner.run(bins, all_facts, cache_prefix="bone_cancer")
    """

    def __init__(
        self,
        generating_llm: ChatOpenAI,
    ) -> None:
        self._writer = PVGWriter(generating_llm)

    def run(
        self,
        bins: List[ClassifiedTextBlock],
        all_facts: Dict[str, Any],
        cache_prefix: str = "unknown",
    ) -> str:
        """
        Process bins in parallel and return the stitched PVG Markdown.
        """
        if not bins:
            return ""

        # 1. Prepare tasks — filter keep=False, pre-compute facts
        tasks: List[Tuple[int, str, str, Dict, str]] = []
        skipped = 0
        for i, mission in enumerate(bins):
            meta = mission.doc.metadata
            context_hint = meta.get("context_hint", "")
            siblings = meta.get("siblings", [])

            if not meta.get("keep", True):
                skipped += 1
                logger.info(
                    "[Runner] Skipping bin %d/%d (keep=False): %s",
                    i + 1, len(bins), context_hint,
                )
                continue

            relevant_facts = _filter_facts(all_facts, context_hint, siblings)
            facts_text = json.dumps(relevant_facts, ensure_ascii=False, indent=2)
            prompt_preview = STREAM_BIN_PROMPT.format(
                context_hint=context_hint, facts=facts_text,
            )
            tasks.append((
                i + 1, context_hint, mission.doc.page_content,
                relevant_facts, facts_text, prompt_preview,
            ))

        if skipped:
            logger.info(
                "[Runner] Skipped %d/%d bins (keep=False).", skipped, len(bins),
            )

        if not tasks:
            return "No content generated."

        logger.info(
            "[Runner] Generating %d bins with %d workers...",
            len(tasks), GENERATE_MAX_WORKERS,
        )

        # 2. Fire all generation calls concurrently
        results: List[Optional[str]] = [None] * len(tasks)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=GENERATE_MAX_WORKERS,
        ) as pool:
            future_to_idx = {
                pool.submit(
                    self._writer.write_bin,
                    task[3], task[1], task[2],
                ): idx
                for idx, task in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Generation failed for task %d: %s", idx, exc)
                    results[idx] = ""

        # 3. Log to Invoke_Trace and collect sections (in original order)
        tracer = InvokeTracer(cache_prefix)
        sections: List[str] = []

        for idx, task in enumerate(tasks):
            bin_num, context_hint, content, _, facts_text, prompt_preview = task
            generated = results[idx] or ""

            tracer.log_invoke(
                index=bin_num,
                context=context_hint,
                prompt_preview=prompt_preview,
                payload={"facts": facts_text, "content": content},
                output=generated,
            )
            if generated:
                sections.append(generated)

        if not sections:
            return "No content generated."

        full = "\n\n---\n\n".join(sections)
        return self._final_polish(full)

    @staticmethod
    def _final_polish(text: str) -> str:
        """Remove internal protocol markers."""
        return re.sub(r"### \[TOPOLOGICAL LOCK].*?\n", "", text)
