"""
ExtractAgent — concurrent QA-based fact extraction.

For each semantic category (A–E) the agent fires a set of questions (from
``CHECK_LIST``) against the relevant text slices and collects structured
answers.  The result is a "facts sidebar" used by the generation stage.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import EXTRACT_PASSAGE_TOKEN_LIMIT, EXTRACTOR_MAX_WORKERS
from models import ExtractOutput, SerializableDocument
from prompts import CHECK_LIST, QA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level QA agent (one question → one answer)
# ---------------------------------------------------------------------------

class _QAAgent:
    """Ask a single question against a text passage."""

    def __init__(self, llm: ChatOpenAI) -> None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nPassage:\n{text}"),
        ])
        self._chain = prompt | llm | StrOutputParser()

    def ask(self, text: str, question: str) -> Optional[str]:
        try:
            answer = self._chain.invoke({"question": question, "text": text})
            if answer and "NOTFOUND" not in answer:
                return answer.strip()
        except Exception as exc:
            logger.debug("QA call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# High-level extraction orchestrator
# ---------------------------------------------------------------------------

class ExtractAgent:
    """
    Extract structured facts from raw chunks grouped by semantic label.

    Usage::

        agent = ExtractAgent(llm)
        output: ExtractOutput = agent.extract(raw_chunks, metadata_map)
        # output.A_extraction -> {"Title of the Medical Guideline": "...", ...}
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        max_workers: int = EXTRACTOR_MAX_WORKERS,
    ) -> None:
        self._qa = _QAAgent(llm)
        self._max_workers = max_workers

    def extract(
        self,
        raw_chunks: List[Any],
        metadata_map: Dict[int, Any],
    ) -> ExtractOutput:
        """Run extraction over all chunks and return categorised facts."""
        start = time.time()

        # Group chunk text by label
        label_texts: Dict[str, List[str]] = defaultdict(list)
        for chunk in raw_chunks:
            meta = metadata_map.get(chunk.id, {})
            label = meta.get("label", "B")
            path = meta.get("path", "General")
            label_texts[label].append(f"### [PATH: {path}]\n{chunk.content}")

        # Build per-label joined text
        own_text: Dict[str, str] = {
            lbl: "\n\n".join(parts) for lbl, parts in label_texts.items()
        }

        # Clinical pool (B+C+D) for B and C questions — solves B=0 problem
        clinical_parts = (
            label_texts.get("B", [])
            + label_texts.get("C", [])
            + label_texts.get("D", [])
        )
        clinical_pool = "\n\n".join(clinical_parts)
        pool_tokens = len(clinical_pool) // 4
        logger.info(
            "Clinical pool: %d parts, ~%d tokens (limit %d).",
            len(clinical_parts), pool_tokens, EXTRACT_PASSAGE_TOKEN_LIMIT,
        )

        results: Dict[str, Dict[str, str]] = {lbl: {} for lbl in "ABCDE"}

        for label in "ABCDE":
            questions = CHECK_LIST.get(label)
            if not questions:
                continue

            if label in ("B", "C"):
                target = clinical_pool
            else:
                target = own_text.get(label, "")

            if not target.strip():
                continue

            passages = self._split_passage(target)
            if len(passages) == 1:
                facts = self._ask_all(passages[0], questions)
            else:
                logger.info(
                    "Label %s: passage split into %d segments.", label, len(passages),
                )
                facts = self._ask_all_chunked(passages, questions)
            results[label].update(facts)

        output = ExtractOutput(
            A_extraction=results["A"],
            B_extraction=results["B"],
            C_extraction=results["C"],
            D_extraction=results["D"],
            E_extraction=results["E"],
        )
        elapsed = round(time.time() - start, 2)
        summary = {lbl: len(getattr(output, f"{lbl}_extraction")) for lbl in "ABCDE"}
        logger.info("Extraction completed in %.1fs: %s", elapsed, summary)
        return output

    # ----- internals -----

    @staticmethod
    def _split_passage(
        text: str,
        limit: int = EXTRACT_PASSAGE_TOKEN_LIMIT,
    ) -> List[str]:
        """Split *text* into segments that each fit within *limit* tokens.

        Splits on ``### [PATH:`` boundaries so that individual chunks stay
        intact.  Returns a list with one or more segments.
        """
        if len(text) // 4 <= limit:
            return [text]

        parts = text.split("### [PATH:")
        segments: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for i, part in enumerate(parts):
            piece = part if i == 0 else f"### [PATH:{part}"
            piece_tokens = len(piece) // 4
            if current_tokens + piece_tokens > limit and current:
                segments.append("".join(current))
                current = [piece]
                current_tokens = piece_tokens
            else:
                current.append(piece)
                current_tokens += piece_tokens

        if current:
            segments.append("".join(current))
        return segments or [text]

    def _ask_all(
        self, full_text: str, questions: Dict[str, str],
    ) -> Dict[str, str]:
        """Fire all *questions* concurrently against *full_text*."""
        merged: Dict[str, List[str]] = defaultdict(list)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
        ) as pool:
            future_to_key = {
                pool.submit(self._qa.ask, full_text, q_text): q_key
                for q_key, q_text in questions.items()
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    answer = future.result()
                    if answer:
                        merged[key].append(answer)
                except Exception as exc:
                    logger.warning("QA future failed for '%s': %s", key, exc)

        return {
            k: "\n\n".join(dict.fromkeys(v))
            for k, v in merged.items()
        }

    def _ask_all_chunked(
        self, passages: List[str], questions: Dict[str, str],
    ) -> Dict[str, str]:
        """Ask *questions* against each segment in *passages*, merge answers."""
        merged: Dict[str, List[str]] = defaultdict(list)
        for passage in passages:
            facts = self._ask_all(passage, questions)
            for key, answer in facts.items():
                if answer:
                    merged[key].append(answer)
        return {
            k: "\n\n".join(dict.fromkeys(v))
            for k, v in merged.items()
        }
