"""
SmartSplitter — split a Markdown document into structural chunks.

This is a deterministic, code-only stage (no LLM calls).  It produces a list
of ``RawChunk`` objects, each annotated with whether it looks like a heading.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from config import SPLITTER_CHUNK_SIZE_LIMIT
from models import RawChunk


class SmartSplitter:
    """
    Split raw Markdown into chunks by detecting heading boundaries.

    Heading detection rules (applied in order):
      1. Standard Markdown ``# … ######``
      2. ALL-CAPS lines (heuristic for implied headings)
      3. Short lines ending with ``:``

    Lines that look like list items or HTML table fragments are *never*
    treated as headings, even if they match one of the rules above.
    """

    def __init__(self, chunk_size_limit: int = SPLITTER_CHUNK_SIZE_LIMIT) -> None:
        self.chunk_size_limit = chunk_size_limit
        self._md_header = re.compile(r"^(#{1,6})\s+(.+)$")
        self._implied_header = re.compile(r"^([A-Z\s\d\W]{5,80})$")
        self._list_prefix = re.compile(r"^(?:[-*•·]\s+|\d{1,3}[.)]\s+)")
        self._html_table = re.compile(r"</?(?:table|tr|td|th)\b", re.IGNORECASE)

    # ----- public API -----

    def split_text(self, text: str) -> List[RawChunk]:
        """Return a list of ``RawChunk`` preserving reading order."""
        lines = text.split("\n")
        chunks: List[RawChunk] = []
        current_lines: List[str] = []
        chunk_id = 0
        in_table = False
        in_html_table = False
        starts_with_header = False
        header_level = 0

        def _commit() -> None:
            nonlocal chunk_id, starts_with_header, header_level
            if not current_lines:
                return
            content = "\n".join(current_lines).strip()
            if not content:
                return

            has_follow = self._has_body_text(current_lines, starts_with_header)
            chunks.append(RawChunk(
                id=chunk_id,
                content=content,
                is_potential_header=starts_with_header,
                header_level=header_level,
                represent_text=content[:100],
                original_idx=chunk_id,
                has_following_text=has_follow,
            ))
            chunk_id += 1
            current_lines.clear()
            starts_with_header = False
            header_level = 0

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            # HTML table state tracking
            if "<table" in lower:
                in_html_table = True
            end_html_table = "</table" in lower

            # Markdown table detection
            if stripped.count("|") >= 2:
                in_table = True
            elif not stripped and in_table:
                in_table = False

            # Heading detection (suppressed inside tables)
            is_h, lvl = (
                (False, 0)
                if (in_table or in_html_table)
                else self._is_header(stripped)
            )

            if is_h:
                _commit()
                starts_with_header = True
                header_level = lvl

            current_lines.append(line)

            if end_html_table:
                in_html_table = False

        _commit()
        return chunks

    def get_smart_preview(
        self, text: str, min_words: int = 10, max_words: int = 20,
    ) -> str:
        """Return a brief preview of the first sentence(s) in *text*."""
        if not text:
            return ""
        cleaned = re.sub(r"!\[.*?]\(.*?\)", "", text).strip()
        words = cleaned.split()
        if not words:
            return ""
        first_sent = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
        first_words = first_sent.split()
        count = len(first_words)
        if min_words <= count <= max_words:
            return " ".join(first_words)
        if count < min_words:
            return " ".join(words[:max_words])
        return " ".join(first_words[:max_words])

    # ----- internals -----

    def _is_header(self, stripped: str) -> Tuple[bool, int]:
        if not stripped:
            return False, 0
        if self._list_prefix.match(stripped):
            return False, 0
        if self._html_table.search(stripped):
            return False, 0

        m = self._md_header.match(stripped)
        if m:
            return True, len(m.group(1))

        if self._implied_header.match(stripped) and not stripped.endswith("."):
            return True, 0

        if stripped.endswith(":") and len(stripped.split()) <= 7:
            return True, 0

        return False, 0

    @staticmethod
    def _has_body_text(lines: List[str], starts_with_header: bool) -> bool:
        """Return True if the chunk contains body text beyond the heading."""
        if not starts_with_header:
            return True
        header_lines = 0
        for line in lines:
            s = line.strip()
            if not s:
                header_lines += 1
                break
            if re.match(r"^#{1,6}\s+", s):
                header_lines += 1
                continue
            if len(s) < 120 and not re.search(r"[.!?]$", s):
                header_lines += 1
            else:
                break
        body = "\n".join(lines[header_lines:]).strip()
        return bool(body)
