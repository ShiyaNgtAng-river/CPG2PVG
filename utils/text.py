"""
Low-level text utilities: deduplication, overlap prevention, cleaning helpers.

These are pure functions with no LLM or I/O dependencies.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Any, List, Set


# ---------------------------------------------------------------------------
# RAG evaluation JSON detection
# ---------------------------------------------------------------------------

RAG_EVAL_KEYWORDS: Set[str] = {
    "has_hallucinations", "hallucination_count", "hallucination_details",
    "verification_passed", "specificity_passed", "specificity_score",
    "vague_phrases_count", "missing_entities", "vague_phrases",
    "improvement_suggestions", "total_statements", "correct_statements",
    "hallucinated_statements", "contradictory_statements", "hallucination_rate",
}


def is_rag_eval_json(obj: Dict[str, Any]) -> bool:
    """Return True if *obj* looks like an internal RAG evaluation block."""
    return bool(RAG_EVAL_KEYWORDS & obj.keys())


def remove_rag_eval_blocks(text: str) -> str:
    """Strip ```json blocks that contain RAG evaluation fields."""
    if not text:
        return text
    keys_alt = "|".join(RAG_EVAL_KEYWORDS)
    pattern = rf'```json\s*\{{[\s\S]*?"(?:{keys_alt})[\s\S]*?\}}\s*```'
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Overlap / repetition prevention (used by streaming layer)
# ---------------------------------------------------------------------------

def anti_overlap_append(current: str, new_token: str, window: int = 200) -> str:
    """
    Append *new_token* to *current* while removing any suffix/prefix overlap.

    Prevents the common streaming artefact where the end of *current* is
    repeated at the start of *new_token*.
    """
    if not new_token:
        return current
    tail = current[-window:] if current else ""
    max_check = min(len(new_token), len(tail))
    overlap = 0
    for i in range(1, max_check + 1):
        if tail.endswith(new_token[:i]):
            overlap = i
    return current + new_token[overlap:]


def ngram_block(text: str, n: int = 5) -> str:
    """Remove the last n-gram if it duplicates the one immediately before it."""
    tokens = text.split()
    if len(tokens) >= 2 * n and tokens[-n:] == tokens[-2 * n : -n]:
        return " ".join(tokens[:-n])
    return text


# ---------------------------------------------------------------------------
# Paragraph-level hashing (for cross-bin dedup)
# ---------------------------------------------------------------------------

def para_hash(text: str, min_length: int = 80) -> str:
    """
    Return an MD5 hex digest for *text*, or "" if it is too short to hash
    reliably (avoids false-positive dedup of headings / short sentences).
    """
    if not text or len(text.strip()) < min_length:
        return ""
    norm = re.sub(r"\n{3,}", "\n\n", text.strip())
    norm = re.sub(r" +", " ", norm).lower()
    return hashlib.md5(norm.encode()).hexdigest()


def dedupe_markdown_blocks(md: str, min_block_len: int = 80) -> str:
    """
    Conservative paragraph-level deduplication for the final PVG Markdown.

    Only plain text blocks longer than *min_block_len* are hashed; code fences
    and tables are always kept to avoid accidental deletion.
    """
    if not md:
        return md

    lines = md.split("\n")
    out: List[str] = []
    seen: Set[str] = set()
    in_code = False
    fence = re.compile(r"^\s*```")
    buf: List[str] = []

    def _flush():
        if not buf:
            return
        block = "\n".join(buf).strip("\n")
        if not block.strip():
            out.append("")
            return
        # Never dedup tables
        if any(line.lstrip().startswith("|") for line in buf):
            out.append(block)
            return
        if len(block) < min_block_len:
            out.append(block)
            return
        h = para_hash(block, min_length=min_block_len)
        if h and h in seen:
            return
        if h:
            seen.add(h)
        out.append(block)

    for line in lines:
        if fence.match(line):
            _flush()
            buf = []
            out.append(line)
            in_code = not in_code
            continue
        if in_code:
            out.append(line)
            continue
        if not line.strip():
            _flush()
            buf = []
            out.append("")
            continue
        if line.lstrip().startswith("#"):
            _flush()
            buf = []
            out.append(line.rstrip())
            continue
        buf.append(line.rstrip())

    _flush()
    result = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def is_separator_line(line: str) -> bool:
    """Return True if *line* is a Markdown table separator (e.g. |---|---|)."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [c.strip() for c in stripped[1:-1].split("|")]
    return all(
        any(p in cell or cell == "" for p in ("---", ":---", "---:", ":---:"))
        for cell in cells
    )
