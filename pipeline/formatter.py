"""
Markdown post-processing for the final PVG output.

All regex-based cleanup lives here.  The single public entry point is
``finalize_pvg_markdown()``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

from utils.text import dedupe_markdown_blocks


# ===================================================================
# Internal helpers
# ===================================================================

_CODE_PLACEHOLDER = "__PVG_CODE__"


def _extract_code_blocks(md: str) -> Tuple[str, Dict[str, str]]:
    """Replace fenced code blocks with placeholders to protect them."""
    if "```" not in (md or ""):
        return md, {}

    lines = (md or "").split("\n")
    out: List[str] = []
    mapping: Dict[str, str] = {}
    in_code = False
    buf: List[str] = []
    idx = 0

    for line in lines:
        if not in_code and line.lstrip().startswith("```"):
            in_code = True
            buf = [line]
            continue
        if in_code:
            buf.append(line)
            if line.strip() == "```":
                key = f"{_CODE_PLACEHOLDER}{idx}__"
                mapping[key] = "\n".join(buf)
                out.append(key)
                idx += 1
                in_code = False
                buf = []
            continue
        out.append(line)

    if in_code and buf:
        out.extend(buf)

    return "\n".join(out), mapping


def _restore_code_blocks(md: str, mapping: Dict[str, str]) -> str:
    for key, block in mapping.items():
        md = md.replace(key, block)
    return md


# ===================================================================
# Individual fixers (each does ONE thing)
# ===================================================================

def _fix_unclosed_mermaid(md: str) -> str:
    """Close ``mermaid`` fences that were truncated by the LLM."""
    if "```mermaid" not in md.lower():
        return md

    lines = md.split("\n")
    out: List[str] = []
    in_mermaid = False

    def _is_md_boundary(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s in ("---", "***"):
            return True
        if re.match(r"^#{1,6}\s+[0-9a-fA-F]{3,8}\s*$", s):
            return False
        if re.match(r"^#{1,6}\s+\S", s):
            return True
        if s.startswith("|"):
            return True
        if re.match(r"^(\*|-|\d+\.)\s+", s):
            return True
        return False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^```\s*mermaid\b", stripped, re.IGNORECASE):
            in_mermaid = True
            out.append("```mermaid")
            continue
        if stripped == "```" and in_mermaid:
            in_mermaid = False
            out.append("```")
            continue
        if in_mermaid and _is_md_boundary(line):
            out.append("```")
            in_mermaid = False
        out.append(line)

    if in_mermaid:
        out.append("```")
    return "\n".join(out)


def _fix_heading_spaces(md: str) -> str:
    """Remove empty headings and normalise ``# # #`` patterns."""
    md_safe, code_map = _extract_code_blocks(md)

    md_safe = re.sub(r"(?m)^\s*#+\s*$", "", md_safe)
    md_safe = re.sub(r"#+\s*\*\*([^*]+)\*\*#*", r"\n\n**\1**\n\n", md_safe)
    md_safe = re.sub(r"\n{3,}", "\n\n", md_safe)

    def _merge_hashes(m: re.Match) -> str:
        count = min(m.group(0).count("#"), 6)
        return "#" * count + " "

    md_safe = re.sub(r"(?m)^#(\s+#)+\s+", _merge_hashes, md_safe)
    md_safe = re.sub(r"(?m)^(##)\s+#\s+", r"## ", md_safe)
    md_safe = re.sub(r"(?m)^(###)\s+#\s+", r"### ", md_safe)
    md_safe = re.sub(r"(?m)^(#)\s+(#+)\s+", r"\1\2 ", md_safe)

    # Ensure headings have vertical spacing
    md_safe = re.sub(r"([^\n])(#{1,6}\s+)", r"\1\n\n\2", md_safe)
    md_safe = re.sub(r"([^\n])\n(#{1,6}\s+)", r"\1\n\n\2", md_safe)
    md_safe = re.sub(r"(#{1,6}\s+[^\n]+)\n([^\n])", r"\1\n\n\2", md_safe)
    md_safe = re.sub(r"\n{3,}", "\n\n", md_safe)

    return _restore_code_blocks(md_safe, code_map)


def _fix_code_fences(md: str) -> str:
    """Ensure every opening ``` has a matching close."""
    lines = md.split("\n")
    in_code = False
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") and not in_code:
            in_code = True
        elif stripped == "```" and in_code:
            in_code = False
        elif stripped.startswith("```") and in_code:
            out.append("```")
            in_code = True
        out.append(line)
    if in_code:
        out.append("```")
    return "\n".join(out)


def _fix_mermaid_format(md: str) -> str:
    """Deduplicate nested mermaid fences, clean internal heading artefacts."""
    md = re.sub(
        r"```mermaid\s*\n\s*```mermaid", "```mermaid",
        md, flags=re.IGNORECASE,
    )

    def _clean_block(m: re.Match) -> str:
        content = re.sub(r"^\s*#+\s*", "", m.group(1), flags=re.MULTILINE)
        return f"```mermaid\n{content.strip()}\n```"

    md = re.sub(
        r"```mermaid\n([\s\S]*?)\n```", _clean_block,
        md, flags=re.IGNORECASE,
    )
    return md


def _fix_mermaid_style_fill(md: str) -> str:
    """Repair ``fill:\\n# e1f5e1`` corruption inside mermaid blocks."""
    if "```mermaid" not in (md or "").lower():
        return md
    pat = re.compile(r"```mermaid\s*\n([\s\S]*?)\n```", re.IGNORECASE)

    def _repair(m: re.Match) -> str:
        content = re.sub(
            r"(fill:)\s*\n\s*(?:\n\s*)?#\s*([0-9a-fA-F]{3,8})\b",
            r"\1#\2", m.group(1), flags=re.IGNORECASE,
        )
        return "```mermaid\n" + content + "\n```"

    return pat.sub(_repair, md)


def _remove_duplicate_mermaid(md: str) -> str:
    """Keep only the first occurrence of each mermaid diagram (by hash)."""
    pat = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    seen: set = set()

    def _dedup(m: re.Match) -> str:
        digest = hashlib.md5(m.group(1).strip().encode()).hexdigest()
        if digest in seen:
            return ""
        seen.add(digest)
        return m.group(0)

    return pat.sub(_dedup, md)


def _remove_meta_talk(md: str) -> str:
    """Strip common LLM preambles and outros."""
    intro = [
        r"(?im)^Here is the (?:rewritten|adapted|summarized|public|final|converted) (?:version|content|guide|section).*?[:：]\s*\n*",
        r"(?im)^This (?:information|guide|version) is adapted from.*?\n*",
        r"(?im)^Below is (?:the|a) (?:rewritten|public-friendly).*?\n*",
        r"(?im)^Based on the provided (?:text|source|guideline).*?\n*",
    ]
    outro = [
        r"(?im)\n*Please let me know if you need any (?:further|more) (?:help|assistance).*",
        r"(?im)\n*Let me know if you have any questions.*",
    ]
    for p in intro + outro:
        md = re.sub(p, "", md)
    return md.strip()


def _remove_empty_markers(md: str) -> str:
    md = re.sub(r"\[EMPTY CONTENT NO NEED TO OUTPUT\]", "", md, flags=re.IGNORECASE)
    md = re.sub(r"\[NO CONTENT TO ADAPT\]", "", md, flags=re.IGNORECASE)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", md)


def _demote_long_headings(md: str) -> str:
    """Turn headings that are really full sentences back into body text."""
    lines = md.split("\n")
    out: List[str] = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            content = m.group(2).strip()
            if len(content) > 60 or (len(content) > 20 and content[-1] in ".!?"):
                out.append(content)
                continue
        out.append(line)
    return "\n".join(out)


def _join_split_headings(md: str) -> str:
    """Merge headings that were split across multiple lines."""
    connector = (
        r"(\bthe|\bof|\bfor|\bis|\band|\bwith|\bto|\bin|\ba|\bor"
        r"|Sarcoma|Sarcomas|Guideline|Update|Management|Part|Section"
        r"|[\-—\(\[（])\s*$"
    )
    lines = md.split("\n")
    joined: List[str] = []
    i = 0
    while i < len(lines):
        curr = lines[i].rstrip()
        if curr.lstrip().startswith("#"):
            while i + 1 < len(lines):
                if re.search(connector, curr, re.IGNORECASE) or curr.endswith("**"):
                    ni = i + 1
                    while ni < len(lines) and not lines[ni].strip():
                        ni += 1
                    if ni < len(lines):
                        nxt = lines[ni].strip()
                        if not nxt.startswith("#") and len(nxt.split()) <= 6:
                            curr = f"{curr} {nxt}"
                            i = ni
                            continue
                break
        joined.append(curr)
        i += 1
    return "\n".join(joined)


def _revive_mermaid(md: str) -> str:
    """Remove ``#`` prefixes that the LLM inserted inside mermaid blocks."""
    lines = md.split("\n")
    in_mermaid = False
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^```\s*mermaid", stripped, re.IGNORECASE):
            in_mermaid = True
            out.append(line)
            continue
        if stripped == "```" and in_mermaid:
            in_mermaid = False
            out.append(line)
            continue
        if in_mermaid:
            clean = re.sub(r"^\s*#+\s*", "", line)
            clean = re.sub(r'(?<!["\'])##\s*', "", clean)
            if clean.strip():
                out.append(clean)
        else:
            out.append(line)
    return "\n".join(out)


def _move_noisy_sections_to_appendix(md: str, min_chars: int = 1500) -> str:
    """Wrap long reference-like sections in a collapsible ``<details>`` block."""
    if not md:
        return md
    lines = md.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("#"):
            continue
        title = re.sub(r"^#+\s*", "", s).strip().lower()
        if any(k in title for k in ("references", "reference", "key research",
                                     "research studies", "studies by topic")):
            start_idx = i
            break
    if start_idx is None:
        return md
    appendix = "\n".join(lines[start_idx:]).strip()
    main = "\n".join(lines[:start_idx]).rstrip()
    if len(appendix) < min_chars:
        return md
    if re.search(r"(?im)^##\s+Appendix\b", main):
        return main + "\n\n" + appendix
    block = (
        "\n\n## Appendix\n\n"
        "<details>\n"
        "<summary><strong>References and long supporting lists "
        "(click to expand)</strong></summary>\n\n"
        f"{appendix}\n\n"
        "</details>\n"
    )
    return (main + block).strip()


def _enforce_style(md: str) -> str:
    """Final whitespace normalisation."""
    md = md.replace("***", "---")
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ===================================================================
# Public entry point
# ===================================================================

def finalize_pvg_markdown(md: str) -> str:
    """
    One-shot finaliser applied to the concatenated PVG output.

    Applies all fixes in a carefully ordered sequence.  Never raises —
    returns the original text on unexpected errors.
    """
    try:
        md = _revive_mermaid(md)
        md = _remove_meta_talk(md)
        md = _remove_empty_markers(md)
        md = _demote_long_headings(md)
        md = _fix_heading_spaces(md)
        md = _join_split_headings(md)

        md = _fix_unclosed_mermaid(md)
        md = _fix_code_fences(md)
        md = _fix_mermaid_format(md)
        md = _fix_mermaid_style_fill(md)
        md = _remove_duplicate_mermaid(md)

        md = dedupe_markdown_blocks(md)
        md = _move_noisy_sections_to_appendix(md)
        return _enforce_style(md)
    except Exception:
        return (md or "").strip()
