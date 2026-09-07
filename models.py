"""
Shared data types used across the entire pipeline.

Keeping all model definitions in one place avoids circular imports and makes
the data flow explicit: every module imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document wrappers  (Pydantic – serialisable, used in LangGraph state)
# ---------------------------------------------------------------------------

class SerializableDocument(BaseModel):
    """Minimal document that can travel through JSON / LangGraph state."""
    page_content: str = Field(..., description="Main text content")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextBlock(BaseModel):
    """A document paired with its positional index."""
    doc: SerializableDocument
    idx: int


class ClassifiedTextBlock(TextBlock):
    """A TextBlock that has been assigned a semantic label."""
    label: str = Field(..., description="Semantic category (A-E or STREAM_BIN)")
    confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# Chunking types  (dataclass – lightweight, used inside splitter / tree)
# ---------------------------------------------------------------------------

@dataclass
class RawChunk:
    """One chunk produced by SmartSplitter."""
    id: int
    content: str
    is_potential_header: bool
    header_level: int           # 1-6 for markdown headers, 0 for inferred
    represent_text: str         # short text optimised for LLM perception
    original_idx: int
    has_following_text: bool = False


@dataclass
class HeadingCandidate:
    """A chunk that *might* be a section heading, sent to the hierarchy agent."""
    heading_id: int
    text: str
    preview: str


# ---------------------------------------------------------------------------
# Smart-tree types  (dataclass – used by the tree builder and binner)
# ---------------------------------------------------------------------------

@dataclass
class SmartTreeNode:
    """One node in the reconstructed document tree."""
    id: int
    content: str
    label: str
    parent_id: Optional[int]
    is_heading: bool
    title: str = ""
    path: str = ""
    keep: bool = True
    children: List[SmartTreeNode] = field(default_factory=list)

    @property
    def self_tokens(self) -> int:
        return len(self.content) // 4

    @property
    def subtree_tokens(self) -> int:
        total = self.self_tokens
        for child in self.children:
            total += child.subtree_tokens
        return total


# ---------------------------------------------------------------------------
# Extraction output
# ---------------------------------------------------------------------------

class ExtractOutput(BaseModel):
    """Facts extracted per semantic category (A–E)."""
    A_extraction: Dict[str, str] = Field(default_factory=dict)
    B_extraction: Dict[str, str] = Field(default_factory=dict)
    C_extraction: Dict[str, str] = Field(default_factory=dict)
    D_extraction: Dict[str, str] = Field(default_factory=dict)
    E_extraction: Dict[str, str] = Field(default_factory=dict)
