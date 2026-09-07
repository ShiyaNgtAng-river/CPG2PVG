"""
LangGraph shared state definition.

The entire pipeline reads from and writes to this single TypedDict so that
every node has a clear contract of what it consumes and produces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI

from models import ClassifiedTextBlock, HeadingCandidate, RawChunk


class GraphState(TypedDict, total=False):
    # --- Inputs ---
    documents: List[str]              # file paths or raw markdown strings
    reasoning_llm: ChatOpenAI         # LLM for analytical tasks (hierarchy, extraction)
    generating_llm: ChatOpenAI        # LLM for creative writing (PVG generation)
    cache_prefix: str                 # prefix for debug snapshot filenames
    is_resume: bool                   # True if resuming from cached snapshots

    # --- Intermediate ---
    context: List[str]                # raw text read from documents
    raw_chunks: List[RawChunk]
    heading_candidates: List[HeadingCandidate]
    chunk_metadata_map: Dict[int, Dict[str, Any]]
    pilot: List[ClassifiedTextBlock]  # bins produced by the tree binner

    # --- Output ---
    messages: List[str]               # final PVG markdown segments
