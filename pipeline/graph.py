"""
LangGraph workflow — the four-node pipeline that transforms CPG → PVG.

    START → file_process → prep → build → generate → END

Each node function reads from and writes to ``GraphState``.
``create_graph()`` wires them together and returns a compiled graph.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from config import TREE_BIN_TOKEN_LIMIT
from models import ClassifiedTextBlock, HeadingCandidate, RawChunk
from pipeline.extractor import ExtractAgent
from pipeline.formatter import finalize_pvg_markdown
from pipeline.generator import PVGRunner
from pipeline.hierarchy import TitleHierarchyAgent
from pipeline.splitter import SmartSplitter
from pipeline.state import GraphState
from pipeline.tree import build_tree, stream_aggregate
from utils.debug import load_json, save_json, save_markdown

logger = logging.getLogger(__name__)


# ===================================================================
# Node 1 — file_process
# ===================================================================

def file_process(state: GraphState) -> GraphState:
    """Read documents (file paths or raw strings) into ``context``."""
    texts: List[str] = []
    for doc in state.get("documents") or []:
        if not isinstance(doc, str):
            continue
        if os.path.isfile(doc):
            with open(doc, "r", encoding="utf-8") as fh:
                content = fh.read()
            if content.strip():
                texts.append(content)
        elif doc.strip():
            texts.append(doc)
    state["context"] = texts
    return state


# ===================================================================
# Node 2 — prep  (SmartTreePreparation)
# ===================================================================

def _split_title_body(content: str) -> Tuple[str, str]:
    lines = content.split("\n")
    if not lines:
        return "", ""
    return lines[0], "\n".join(lines[1:]).strip()


def prep(state: GraphState, config: RunnableConfig) -> GraphState:
    """Split raw text into chunks and identify heading candidates."""
    if state.get("is_resume"):
        return state

    splitter = SmartSplitter()
    all_chunks: List[RawChunk] = []
    for text in state.get("context") or []:
        all_chunks.extend(splitter.split_text(text))

    candidates: List[HeadingCandidate] = []
    for chunk in all_chunks:
        looks_like_header = (
            chunk.is_potential_header
            or bool(re.match(r"^\s*#{1,6}\s+", chunk.content))
        )
        if not looks_like_header:
            continue
        title_line, body = _split_title_body(chunk.content)
        raw_title = re.sub(r"^#+\s*", "", title_line).strip()
        preview = (
            f"[PREVIEW: {splitter.get_smart_preview(body)}]"
            if body
            else "[STRUCTURAL HEADER]"
        )
        candidates.append(HeadingCandidate(
            heading_id=chunk.id,
            text=raw_title,
            preview=preview,
        ))

    state["raw_chunks"] = all_chunks
    state["heading_candidates"] = candidates
    return state


# ===================================================================
# Node 3 — build  (SmartTreeBuilding)
# ===================================================================

def build(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use the hierarchy agent to classify headings, then build a metadata map
    that annotates every chunk with its label, path, and parent.
    """
    prefix = state.get("cache_prefix", "unknown")

    # --- Resume path ---
    cached = load_json(prefix, "SmartTree_Blocks") if state.get("is_resume") else None
    if cached is not None:
        logger.info("Resuming from cached SmartTree (prefix=%s)", prefix)

        restored_chunks: List[RawChunk] = []
        metadata_map: Dict[int, Any] = {}
        for item in cached:
            restored_chunks.append(RawChunk(
                id=item["idx"],
                content=item["content"],
                is_potential_header=item["metadata"].get("is_heading", False),
                header_level=item["metadata"].get("level", 0),
                represent_text="",
                original_idx=item["idx"],
            ))
            meta = item["metadata"]
            meta["label"] = item["label"]
            meta["path"] = meta.get("context_hint") or meta.get("path", "General")
            meta["title"] = meta.get("chapter") or meta.get("title", "General")
            metadata_map[item["idx"]] = meta

        state["raw_chunks"] = restored_chunks
        state["chunk_metadata_map"] = metadata_map
        return state

    # --- Normal path ---
    all_chunks = state.get("raw_chunks", [])
    candidates = state.get("heading_candidates", [])
    if not candidates:
        return state

    agent = TitleHierarchyAgent(llm=state["reasoning_llm"])
    nodes_map = agent.classify(candidates)

    def _get_path(node_id: int, visited: set | None = None) -> List[str]:
        if visited is None:
            visited = set()
        node = nodes_map.get(node_id)
        if not node or node_id in visited:
            return []
        visited.add(node_id)
        title = node.title or "Root"
        if node.parent_id is not None and node.parent_id in nodes_map:
            return _get_path(node.parent_id, visited) + [title]
        return [title]

    metadata_map = {}
    last_meta: Dict[str, Any] = {
        "path": "General", "label": "B", "keep": True,
        "title": "General", "parent_id": None,
    }

    for chunk in all_chunks:
        if chunk.id in nodes_map:
            node = nodes_map[chunk.id]
            path = " > ".join(_get_path(chunk.id))
            last_meta = {
                "path": path,
                "label": node.label,
                "keep": node.keep,
                "title": node.title,
                "parent_id": node.parent_id,
            }
            metadata_map[chunk.id] = {**last_meta, "is_heading": True}
        else:
            # Body chunk inherits from the nearest preceding heading
            parent_id = None
            for pid in range(chunk.id - 1, -1, -1):
                if pid in metadata_map and metadata_map[pid].get("is_heading"):
                    parent_id = pid
                    break
            metadata_map[chunk.id] = {
                **last_meta, "parent_id": parent_id, "is_heading": False,
            }

    state["chunk_metadata_map"] = metadata_map

    # Save SmartTree snapshot for debugging and future resume
    prefix = state.get("cache_prefix", "unknown")
    snapshot_data = [
        {
            "idx": chunk.id,
            "content": chunk.content,
            "label": metadata_map[chunk.id].get("label", "B"),
            "metadata": metadata_map[chunk.id],
        }
        for chunk in all_chunks
        if chunk.id in metadata_map
    ]
    save_json(prefix, "SmartTree_Blocks", snapshot_data)
    logger.info(
        "SmartTree built: %d chunks, %d headings.",
        len(metadata_map),
        sum(1 for m in metadata_map.values() if m.get("is_heading")),
    )

    return state


# ===================================================================
# Node 4 — generate  (public_version_generate)
# ===================================================================

def generate(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    1. Extract global facts  (or load cached).
    2. Build the SmartTree and bin it.
    3. Run PVGRunner to generate per-bin Markdown.
    """
    raw_chunks = state.get("raw_chunks")
    metadata_map = state.get("chunk_metadata_map")
    if not raw_chunks or not metadata_map:
        raise KeyError("Missing raw_chunks or chunk_metadata_map in state.")

    prefix = state.get("cache_prefix", "unknown")

    # 1. Facts extraction
    cached_facts = load_json(prefix, "All_Facts")
    if state.get("is_resume") and cached_facts:
        logger.info("Resuming from cached facts sidebar.")
        all_facts = cached_facts
    else:
        extractor = ExtractAgent(llm=state["reasoning_llm"])
        output = extractor.extract(raw_chunks, metadata_map)
        all_facts = {
            "A": output.A_extraction,
            "B": output.B_extraction,
            "C": output.C_extraction,
            "D": output.D_extraction,
            "E": output.E_extraction,
        }
        save_json(prefix, "All_Facts", all_facts)

        # Human-readable version
        md_lines = ["# Extracted Global Facts\n"]
        for cat, qa in all_facts.items():
            md_lines.append(f"## Category {cat}")
            for q, a in qa.items():
                md_lines.append(f"**{q}**\n{a}\n")
            md_lines.append("---")
        save_markdown(prefix, "All_Facts", "\n".join(md_lines))

    # 2. Build tree + bin
    tree = build_tree(raw_chunks, metadata_map)
    logger.info(
        "SmartTree: %d nodes, %d roots.", len(tree.nodes), len(tree.roots),
    )
    bins = stream_aggregate(tree, safe_limit=TREE_BIN_TOKEN_LIMIT)
    logger.info("Aggregated into %d chapter bins.", len(bins))

    # 3. Generate
    runner = PVGRunner(generating_llm=state["generating_llm"])
    raw_pvg = runner.run(bins, all_facts, cache_prefix=prefix)

    state["messages"] = [raw_pvg]
    state["pilot"] = bins
    return state


# ===================================================================
# Graph factory
# ===================================================================

def create_graph():
    """Return a compiled LangGraph ready for ``invoke()`` or ``astream()``."""
    wf = StateGraph(GraphState)
    wf.add_node("file_process", file_process)
    wf.add_node("prep", prep)
    wf.add_node("build", build)
    wf.add_node("generate", generate)

    wf.add_edge(START, "file_process")
    wf.add_edge("file_process", "prep")
    wf.add_edge("prep", "build")
    wf.add_edge("build", "generate")
    wf.add_edge("generate", END)

    return wf.compile()
