"""
SmartTree — build the document tree and bin it for streaming generation.

Two main components:

1. ``SmartTreeGraph`` : in-memory tree built from raw chunks + metadata.
2. ``stream_aggregate`` : greedy DFS binning that packs subtrees into
   token-limited bins ready for one LLM call each.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import TREE_BIN_TOKEN_LIMIT
from models import (
    ClassifiedTextBlock,
    RawChunk,
    SerializableDocument,
    SmartTreeNode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

class SmartTreeGraph:
    """
    Adjacency-list tree of ``SmartTreeNode``.

    Nodes are inserted in chunk-id order.  If a node's declared parent has
    not been added yet it is promoted to a root (best-effort recovery).
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, SmartTreeNode] = {}
        self.roots: List[SmartTreeNode] = []

    def add_node(
        self,
        id: int,
        content: str,
        label: str,
        parent_id: Optional[int],
        is_heading: bool,
        title: str = "",
        path: str = "",
        keep: bool = True,
    ) -> None:
        node = SmartTreeNode(
            id=id, content=content, label=label,
            parent_id=parent_id, is_heading=is_heading,
            title=title, path=path, keep=keep,
        )
        self.nodes[id] = node
        if parent_id is not None and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node)
        else:
            self.roots.append(node)

    def dfs_order(self) -> List[SmartTreeNode]:
        """All nodes in depth-first (natural reading) order."""
        result: List[SmartTreeNode] = []

        def _visit(n: SmartTreeNode) -> None:
            result.append(n)
            for child in n.children:
                _visit(child)

        for root in self.roots:
            _visit(root)
        return result

    def ancestors(self, node_id: int) -> List[SmartTreeNode]:
        """Return the chain of ancestors from root down (excluding *node_id*)."""
        chain: List[SmartTreeNode] = []
        curr = self.nodes.get(node_id)
        while curr and curr.parent_id is not None:
            parent = self.nodes.get(curr.parent_id)
            if parent:
                chain.insert(0, parent)
                curr = parent
            else:
                break
        return chain


# ---------------------------------------------------------------------------
# Tree builder (from raw chunks + metadata)
# ---------------------------------------------------------------------------

def build_tree(
    raw_chunks: List[Any],
    metadata_map: Dict[int, Any],
) -> SmartTreeGraph:
    """Construct a ``SmartTreeGraph`` from the splitter output + metadata."""
    tree = SmartTreeGraph()
    for chunk in raw_chunks:
        meta = metadata_map.get(chunk.id)
        if meta is None:
            continue
        tree.add_node(
            id=chunk.id,
            content=chunk.content,
            label=meta.get("label", "B"),
            parent_id=meta.get("parent_id"),
            is_heading=meta.get("is_heading", False),
            title=meta.get("title", ""),
            path=meta.get("path", ""),
            keep=meta.get("keep", True),
        )
    return tree


# ---------------------------------------------------------------------------
# Greedy streaming binner
# ---------------------------------------------------------------------------

def stream_aggregate(
    tree: SmartTreeGraph,
    safe_limit: int = TREE_BIN_TOKEN_LIMIT,
) -> List[ClassifiedTextBlock]:
    """
    Pack tree nodes into token-bounded bins using greedy DFS.

    Strategy:
    - Walk roots in order.
    - If a whole subtree fits within *safe_limit*, take it as one unit
      (preserves context).
    - Otherwise, add the heading node to the current bin and recurse into
      children one by one.
    - When a node would overflow the bin, flush the bin first.

    Each bin becomes a ``ClassifiedTextBlock`` with ``label="STREAM_BIN"``
    carrying rich metadata (breadcrumb path, sibling titles, etc.) so that
    the generation prompt has full structural context.
    """
    bins: List[ClassifiedTextBlock] = []
    current_nodes: List[SmartTreeNode] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current_nodes, current_tokens
        if not current_nodes:
            return

        content = "\n\n".join(n.content for n in current_nodes)
        first = current_nodes[0]

        # Breadcrumb path
        ancestors = tree.ancestors(first.id)
        breadcrumbs = " > ".join(
            [a.title for a in ancestors]
            + [first.title if first.is_heading else ""]
        )

        # Sibling titles (for cross-context awareness)
        siblings: List[str] = []
        if first.parent_id is not None and first.parent_id in tree.nodes:
            parent = tree.nodes[first.parent_id]
            siblings = [
                c.title for c in parent.children
                if c.id != first.id and c.title
            ]
        elif first.parent_id is None:
            siblings = [
                r.title for r in tree.roots
                if r.id != first.id and r.title
            ]

        bin_keep = any(n.keep for n in current_nodes)

        bins.append(ClassifiedTextBlock(
            doc=SerializableDocument(
                page_content=content,
                metadata={
                    "context_hint": breadcrumbs,
                    "siblings": siblings,
                    "contained_labels": list({n.label for n in current_nodes}),
                    "token_count": current_tokens,
                    "start_node_id": first.id,
                    "end_node_id": current_nodes[-1].id,
                    "keep": bin_keep,
                },
            ),
            idx=len(bins),
            label="STREAM_BIN",
            confidence=1.0,
        ))
        current_nodes = []
        current_tokens = 0

    def _add(node: SmartTreeNode) -> None:
        nonlocal current_tokens
        if current_tokens + node.self_tokens > safe_limit and current_nodes:
            _flush()
        current_nodes.append(node)
        current_tokens += node.self_tokens

    def _add_subtree(node: SmartTreeNode) -> None:
        _add(node)
        for child in node.children:
            _add_subtree(child)

    def _process(node: SmartTreeNode) -> None:
        if current_tokens + node.subtree_tokens < safe_limit:
            _add_subtree(node)
        else:
            _add(node)
            for child in node.children:
                _process(child)

    for root in tree.roots:
        if current_nodes and root.keep != current_nodes[0].keep:
            _flush()
        _process(root)
    _flush()

    logger.info("Aggregated %d tree nodes into %d bins.", len(tree.nodes), len(bins))
    return bins
