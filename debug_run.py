"""
调试运行脚本 — 直接在终端运行整条流水线，逐节点查看中间结果。

用法:
    python debug_run.py                           # 交互模式（选择文件）
    python debug_run.py --input path/to/file.md   # 指定文件
    python debug_run.py --only-tree                # 只运行到 SmartTree 构建完成就停
    python debug_run.py --resume                   # 使用上次缓存的中间结果继续
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

# 确保项目根目录在导入路径中
sys.path.insert(0, os.path.dirname(__file__))

from config import DEBUG_SNAPSHOTS_DIR
from llm import get_llm
from models import ClassifiedTextBlock, HeadingCandidate, RawChunk
from pipeline.graph import create_graph


# ---------------------------------------------------------------------------
# LLM 调用计数器（基于 LangChain Callback 机制）
# ---------------------------------------------------------------------------

from langchain_core.callbacks import BaseCallbackHandler


class CallCountCallback(BaseCallbackHandler):
    """
    Reasoning LLM 专用 Callback：只打印调用序号和端点，不打印内容（避免刷屏）。
    """

    def __init__(self, role: str, base_url: str) -> None:
        super().__init__()
        self.role       = role
        self.base_url   = base_url
        self.call_count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.call_count += 1
        model = serialized.get("kwargs", {}).get("model_name", "unknown-model")
        print(f"  [LLM ↑ #{self.call_count:>3}] {self.role:<12} "
              f"→ {model}  |  {self.base_url}")


class GeneratingCallback(BaseCallbackHandler):
    """
    Generating LLM 专用 Callback：
    - Terminal：只打印进度（第几次调用），不展示内容
    - 详细内容由 InvokeTracer 在每次调用结束后写入 Invoke_Trace.md
    """

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url   = base_url
        self.call_count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.call_count += 1
        print(f"  [Bin #{self.call_count:>3}] 生成中... ", end="", flush=True)

    def on_llm_end(self, response, **kwargs) -> None:
        print("完成  →  已写入 Invoke_Trace.md")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def divider(title: str) -> None:
    """打印一条醒目的分割线。"""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def save_snapshot(prefix: str, filename: str, content: str) -> None:
    """保存调试快照到 debug_snapshots/<prefix>/ 子目录。"""
    directory = os.path.join(DEBUG_SNAPSHOTS_DIR, prefix)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [快照已保存] -> {path}")


# ---------------------------------------------------------------------------
# 节点级调试检查器
# ---------------------------------------------------------------------------

def inspect_node(node_name: str, state: dict, prefix: str, only_tree: bool) -> None:
    """
    每个节点运行完毕后被调用，检查该节点输出的状态并保存快照。

    这就是 debug 的核心：你能看到数据在管道中每一步的变化。
    """
    divider(f"节点完成: {node_name}")

    # ---- prep 节点：查看分块结果 ----
    if node_name == "prep":
        chunks: List[RawChunk] = state.get("raw_chunks", [])
        candidates: List[HeadingCandidate] = state.get("heading_candidates", [])

        if chunks:
            md = f"# 原始分块结果 ({prefix})\n\n"
            md += f"**总块数**: {len(chunks)}\n\n"
            for c in chunks:
                tag = "HEADER" if c.is_potential_header else "BODY"
                md += f"## Chunk {c.id} | {tag} | Level {c.header_level}\n"
                md += f"{c.content}\n\n---\n\n"
            save_snapshot(prefix, "00_Raw_Chunks.md", md)
            print(f"  -> 共 {len(chunks)} 个原始块")

        if candidates:
            payload = [
                {"id": c.heading_id, "text": c.text, "preview": c.preview}
                for c in candidates
            ]
            save_snapshot(prefix, "00_Heading_Candidates.json",
                          json.dumps(payload, ensure_ascii=False, indent=2))
            print(f"  -> 共 {len(candidates)} 个标题候选")

    # ---- build 节点：查看 SmartTree 索引 ----
    elif node_name == "build":
        metadata_map = state.get("chunk_metadata_map", {})
        chunks = state.get("raw_chunks", [])

        if metadata_map and chunks:
            md = f"# SmartTree 逻辑索引 ({prefix})\n\n"
            md += f"**总块数**: {len(metadata_map)}\n\n"
            for c in chunks:
                meta = metadata_map.get(c.id, {})
                if not meta:
                    continue
                md += (
                    f"## Chunk {c.id} | Label: {meta.get('label')} "
                    f"| Path: {meta.get('path')}\n"
                )
                md += f"{c.content[:200]}...\n\n---\n\n"
            save_snapshot(prefix, "01_SmartTree_Index.md", md)

            headings = sum(1 for m in metadata_map.values() if m.get("is_heading"))
            print(f"  -> 共 {len(metadata_map)} 块，其中 {headings} 个标题")

        if only_tree:
            divider("快速预览模式：SmartTree 构建完成，停止运行")
            sys.exit(0)

    # ---- generate 节点：查看聚合 bins 和最终结果 ----
    elif node_name == "generate":
        bins: List[ClassifiedTextBlock] = state.get("pilot", [])
        if bins:
            md = f"# 聚合后的 ChapterBins ({prefix})\n\n"
            for i, b in enumerate(bins):
                meta = b.doc.metadata
                md += (
                    f"## Bin {i} | Tokens: {meta.get('token_count')} "
                    f"| Path: {meta.get('context_hint')}\n"
                )
                md += f"**兄弟节点**: {meta.get('siblings')}\n\n"
                md += f"{b.doc.page_content}\n\n---\n\n"
            save_snapshot(prefix, "02_ChapterBins.md", md)
            print(f"  -> 共 {len(bins)} 个 ChapterBin")

        messages = state.get("messages", [])
        if messages:
            final = messages[-1]
            save_snapshot(prefix, "03_Final_PVG.md",
                          f"# 最终 PVG 结果 ({prefix})\n\n{final}")
            print(f"  -> 最终文章长度: {len(final)} 字符")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_debug(file_path: str, only_tree: bool = False, resume: bool = False) -> None:
    """运行完整的调试会话。"""
    prefix = os.path.splitext(os.path.basename(file_path))[0]

    divider(f"调试会话开始")
    print(f"  输入文件: {file_path}")
    print(f"  缓存前缀: {prefix}")
    if only_tree:
        print("  模式: 仅运行到 SmartTree 构建")
    if resume:
        print("  模式: 从缓存恢复")

    # 1. 创建流水线
    graph = create_graph()

    # 2. 创建计数器，通过 with_config 注入 LLM（LLM 本身保持合法 Runnable）
    from config import LLM_PROVIDERS
    _base_url = LLM_PROVIDERS.get("deepseek", next(iter(LLM_PROVIDERS.values()))).base_url
    reasoning_cb   = CallCountCallback(role="reasoning",  base_url=_base_url)
    generating_cb  = GeneratingCallback(base_url=_base_url)
    reasoning_llm  = get_llm("deepseek").with_config(callbacks=[reasoning_cb])
    generating_llm = get_llm("deepseek").with_config(callbacks=[generating_cb])

    # 3. 构造初始状态
    initial_state = {
        "documents": [file_path],
        "messages": [],
        "context": [],
        "pilot": [],
        "reasoning_llm": reasoning_llm,
        "generating_llm": generating_llm,
        "is_resume": resume,
        "cache_prefix": prefix,
    }

    # 4. 逐节点运行并检查，同时记录每个节点的耗时
    node_times: dict[str, float] = {}   # 记录每个节点的耗时（秒）

    try:
        t_start = time.time()
        for event in graph.stream(initial_state):
            for node_name, state_update in event.items():
                node_times[node_name] = time.time() - t_start
                inspect_node(node_name, state_update, prefix, only_tree)
                t_start = time.time()
    except SystemExit:
        pass  # only_tree 模式下的正常退出
    except Exception as e:
        print(f"\n[错误] 运行中断: {e}")
        import traceback
        traceback.print_exc()

    # 5. 打印运行摘要
    divider("运行摘要")
    total_api = reasoning_cb.call_count + generating_cb.call_count
    print(f"  {'节点':<16} {'耗时':>8}")
    print(f"  {'-'*16} {'-'*8}")
    for node, elapsed in node_times.items():
        print(f"  {node:<16} {elapsed:>7.1f}s")
    print(f"  {'合计':<16} {sum(node_times.values()):>7.1f}s")
    print()
    print(f"  API 调用次数（reasoning LLM） : {reasoning_cb.call_count}")
    print(f"  API 调用次数（generating LLM）: {generating_cb.call_count}")
    print(f"  API 调用总计                  : {total_api}")

    divider("调试会话结束")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def ask_yes_no(question: str, default: bool = False) -> bool:
    """向用户提一个 y/n 问题，回车则采用默认值。"""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"  {question} {hint}: ").strip().lower()
    except KeyboardInterrupt:
        print()
        sys.exit(0)
    if answer == "":
        return default
    return answer in ("y", "yes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPG2PVG 调试运行工具")
    parser.add_argument("--input", type=str, help="CPG Markdown 文件路径（跳过交互选择）")
    parser.add_argument("--only-tree", action="store_true",
                        help="只运行到 SmartTree 构建完成就停止")
    parser.add_argument("--resume", action="store_true",
                        help="使用上次缓存的中间结果继续运行")
    args = parser.parse_args()

    test_file = args.input
    only_tree = args.only_tree
    resume = args.resume

    # ── 交互模式：没有通过 --input 指定文件时进入 ──────────────────────────
    if not test_file:
        available_dir = os.path.join("TEST_Data", "available_cpgs")

        if not os.path.isdir(available_dir):
            print(f"错误: 找不到测试数据目录 {available_dir}")
            print("请先运行 merge_chunks.py 生成测试数据，或用 --input 指定文件。")
            sys.exit(1)

        md_files: List[str] = sorted(
            f for f in os.listdir(available_dir) if f.endswith(".md")
        )

        if not md_files:
            print(f"在 {available_dir} 里没有找到 .md 文件。")
            sys.exit(1)

        # 列出所有 CPG 供选择
        divider("选择要调试的 CPG")
        for i, fname in enumerate(md_files):
            print(f"  [{i:>2}] {fname}")

        # 让用户输入编号
        try:
            choice = input(f"\n请输入编号 (0 - {len(md_files) - 1}): ").strip()
            idx = int(choice)
            if not 0 <= idx < len(md_files):
                raise ValueError
            test_file = os.path.join(available_dir, md_files[idx])
        except (ValueError, KeyboardInterrupt):
            print("输入无效，退出。")
            sys.exit(0)

        # 追问运行选项
        print()
        only_tree = ask_yes_no("只运行到 SmartTree（不调用 LLM）？", default=False)
        resume    = ask_yes_no("使用上次缓存继续（resume）？",       default=False)

    # ── 文件存在性检查 ──────────────────────────────────────────────────────
    if not os.path.exists(test_file):
        print(f"错误: 找不到文件 {test_file}")
        sys.exit(1)

    run_debug(test_file, only_tree=only_tree, resume=resume)
