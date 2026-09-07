"""
merge_chunks.py

从 TEST_Data/1/splitted_texts/ 读取每个 _chunks.json，
把所有 chunk 按 chunk_index 顺序拼接成干净的 Markdown，
输出到 TEST_Data/available_cpgs/，保留原文件名（去掉 _chunks 后缀）。

运行方式：
    python merge_chunks.py
"""

import json
import os

# ── 路径配置 ──────────────────────────────────────────────────
INPUT_DIR = os.path.join("TEST_Data", "1", "splitted_texts")
OUTPUT_DIR = os.path.join("TEST_Data", "available_cpgs")


def merge_one(json_path: str, output_path: str) -> None:
    """读取一个 _chunks.json，拼接所有 chunk，写成 .md 文件。"""
    with open(json_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # 按 chunk_index 排序，防止顺序乱掉
    chunks.sort(key=lambda c: c["chunk_index"])

    # 把每个 chunk 的 content 拼在一起，chunk 之间保留一个空行
    full_text = "\n\n".join(chunk["content"].strip() for chunk in chunks)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 找出所有 _chunks.json 文件
    json_files = [
        fname
        for fname in os.listdir(INPUT_DIR)
        if fname.endswith("_chunks.json")
    ]

    if not json_files:
        print(f"在 {INPUT_DIR} 里没有找到 _chunks.json 文件")
        return

    for fname in sorted(json_files):
        # 去掉 _chunks.json 后缀，换成 .md
        base_name = fname.replace("_chunks.json", "")
        output_name = base_name + ".md"

        json_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, output_name)

        merge_one(json_path, output_path)
        print(f"✓  {fname}  →  {output_name}")

    print(f"\n完成：共处理 {len(json_files)} 个文件，输出到 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
