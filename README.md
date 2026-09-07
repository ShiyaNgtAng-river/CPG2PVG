# CPG2PVG

A pipeline that rewrites **Clinical Practice Guidelines** (CPG) into **Patient Version Guides** (PVG).

Input is a clinical guideline in Markdown. Output is a patient-facing Markdown document with Q&A-style headings, Mermaid flowcharts, and comparison tables.

The whole thing is orchestrated with **LangGraph** as four nodes:

```
START → file_process → prep → build → generate → END
```

`prep` is deterministic, code-only (no LLM calls). `build` and `generate` both issue LLM calls.

---

## Layout

```
.
├── config.py            # All tunable parameters + LLM provider definitions
├── llm.py               # ChatOpenAI construction
├── models.py            # Shared data types (dataclass + pydantic)
├── prompts.py           # All prompts and the CHECK_LIST question set
├── main.py              # FastAPI server (SSE streaming / non-streaming)
├── app.py               # Streamlit frontend
├── debug_run.py         # Terminal, node-by-node debug runner
├── merge_chunks.py      # Test-data prep: *_chunks.json → *.md
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── graph.py         # The four node functions + create_graph()
│   ├── state.py         # GraphState (TypedDict)
│   ├── splitter.py      # SmartSplitter: Markdown → RawChunk
│   ├── hierarchy.py     # TitleHierarchyAgent: heading classification + tree reconstruction
│   ├── tree.py          # SmartTreeGraph + stream_aggregate binning
│   ├── extractor.py     # ExtractAgent: QA-based fact extraction
│   ├── generator.py     # PVGWriter / PVGRunner: concurrent generation
│   └── formatter.py     # finalize_pvg_markdown: regex post-processing
└── utils/             
    ├── debug.py         # save_json / load_json / save_markdown / InvokeTracer
    └── text.py          # dedupe_markdown_blocks
```

Directories created at runtime:

| Directory | Written by | Contents |
|---|---|---|
| `upload_files/` | `app.py` | Raw `.md` files uploaded through Streamlit |
| `debug_snapshots/<prefix>/` | `utils.debug` + `debug_run.py` | Per-stage intermediate snapshots |
| `TEST_Data/available_cpgs/` | `merge_chunks.py` | Assembled test CPGs |

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install requests             # needed by app.py, missing from requirements.txt
```

Create a `.env` in the project root:

```env
DEEPSEEK_API_KEY=sk-xxx
GAOCHAO_API_KEY=sk-xxx
QWEN_API_KEY=sk-xxx
```

Only the keys for providers you actually use are required. `LLMProvider.api_key` is a lazily-evaluated property — the environment variable is only checked when that provider's LLM is constructed, and raises `EnvironmentError` if absent.

### Available providers

| name | model | base_url | env var |
|---|---|---|---|
| `deepseek` (default) | `deepseek-chat` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `gpt-4o-mini` | `gpt-4o-mini` | `https://api.ai-gaochao.cn/v1` | `GAOCHAO_API_KEY` |
| `gpt-5-mini` | `gpt-5-mini` | `https://api.ai-gaochao.cn/v1` | `GAOCHAO_API_KEY` |
| `claude-small` | `claude-3-5-haiku-20241022` | `https://api.ai-gaochao.cn/v1` | `GAOCHAO_API_KEY` |
| `qwen-long` | `qwen-long` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `QWEN_API_KEY` |

The `temperature = -1.0` on `gpt-5-mini` is a sentinel: `build_llm()` omits the `temperature` argument entirely when it sees a negative value, since reasoning models set their own.

---

## Quick start

### Option 1 — terminal debugging (start here)

```bash
python debug_run.py                            # interactive; pick a file from TEST_Data/available_cpgs/
python debug_run.py --input path/to/file.md    # explicit file
python debug_run.py --input x.md --only-tree   # stop after SmartTree is built
python debug_run.py --input x.md --resume      # reuse cached intermediates from the last run
```

After each node finishes it prints a summary and saves a snapshot. At the end you get per-node timings and LLM call counts (reasoning and generating counted separately, via LangChain `BaseCallbackHandler`).

> `--only-tree` is described in the source as "no LLM calls," but `TitleHierarchyAgent` inside the `build` node has already made LLM calls by that point. What it actually skips is the `generate` node (extraction + generation).

### Option 2 — server + web UI

```bash
python main.py            # FastAPI on 0.0.0.0:8012 by default
streamlit run app.py      # in a second terminal
```

Upload a `.md`, pick a model, click **Run Pipeline**. The frontend consumes the SSE stream, then runs `finalize_pvg_markdown()` for a final cleanup and offers a `.md` download.

### Option 3 — call the graph directly

```python
from llm import get_llm
from pipeline.graph import create_graph

graph = create_graph()
state = graph.invoke({
    "documents": ["path/to/cpg.md"],   # file paths or raw strings both work
    "reasoning_llm": get_llm("deepseek"),
    "generating_llm": get_llm("deepseek"),
    "cache_prefix": "bone_cancer",
    "is_resume": False,
})
print(state["messages"][-1])
```

Note that `state["messages"][-1]` is the **raw stitched output, not passed through `finalize_pvg_markdown()`**.

---

## How the pipeline works

### GraphState

Every node reads from and writes to one `TypedDict` (`pipeline/state.py`):

| Field | Type | Written by |
|---|---|---|
| `documents` | `List[str]` | caller (file paths or raw text) |
| `reasoning_llm` | `ChatOpenAI` | caller (used for hierarchy + extraction) |
| `generating_llm` | `ChatOpenAI` | caller (used for PVG generation) |
| `cache_prefix` | `str` | caller (snapshot filename prefix) |
| `is_resume` | `bool` | caller |
| `context` | `List[str]` | `file_process` |
| `raw_chunks` | `List[RawChunk]` | `prep` (or `build`'s resume branch) |
| `heading_candidates` | `List[HeadingCandidate]` | `prep` |
| `chunk_metadata_map` | `Dict[int, Dict]` | `build` |
| `pilot` | `List[ClassifiedTextBlock]` | `generate` (binning result) |
| `messages` | `List[str]` | `generate` (final PVG) |

### Node 1 — `file_process`

Walks `documents`: reads the file if the string is an existing path, otherwise treats it as raw text. Result goes into `context`.

### Node 2 — `prep` (SmartTreePreparation)

Returns immediately when `is_resume=True`.

`SmartSplitter.split_text()` cuts the text at heading boundaries. Heading detection (`_is_header`, in order):

1. Standard Markdown `#` through `######` → returns the level
2. All-caps line matching `^([A-Z\s\d\W]{5,80})$` not ending in `.` → level 0
3. Line ending in `:` with ≤ 7 words → level 0

These are **never** treated as headings: list-item prefixes (`-` `*` `•` `·`, or `1.` / `1)`), HTML table tags, and any line inside a Markdown table or a `<table>` block.

Afterwards, every chunk with `is_potential_header=True` or whose content starts with `#` becomes a `HeadingCandidate`, carrying either a body preview from `get_smart_preview()` (`[PREVIEW: ...]`) or the marker `[STRUCTURAL HEADER]`.

### Node 3 — `build` (SmartTreeBuilding)

Calls `TitleHierarchyAgent.classify()`, sending headings to the LLM in batches of `HIERARCHY_BATCH_SIZE=80` (JSON mode, three retries via `tenacity`), expecting `{"items": [{"id", "parent_id", "label", "keep"}]}` back.

Cross-batch parent-child links are handled by a **global navigation map**: after each batch, all root nodes plus any node labelled C or D are recorded in `history`. The next batch's prompt includes every root plus the most recent `HIERARCHY_RECENT_NON_ROOTS=20` non-roots, so a subheading in a later batch can still reference a parent chapter discovered earlier.

Candidates the LLM misses fall back to `infer_label_from_text()`, a keyword heuristic over `LABEL_KEYWORDS`.

Metadata is then built for **every** chunk: heading chunks use the LLM result directly; body chunks inherit metadata from the nearest preceding heading and get their `parent_id` pointed at it. The result lands in `chunk_metadata_map` and is snapshotted to `SmartTree_Blocks` for `--resume`.

### Node 4 — `generate` (public_version_generate)

Three steps.

**4.1 Fact extraction** — `ExtractAgent.extract()`

Runs the `CHECK_LIST` questions for each of the five categories. The key design decision: B (disease basics) and C (treatment) questions are not asked against their own label's text but against a **merged clinical pool of B+C+D** — the source comments say this solves the "B=0" problem, where B headings are rare but disease information is scattered elsewhere. A, D, and E still query only their own label's text.

When the text exceeds `EXTRACT_PASSAGE_TOKEN_LIMIT=50000`, it is segmented at `### [PATH:` boundaries, each segment queried separately, and answers merged with `dict.fromkeys` deduplication. All questions run concurrently through a `ThreadPoolExecutor` (`EXTRACTOR_MAX_WORKERS=8`).

Any answer containing `NOTFOUND` is discarded. Results are saved as `All_Facts` in both JSON and human-readable Markdown.

**4.2 Tree building + binning** — `build_tree()` + `stream_aggregate()`

`SmartTreeGraph` is an adjacency list. If a node's declared parent hasn't been inserted yet, that node is promoted to a root (best-effort recovery).

`stream_aggregate()` packs bins with a greedy DFS:

- If an entire subtree fits under `TREE_BIN_TOKEN_LIMIT=12000`, take it whole (preserves context)
- Otherwise take only the heading node and recurse into children individually
- Flush the current bin before a node that would overflow it
- Also flush between consecutive roots whose `keep` values differ

Each bin becomes a `ClassifiedTextBlock(label="STREAM_BIN")` whose metadata carries the breadcrumb `context_hint`, sibling titles, `contained_labels`, `token_count`, start/end node IDs, and `keep`.

> Token counts are character-count-divided-by-four estimates (`SmartTreeNode.self_tokens`), not a real tokenizer.

**4.3 Generation** — `PVGRunner.run()`

- Bins with `keep=False` are skipped
- `_filter_facts()` selects the facts relevant to each bin: A is always included; B and C are included based on keyword matches against `context_hint` + `siblings`; if neither matches, both are included as a safety net
- Calls to `PVGWriter.write_bin()` run concurrently through a `ThreadPoolExecutor` (`GENERATE_MAX_WORKERS=4`)
- Results are stitched in **original order** (not completion order), joined with `\n\n---\n\n`
- Every call is logged to `Invoke_Trace.md` through `InvokeTracer`
- `_final_polish()` strips the internal `### [TOPOLOGICAL LOCK]` marker

### Post-processing — `finalize_pvg_markdown()`

The single entry point for all regex cleanup in `pipeline/formatter.py`. Thirteen steps in a fixed order, wrapped in `try/except` so it returns the original text on any error and never raises:

```
_revive_mermaid          → strip # prefixes the LLM injected into mermaid blocks
_remove_meta_talk        → drop "Here is the rewritten version:" style preambles and outros
_remove_empty_markers    → drop [EMPTY CONTENT NO NEED TO OUTPUT] and similar placeholders
_demote_long_headings    → demote "headings" over 60 chars, or over 20 chars ending in .!?, to body text
_fix_heading_spaces      → normalise malformed "# # #" headings, enforce blank lines (code blocks are swapped for placeholders first)
_join_split_headings     → merge headings the LLM broke across lines
_fix_unclosed_mermaid    → close truncated mermaid fences
_fix_code_fences         → close unmatched ```
_fix_mermaid_format      → collapse nested ```mermaid fences
_fix_mermaid_style_fill  → repair "fill:\n# e1f5e1" corruption where a hex colour became a heading
_remove_duplicate_mermaid→ md5-based dedup, keeping the first occurrence of each diagram
dedupe_markdown_blocks   → (from utils.text)
_move_noisy_sections_to_appendix → fold References-style sections of ≥1500 chars into <details>
_enforce_style           → *** → ---, collapse runs of blank lines
```

**This function is currently only called from `app.py`.** `pipeline/graph.py` imports it at the top but never uses it, so the non-streaming API path and direct `graph.invoke()` both return uncleaned text.

---

## The A–E label scheme

| Label | Meaning (per `CHECK_LIST`) | Questions asked |
|---|---|---|
| A | Guideline metadata | Title, issuing organization, DOI, publication year and version |
| B | Disease basics | Definition, related health problems, risk factors, symptoms, subtypes, complications, staging and progression, epidemiology, duration |
| C | Treatment and management | Treatment/intervention recommendations, patient support and self-management, patient-provider communication and care coordination |
| D | Guideline purpose | Purpose, intended use, target users |
| E | Ancillary information | Terms and abbreviations, funding sources, conflicts of interest |

`TITLE_HIERARCHY_SYSTEM` contains hard override rules: a title containing `Words to know` / `Glossary` / `Contributors` / `Index` must be E with `parent_id=null`; `Tests` / `Diagnosis` / `Workup` must be D; `Treatment` / `Therapy` / `Management` must be C.

`keep=false` is assigned to references, author lists, tables of contents and indexes, version change logs, methodology, conflicts of interest, and appendices. Conversely, sections like `Key messages`, `Summary of recommendations`, and `Key points` — anything with actionable clinical recommendations — are forced to `keep=true`.

---

## Configuration

Everything lives in `config.py`:

| Parameter | Default | Purpose |
|---|---|---|
| `DEFAULT_LLM` | `"deepseek"` | Default provider |
| `SPLITTER_CHUNK_SIZE_LIMIT` | `2000` | Passed to `SmartSplitter` but **currently unused** |
| `HIERARCHY_BATCH_SIZE` | `80` | Heading candidates per classification LLM call |
| `HIERARCHY_RECENT_NON_ROOTS` | `20` | Recent non-root anchors kept in the navigation map |
| `EXTRACTOR_MAX_WORKERS` | `8` | Extraction concurrency |
| `EXTRACT_PASSAGE_TOKEN_LIMIT` | `50000` | Per-question text ceiling (estimated tokens) |
| `GENERATE_MAX_WORKERS` | `4` | Generation concurrency |
| `TREE_BIN_TOKEN_LIMIT` | `12000` | Token ceiling per chapter bin |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `8012` | FastAPI bind address |
| `DEBUG_SNAPSHOTS_DIR` | `"debug_snapshots"` | Snapshot root directory |

---

## API

### `POST /v1/publicversion/completions`

Request body:

```json
{
  "messages": [{"role": "user", "content": "Transform CPG to PVG"}],
  "llm_name": "deepseek",
  "documents": ["upload_files/guideline.md"],
  "to_astream": true,
  "userId": null,
  "conversationId": null,
  "cache_prefix": "guideline"
}
```

`role` accepts only `"user"` or `"PublicVersionTransformer"` (enforced by a `Literal`).

`llm_name` selects the **generating LLM**. The reasoning LLM always uses `DEFAULT_LLM`.

**`to_astream: true`** returns `text/event-stream` in an OpenAI-compatible SSE shape. Two kinds of event share one stream:

- Progress markers: `[Progress] Start: <node>` and `[Progress] Done: <node> (12.3s)` — the frontend distinguishes them with `content.startswith("[Progress]")`
- LLM tokens, only from the `generate` node's `on_chat_model_stream` events
- Termination: `finish_reason: "stop"`

**`to_astream: false`** returns a single JSON response with the PVG in `choices[0].message.content`.

---

## Debug artifacts

After a run, `debug_snapshots/<prefix>/` contains:

| File | Source | Contents |
|---|---|---|
| `00_Raw_Chunks.md` | `debug_run.py` | All raw chunks, tagged HEADER/BODY with level |
| `00_Heading_Candidates.json` | `debug_run.py` | Candidates sent to the classification LLM |
| `01_SmartTree_Index.md` | `debug_run.py` | Per-chunk label and path |
| `02_ChapterBins.md` | `debug_run.py` | Binning result with token counts, breadcrumbs, siblings |
| `03_Final_PVG.md` | `debug_run.py` | Final output |
| `SmartTree_Blocks.json` | `pipeline/graph.py` | Tree snapshot reused by `--resume` |
| `All_Facts.json` / `.md` | `pipeline/graph.py` | Facts sidebar, reused by `--resume` |
| `Invoke_Trace.md` | `utils.debug.InvokeTracer` | Prompt / payload / output for every generation call |

Suggested order when diagnosing bad output: check `02_ChapterBins.md` first to confirm the binning and `context_hint` look sane, then `All_Facts.md` to see what extraction missed, then `Invoke_Trace.md` to inspect the prompt for a specific bin.

---

## Preparing test data

```bash
python merge_chunks.py
```

Reads `TEST_Data/1/splitted_texts/*_chunks.json`, sorts by `chunk_index`, concatenates into clean Markdown, and writes to `TEST_Data/available_cpgs/<name>.md`. That directory is what `debug_run.py`'s interactive mode reads from.

---

## Known issues

These are inconsistencies found while reading the source, not speculation:

1. **`_filter_facts`'s docstring contradicts its implementation.** The docstring in `pipeline/generator.py` says A (metadata) and E (terms) are always included, but the body only handles A — **E is never added to `filtered`**. The generation stage therefore never sees the terms-and-abbreviations table.

2. **Label D has two conflicting definitions.** `prompts.CHECK_LIST["D"]` asks about the guideline's purpose, intended use, and target users, while `LABEL_KEYWORDS["D"]` and the hard override rules in `TITLE_HIERARCHY_SYSTEM` define D as diagnosis / tests / workup. Sections labelled D end up answering an unrelated set of questions.

3. **The SSE token stream may never fire.** `main.py` depends on `on_chat_model_stream` events, but `llm.build_llm()` defaults `streaming` to `False` and `get_llm()` never passes `True`. Content still reaches the frontend through other means, but real-time streaming doesn't happen.

4. **`finalize_pvg_markdown()` only runs on the Streamlit path** (see the post-processing section). The non-streaming API response and `graph.invoke()` both return uncleaned text.

5. **`config = {"configurable": {"thread_id": ...}}` has no effect.** `create_graph()` compiles without a checkpointer, so `thread_id` persists nothing.

6. **`SPLITTER_CHUNK_SIZE_LIMIT` is dead.** `SmartSplitter.__init__` stores `self.chunk_size_limit` but `split_text()` never reads it — chunk length is determined entirely by heading boundaries, so an oversized section is never force-split.

7. **`requirements.txt` is missing `requests`** (imported directly by `app.py`).

8. **`tasks` in `generator.py` is annotated as a 5-tuple but 6-tuples are appended.** Harmless at runtime, misleading to read.

9. **`_implied_header` is too permissive.** The `\W` in `^([A-Z\s\d\W]{5,80})$` matches every non-word character, so punctuation-only lines, digit-only lines, and horizontal rules all classify as headings.
