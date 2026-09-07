#Test V3
"""
Streamlit UI for the CPG → PVG pipeline.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from typing import List

import requests
import streamlit as st

from config import LLM_PROVIDERS, SERVER_PORT
from pipeline.formatter import finalize_pvg_markdown

logger = logging.getLogger(__name__)

BACKEND_URL = f"http://localhost:{SERVER_PORT}/v1/publicversion/completions"
UPLOAD_DIR = Path("upload_files")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CPG → PVG Transformer",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
.progress-msg {
    font-family: 'Courier New', monospace;
    font-size: 0.82em;
    color: #888;
    opacity: 0.6;
    padding: 4px 8px;
    border-left: 3px solid cornflowerblue;
    margin: 2px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("LLM Model", list(LLM_PROVIDERS.keys()))
    st.divider()
    st.caption("Upload one or more `.md` files, then click **Run**.")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

st.title("CPG → PVG Transformer")

uploaded_files = st.file_uploader(
    "Upload Clinical Practice Guidelines (.md)",
    type=["md"],
    accept_multiple_files=True,
)


def _save_upload(file) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / file.name
    target.write_bytes(file.getbuffer())
    return str(target)


# ---------------------------------------------------------------------------
# Main action
# ---------------------------------------------------------------------------

if st.button("Run Pipeline", type="primary", disabled=not uploaded_files):
    doc_paths: List[str] = [_save_upload(f) for f in uploaded_files]
    prefix = re.sub(r"\W+", "_", uploaded_files[0].name.rsplit(".", 1)[0])[:48]

    payload = {
        "messages": [{"role": "user", "content": "Transform CPG to PVG"}],
        "llm_name": model_name,
        "documents": doc_paths,
        "to_astream": True,
        "cache_prefix": prefix,
    }

    progress_area = st.empty()
    output_area = st.empty()
    full_text = ""

    try:
        with requests.post(BACKEND_URL, json=payload, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                finish = choices[0].get("finish_reason")

                if finish == "stop":
                    break

                # Progress messages
                if content.startswith("[Progress]"):
                    progress_area.markdown(
                        f'<div class="progress-msg">{content.strip()}</div>',
                        unsafe_allow_html=True,
                    )
                    continue

                full_text += content
                output_area.markdown(full_text, unsafe_allow_html=True)

        # Final polish
        progress_area.empty()
        if full_text.strip():
            polished = finalize_pvg_markdown(full_text)
            output_area.markdown(polished, unsafe_allow_html=True)

            # Download button
            buf = BytesIO(polished.encode("utf-8"))
            st.download_button(
                "Download PVG (.md)",
                data=buf,
                file_name=f"{prefix}_PVG.md",
                mime="text/markdown",
            )
        else:
            st.warning("No output was generated. Check the backend logs.")

    except requests.ConnectionError:
        st.error(
            f"Cannot connect to backend at {BACKEND_URL}. "
            f"Start it first with: `python main.py`"
        )
    except Exception as exc:
        st.error(f"Error: {exc}")
