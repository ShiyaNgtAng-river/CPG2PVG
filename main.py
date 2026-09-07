"""
FastAPI server for the CPG → PVG pipeline.

    POST /v1/publicversion/completions
        - streaming (SSE)  : returns real-time token events
        - non-streaming    : returns the complete PVG in one JSON response
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import SERVER_HOST, SERVER_PORT
from llm import get_llm
from pipeline.graph import create_graph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "PublicVersionTransformer"]
    content: str


class CompletionRequest(BaseModel):
    messages: List[Message]
    llm_name: str
    documents: List[str]
    to_astream: Optional[bool] = True
    userId: Optional[str] = None
    conversationId: Optional[str] = None
    cache_prefix: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "publicversion.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[Choice]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

graph = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global graph
    logger.info("Compiling LangGraph workflow ...")
    graph = create_graph()
    logger.info("Graph ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(lifespan=lifespan)


def _thread_id(user: Optional[str], conv: Optional[str]) -> str:
    u = re.sub(r"[^A-Za-z0-9_.@-]+", "_", (user or "anon").strip())[:64]
    c = re.sub(r"[^A-Za-z0-9_.@-]+", "_", (conv or "default").strip())[:64]
    return f"{u}@@{c}"


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse_chunk(chunk_id: str, content: str = "", finish: str | None = None):
    """Format one SSE ``data:`` line in OpenAI-compatible shape."""
    delta = {"content": content} if content else {}
    return (
        f"data: {json.dumps({'id': chunk_id, 'object': 'publicversion.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/publicversion/completions")
async def completions(request: CompletionRequest):
    if graph is None:
        raise HTTPException(500, "Service not initialised.")

    generating_llm = get_llm(request.llm_name)
    reasoning_llm = get_llm()  # always uses default provider for reasoning

    thread_id = _thread_id(request.userId, request.conversationId)
    config = {"configurable": {"thread_id": thread_id}}

    prefix = request.cache_prefix or f"api_{thread_id}_{int(time.time())}"
    prefix = re.sub(r"[^A-Za-z0-9_.@-]+", "_", prefix)[:96]

    input_state = {
        "messages": [],
        "documents": request.documents,
        "context": [],
        "pilot": [],
        "reasoning_llm": reasoning_llm,
        "generating_llm": generating_llm,
        "cache_prefix": prefix,
    }

    # ---- streaming (SSE) ----
    if request.to_astream:

        async def _stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            start = time.time()
            last_node = None

            try:
                async for event in graph.astream_events(input_state, config):
                    etype = event["event"]
                    node = event["metadata"].get("langgraph_node", "")

                    # Progress markers
                    if etype == "on_chain_start" and node and node != last_node:
                        yield _sse_chunk(chunk_id, f"[Progress] Start: {node}\n")
                        last_node = node
                    elif etype == "on_chain_end" and node:
                        elapsed = round(time.time() - start, 1)
                        yield _sse_chunk(chunk_id, f"[Progress] Done: {node} ({elapsed}s)\n")

                    # Stream LLM tokens from the generate node
                    if etype == "on_chat_model_stream" and node == "generate":
                        data = event.get("data", {})
                        chunk_obj = data.get("chunk")
                        token = ""
                        if hasattr(chunk_obj, "content"):
                            token = chunk_obj.content
                        elif isinstance(chunk_obj, dict):
                            token = chunk_obj.get("content", "")
                        if token:
                            yield _sse_chunk(chunk_id, token)

                yield _sse_chunk(chunk_id, finish="stop")
            except Exception as exc:
                logger.error("Streaming error: %s", exc, exc_info=True)
                yield _sse_chunk(chunk_id, f"\n\nError: {exc}", finish="error")

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # ---- non-streaming ----
    try:
        result_text = ""
        for event in graph.stream(input_state, config):
            for value in event.values():
                if isinstance(value, dict) and "messages" in value:
                    msgs = value["messages"]
                    if msgs:
                        result_text = msgs[-1]

        response = CompletionResponse(choices=[
            Choice(
                index=0,
                message=Message(role="PublicVersionTransformer", content=str(result_text)),
                finish_reason="stop",
            ),
        ])
        return JSONResponse(content=response.model_dump())

    except Exception as exc:
        logger.error("Workflow error: %s", exc, exc_info=True)
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
