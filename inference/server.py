from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference.context_builder import build_context, extract_source_ids
from inference.domain_gate import DomainGate
from inference.prompt import build_chat_messages, format_for_completion
from inference.retriever import Retriever
from inference.validator import validate_response

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/app/config.yaml")
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else LOCAL_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state["config"] = cfg

    _state["retriever"] = Retriever(
        index_path=cfg["faiss"]["index_path"],
        metadata_path=cfg["faiss"]["metadata_path"],
        embedding_model_name=cfg["embedding"]["model_name"],
        top_k=cfg["retrieval"]["top_k"],
        relevance_threshold=cfg["retrieval"]["relevance_threshold"],
    )

    _state["domain_gate"] = DomainGate(
        centroid_path=cfg["domain_gate"]["centroid_path"],
        similarity_threshold=cfg["domain_gate"]["similarity_threshold"],
    )

    model_path = cfg["model"]["path"]
    if Path(model_path).exists():
        from llama_cpp import Llama

        _state["llm"] = Llama(
            model_path=model_path,
            n_ctx=cfg["model"].get("n_ctx", 2048),
            n_gpu_layers=cfg["model"].get("n_gpu_layers", 0),
            verbose=False,
        )
        logger.info("LLM loaded from %s", model_path)
    else:
        _state["llm"] = None
        logger.warning(
            "Model not found at %s — running in retrieval-only mode", model_path
        )

    logger.info("Server ready — business: %s", cfg["business_name"])
    yield
    _state.clear()


app = FastAPI(title="BabyYoday Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    domain_score: float
    latency_ms: float
    grounded: bool


class ErrorResponse(BaseModel):
    error: str
    domain_score: float | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "business": _state.get("config", {}).get("business_name", "unknown"),
        "model_loaded": _state.get("llm") is not None,
        "index_size": _state["retriever"].index.ntotal
        if "retriever" in _state
        else 0,
    }


@app.post("/query", response_model=QueryResponse | ErrorResponse)
def query(req: QueryRequest):
    t0 = time.time()
    cfg = _state["config"]
    retriever: Retriever = _state["retriever"]
    gate: DomainGate = _state["domain_gate"]

    query_embedding = retriever.embed_query(req.query)
    allowed, similarity = gate.check(query_embedding)

    if not allowed:
        return ErrorResponse(
            error=(
                f"I can only help with questions about "
                f"{cfg['business_name']}. How can I help with that?"
            ),
            domain_score=similarity,
        )

    chunks = retriever.search(req.query)

    if not chunks:
        return ErrorResponse(
            error="I don't have information on that topic.",
            domain_score=similarity,
        )

    context = build_context(chunks)
    source_ids = extract_source_ids(chunks)

    llm = _state.get("llm")
    if llm is None:
        answer_text = (
            f"[Retrieval-only mode] Found {len(chunks)} relevant chunks. "
            f"Sources: {', '.join(source_ids)}"
        )
    else:
        prompt = format_for_completion(
            cfg["business_name"], context, req.query
        )
        result = llm(
            prompt,
            max_tokens=512,
            temperature=cfg["model"].get("temperature", 0.3),
            stop=["\n\nQuestion:", "\n\n---"],
        )
        answer_text = result["choices"][0]["text"].strip()

    validation = validate_response(answer_text, source_ids)

    latency = (time.time() - t0) * 1000
    return QueryResponse(
        answer=validation.answer,
        sources=[
            {
                "id": c.source_id,
                "name": c.source_name,
                "score": round(c.score, 3),
            }
            for c in chunks
        ],
        domain_score=round(similarity, 3),
        latency_ms=round(latency, 1),
        grounded=validation.is_valid,
    )
