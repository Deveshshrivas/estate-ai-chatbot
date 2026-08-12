"""Embedding retrieval and evidence helpers for grounded chatbot answers."""

from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import get_settings
from .database import knowledge_base


settings = get_settings()
_query_cache: dict[str, list[float]] = {}
_embedding_unavailable_until = 0.0


def knowledge_text(document: dict[str, Any]) -> str:
    keywords = document.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = [str(keywords)]
    return "\n".join(filter(None, [
        str(document.get("title") or ""),
        str(document.get("category") or ""),
        " ".join(map(str, keywords)),
        str(document.get("content") or ""),
    ])).strip()


def content_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(knowledge_text(document).encode("utf-8")).hexdigest()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings through OpenRouter's OpenAI-compatible endpoint."""
    if not texts:
        return []
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.app_url,
        "X-OpenRouter-Title": settings.app_name,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers,
            json={"model": settings.openrouter_embedding_model, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()
    rows = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
    vectors = [row.get("embedding") for row in rows]
    if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
        raise RuntimeError("Embedding provider returned an incomplete response")
    return vectors


async def refresh_knowledge_embeddings(batch_size: int = 24) -> int:
    """Embed new or edited MongoDB knowledge documents without touching valid vectors."""
    pending: list[dict[str, Any]] = []
    cursor = knowledge_base.find({"active": {"$ne": False}})
    async for document in cursor:
        digest = content_hash(document)
        if (
            document.get("embedding_model") != settings.openrouter_embedding_model
            or document.get("embedding_hash") != digest
            or not document.get("embedding")
        ):
            document["_embedding_hash"] = digest
            pending.append(document)

    updated = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        vectors = await embed_texts([knowledge_text(document) for document in batch])
        for document, vector in zip(batch, vectors):
            await knowledge_base.update_one(
                {"_id": document["_id"]},
                {"$set": {
                    "embedding": vector,
                    "embedding_model": settings.openrouter_embedding_model,
                    "embedding_hash": document["_embedding_hash"],
                    "embedded_at": datetime.now(timezone.utc),
                }},
            )
            updated += 1
    return updated


async def semantic_knowledge_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return only knowledge records that clear the configured similarity threshold."""
    normalized = " ".join(query.split())[:4000]
    if not normalized:
        return []
    global _embedding_unavailable_until
    if time.monotonic() < _embedding_unavailable_until:
        return []
    try:
        await refresh_knowledge_embeddings()
        cache_key = f"{settings.openrouter_embedding_model}:{normalized}"
        query_vector = _query_cache.get(cache_key)
        if query_vector is None:
            query_vector = (await embed_texts([normalized]))[0]
            if len(_query_cache) >= 500:
                _query_cache.pop(next(iter(_query_cache)))
            _query_cache[cache_key] = query_vector
    except Exception:
        _embedding_unavailable_until = time.monotonic() + 300
        # The caller uses MongoDB lexical retrieval as a grounded fallback.
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    cursor = knowledge_base.find(
        {
            "active": {"$ne": False},
            "embedding_model": settings.openrouter_embedding_model,
            "embedding": {"$exists": True},
        }
    ).limit(1000)
    async for document in cursor:
        score = cosine_similarity(query_vector, document.get("embedding") or [])
        if score >= settings.knowledge_similarity_threshold:
            clean = {
                key: value for key, value in document.items()
                if key not in {"_id", "embedding", "embedding_hash"}
            }
            clean["retrieval_score"] = round(score, 4)
            clean["retrieval_method"] = "embedding"
            scored.append((score, clean))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:limit]]


def evidence_answer(documents: list[dict[str, Any]]) -> str:
    """Return source text directly, so an LLM cannot add unsupported facts."""
    if not documents:
        return ""
    document = documents[0]
    content = str(document.get("content") or "").strip()
    source = str(document.get("source") or "").strip()
    if not content:
        return ""
    return f"{content[:700]}" + (f" Source: {source}." if source else "")
