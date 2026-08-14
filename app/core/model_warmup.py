"""Optional best-effort warm-up for local inference models.

Web processes must not call this during ASGI startup: model residency is too
large for small service instances. It is retained for explicit worker/CLI use.
"""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Callable

from app.rag.embeddings.service import get_embedding_service
from app.rag.retrieval.reranker import get_reranker
from app.services.forecasting.chronos import chronos_service

logger = logging.getLogger(__name__)


async def warm_local_models() -> None:
    """Warm local models sequentially when explicitly requested."""
    models: tuple[tuple[str, Callable[[], None]], ...] = (
        ("chronos", chronos_service.warm),
        ("embedding", get_embedding_service().warm),
        ("reranker", get_reranker().warm),
    )
    for name, warm in models:
        started_at = perf_counter()
        try:
            await asyncio.to_thread(warm)
        except Exception:
            logger.warning(
                "Local model warm-up failed model=%s latency_ms=%.1f; "
                "the model will be retried on first use.",
                name,
                (perf_counter() - started_at) * 1000,
                exc_info=True,
            )
        else:
            logger.info(
                "Local model warm-up completed model=%s latency_ms=%.1f",
                name,
                (perf_counter() - started_at) * 1000,
            )
