"""Best-effort startup warm-up for local inference models."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Callable

from app.rag.embeddings.service import get_embedding_service
from app.services.forecasting.chronos import chronos_service

logger = logging.getLogger(__name__)


async def warm_local_models() -> None:
    """Warm local models sequentially; an unavailable model never blocks startup."""
    models: tuple[tuple[str, Callable[[], None]], ...] = (
        ("chronos", chronos_service.warm),
        ("embedding", get_embedding_service().warm),
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
