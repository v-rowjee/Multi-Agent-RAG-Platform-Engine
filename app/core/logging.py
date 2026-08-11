"""Application logging configuration."""

from __future__ import annotations

import logging


def configure_logging() -> None:
    """Configure process logging once using a conservative default format."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
