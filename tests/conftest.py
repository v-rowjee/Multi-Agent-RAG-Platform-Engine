from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Production requires an explicit mode. Tests use the normal multi-agent mode
# unless a test intentionally overrides or removes it.
os.environ.setdefault("BI_PIPELINE_MODE", "multi")
