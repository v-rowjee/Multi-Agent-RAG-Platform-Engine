"""Stable public facade for business-intelligence API callers."""

from app.core.exceptions import (
    DatasetAlreadyExistsError,
    InvalidUploadError,
    SessionNotFoundError,
)
from app.services.analysis.facade import BusinessIntelligenceService as _Facade
from app.services.analysis.models import PipelineExecution

# One graph-override seam for tests and supported integrations. The facade
# resolves this lazily, so importing the API does not compile the graph.
analysis_graph = None
BusinessIntelligenceService = _Facade


business_intelligence_service = BusinessIntelligenceService()

__all__ = [
    "BusinessIntelligenceService",
    "DatasetAlreadyExistsError",
    "InvalidUploadError",
    "PipelineExecution",
    "SessionNotFoundError",
    "business_intelligence_service",
]
