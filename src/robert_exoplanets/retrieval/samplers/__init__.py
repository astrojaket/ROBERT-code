"""Sampler adapters for retrieval problems."""

from .base import NestedSamplerResult
from .ultranest import run_ultranest

__all__ = ["NestedSamplerResult", "run_ultranest"]
