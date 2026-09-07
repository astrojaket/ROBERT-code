"""Sampler adapters for retrieval problems."""

from .base import NestedSamplerResult
from .multinest import run_multinest

__all__ = ["NestedSamplerResult", "run_multinest"]
