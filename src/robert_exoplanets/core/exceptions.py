"""ROBERT exception hierarchy."""

from __future__ import annotations


class RobertError(Exception):
    """Base class for ROBERT errors."""


class RobertConfigError(RobertError):
    """Raised when user configuration is invalid."""


class RobertDataError(RobertError):
    """Raised when input scientific data are invalid."""


class RobertCoverageError(RobertError):
    """Raised when requested model state is outside data coverage."""


class RobertValidationError(RobertError, ValueError):
    """Raised when a domain object violates its invariants."""
