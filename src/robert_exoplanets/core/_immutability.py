"""Internal helpers for immutable validated domain objects."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, TypeVar

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


def immutable_mapping(values: Mapping[_Key, _Value]) -> Mapping[_Key, _Value]:
    """Return a read-only copy of a mapping."""

    return MappingProxyType(dict(values))
