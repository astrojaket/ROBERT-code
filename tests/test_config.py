"""Tests for retrieval configuration objects."""

from __future__ import annotations

import pytest

from robert_exoplanets import RetrievalConfig


def test_retrieval_config_has_default_parameters() -> None:
    config = RetrievalConfig(target_name="WASP-43b", instrument="JWST/MIRI LRS")

    assert config.target_name == "WASP-43b"
    assert config.instrument == "JWST/MIRI LRS"
    assert "temperature" in config.parameters


def test_retrieval_config_requires_target_name() -> None:
    with pytest.raises(ValueError, match="target_name"):
        RetrievalConfig(target_name="", instrument="JWST/NIRSpec")


def test_retrieval_config_requires_instrument() -> None:
    with pytest.raises(ValueError, match="instrument"):
        RetrievalConfig(target_name="WASP-43b", instrument="")
