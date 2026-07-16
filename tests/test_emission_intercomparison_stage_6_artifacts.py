"""Integrity checks for the versioned Stage-6 emission artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/emission_intercomparison"
REPORT = DATA / "stage_6_report.json"
TENSORS = DATA / "stage_6_response_tensors.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_stage_6_report_and_tensor_checksums_are_current() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    checksums = json.loads((DATA / "checksums.json").read_text(encoding="utf-8"))

    assert report["stage"] == 6
    assert report["status"] == "passed"
    assert all(
        gate["passed"]
        for gate in report["tracks"]["track_a_shared_tau"][
            "cross_code_acceptance_gates"
        ].values()
    )
    assert report["response_artifact"]["sha256"] == _sha256(TENSORS)
    assert checksums[REPORT.name] == _sha256(REPORT)
    assert checksums[TENSORS.name] == _sha256(TENSORS)


def test_stage_6_tensor_shapes_normalization_and_finiteness() -> None:
    with np.load(TENSORS, allow_pickle=False) as artifact:
        assert artifact["profile_name"].tolist() == [
            "isothermal",
            "monotonic",
            "inverted",
            "retrieved_like",
        ]
        assert artifact["species_name"].tolist() == ["H2O", "CO", "CO2", "CH4"]
        assert artifact["wavelength_r100_micron"].shape == (315,)
        for resolution in (40, 80, 160):
            for track in ("track_a_shared_tau", "track_b_native_opacity"):
                for model in ("robert", "picaso", "petitradtrans"):
                    suffix = f"{track}_L{resolution}_{model}"
                    jacobian = artifact[f"flux_jacobian_{suffix}_w_m2_m_dex"]
                    response = artifact[f"normalized_vertical_response_{suffix}"]
                    fractions = artifact[f"cross_species_sensitivity_fraction_{suffix}"]
                    assert jacobian.shape == response.shape == (4, 4, 6, 315)
                    assert fractions.shape == (4, 4, 315)
                    assert np.all(np.isfinite(jacobian))
                    assert np.all(np.isfinite(response))
                    assert np.all(np.isfinite(fractions))
                    response_sum = np.sum(response, axis=2)
                    fraction_sum = np.sum(fractions, axis=1)
                    assert np.all(
                        np.isclose(response_sum, 0.0) | np.isclose(response_sum, 1.0)
                    )
                    assert np.all(
                        np.isclose(fraction_sum, 0.0) | np.isclose(fraction_sum, 1.0)
                    )
