"""Integrity checks for the versioned Stage-7 emission artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/emission_intercomparison"
REPORT = DATA / "stage_7_report.json"
ARRAYS = DATA / "stage_7_absorbing_cloud_arrays.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unpack_uint12(packed: np.ndarray, shape_record: np.ndarray) -> np.ndarray:
    data = np.asarray(packed, dtype=np.uint8)
    left = data[0::3].astype(np.uint16) | (
        (data[1::3].astype(np.uint16) & 0x0F) << 8
    )
    right = (data[1::3].astype(np.uint16) >> 4) | (
        data[2::3].astype(np.uint16) << 4
    )
    interleaved = np.empty(left.size * 2, dtype=np.uint16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    original_size = int(shape_record[-1])
    shape = tuple(int(value) for value in shape_record[:-1])
    return interleaved[:original_size].reshape(shape).astype(float) / 4095.0


def test_stage_7_report_and_array_checksums_are_current() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    checksums = json.loads((DATA / "checksums.json").read_text(encoding="utf-8"))

    assert report["stage"] == 7
    assert report["status"] == "failed_track_a_gates"
    gates = report["tracks"]["track_a_shared_tau"]["acceptance_gates"]
    assert gates["omega0_max_abs"]["passed"]
    assert gates["isothermal_cloud_effect_max_abs_ppm"]["passed"]
    assert not gates["primary_absolute_spectrum_p95_symmetric_relative"]["passed"]
    assert report["cloud_array_artifact"]["sha256"] == _sha256(ARRAYS)
    assert checksums[REPORT.name] == _sha256(REPORT)
    assert checksums[ARRAYS.name] == _sha256(ARRAYS)


def test_stage_7_arrays_preserve_complete_shapes_and_absorption_contract() -> None:
    with np.load(ARRAYS, allow_pickle=False) as artifact:
        assert artifact["profile_name"].tolist() == [
            "isothermal",
            "monotonic",
            "inverted",
            "retrieved_like",
        ]
        assert artifact["cloud_label"].size == 50
        assert artifact["wavelength_r100_micron"].shape == (315,)
        assert np.all(artifact["single_scattering_albedo"] == 0.0)
        assert float(artifact["normalized_profile_quantization_scale"]) == 4095.0
        for resolution in (40, 80, 160):
            assert artifact[f"case_id_L{resolution}"].shape == (200,)
            for track in ("track_a_shared_tau", "track_b_native_cloud"):
                for model in ("robert", "picaso", "petitradtrans"):
                    suffix = f"{track}_L{resolution}_{model}"
                    flux = artifact[f"flux_{suffix}_w_m2_m"]
                    effect = artifact[f"cloud_effect_eclipse_{suffix}_ppm"]
                    assert flux.shape == effect.shape == (200, 315)
                    assert np.all(np.isfinite(flux))
                    assert np.all(np.isfinite(effect))
            native_suffix = f"track_b_native_cloud_L{resolution}_robert"
            shape = artifact[f"normalized_contribution_{native_suffix}_shape"]
            assert tuple(shape[:-1]) == (200, resolution, 315)
            shared_suffix = (
                f"track_a_shared_tau_L{resolution}_identical_formal_profile"
            )
            shared_shape = artifact[f"normalized_contribution_{shared_suffix}_shape"]
            assert tuple(shared_shape[:-1]) == (200, resolution, 315)
            assert artifact[
                f"cloud_extinction_tau_track_a_shared_tau_L{resolution}"
            ].shape == (50, resolution, 315)


def test_stage_7_packed_profiles_decode_to_normalized_tensors() -> None:
    with np.load(ARRAYS, allow_pickle=False) as artifact:
        suffix = "track_a_shared_tau_L40_identical_formal_profile"
        contribution = _unpack_uint12(
            artifact[f"normalized_contribution_{suffix}_uint12_packed"],
            artifact[f"normalized_contribution_{suffix}_shape"],
        )
        response = _unpack_uint12(
            artifact[f"normalized_cloud_response_{suffix}_uint12_packed"],
            artifact[f"normalized_cloud_response_{suffix}_shape"],
        )

    contribution_sum = np.sum(contribution, axis=1)
    response_sum = np.sum(response, axis=1)
    assert np.allclose(contribution_sum, 1.0, atol=40.0 / 4095.0)
    assert np.all(
        np.isclose(response_sum, 0.0, atol=40.0 / 4095.0)
        | np.isclose(response_sum, 1.0, atol=40.0 / 4095.0)
    )
