"""Safety and shape tests for the Smith et al. (2024) HRS inputs."""

from __future__ import annotations

import csv
from pathlib import Path
import pickle

import numpy as np
import pytest

from robert_exoplanets.core import RobertDataError
from robert_exoplanets.io import (
    SMITH2024_WASP77AB_CUBE14_SHA256,
    SMITH2024_WASP77AB_N_FRAMES,
    SMITH2024_WASP77AB_N_ORDERS,
    SMITH2024_WASP77AB_N_PIXELS,
    load_smith2024_wasp77ab_hrs,
    load_smith2024_wasp77ab_template,
)


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external_data" / "wasp77ab_smith2024"


def _info_file(path: Path, n_frames: int = 2) -> None:
    """Write a minimal frame CSV for unverified unit-test payloads."""

    columns = (
        "Time (BJD)",
        "Phase",
        "RV [km/s]",
        "Air Mass",
        "Humidity [% dew point]",
        "Med. SNR",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(n_frames):
            writer.writerow(
                (
                    2_459_197.5 + index,
                    0.3 + index * 0.01,
                    23.0 + index,
                    1.2,
                    35.0,
                    180.0,
                )
            )


def test_loader_rejects_a_forbidden_pickle_global_before_array_use(tmp_path: Path) -> None:
    class Unsafe:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    cube_path = tmp_path / "cube14v3.pic"
    info_path = tmp_path / "20201214_info.csv"
    with cube_path.open("wb") as stream:
        pickle.dump(Unsafe(), stream, protocol=2)
    _info_file(info_path)

    with pytest.raises(RobertDataError, match="safely load"):
        load_smith2024_wasp77ab_hrs(
            cube_path,
            info_path=info_path,
            verify_checksum=False,
        )


def test_loader_checks_size_and_md5_before_unpickling(tmp_path: Path) -> None:
    cube_path = tmp_path / "cube14v3.pic"
    cube_path.write_bytes(b"not a Smith cube")
    with pytest.raises(RobertDataError, match="size mismatch"):
        load_smith2024_wasp77ab_hrs(cube_path)


def test_loader_rejects_payload_with_wrong_shape_after_safe_load(tmp_path: Path) -> None:
    cube_path = tmp_path / "cube14v3.pic"
    info_path = tmp_path / "20201214_info.csv"
    wavelengths = np.ones((1, 4))
    flux = np.ones((1, 2, 4))
    with cube_path.open("wb") as stream:
        pickle.dump([wavelengths, flux], stream, protocol=2)
    _info_file(info_path)

    with pytest.raises(RobertDataError, match="order wavelengths must have shape"):
        load_smith2024_wasp77ab_hrs(
            cube_path,
            info_path=info_path,
            verify_checksum=False,
        )


def test_real_smith_cube_load_is_optional_and_immutable() -> None:
    cube_path = EXTERNAL / "cube14v3.pic"
    info_path = EXTERNAL / "20201214_info.csv"
    if not cube_path.is_file() or not info_path.is_file():
        pytest.skip("Smith et al. external files are not available")

    observation = load_smith2024_wasp77ab_hrs(
        cube_path,
        info_path=info_path,
        verify_sha256=True,
    )
    assert observation.n_orders == SMITH2024_WASP77AB_N_ORDERS
    assert observation.n_frames == SMITH2024_WASP77AB_N_FRAMES
    assert observation.n_pixels == SMITH2024_WASP77AB_N_PIXELS
    assert observation.metadata["cube_sha256"] == SMITH2024_WASP77AB_CUBE14_SHA256
    assert observation.order_wavelengths.flags.writeable is False
    assert observation.flux.flags.writeable is False
    assert observation.mask.flags.writeable is False
    with pytest.raises(ValueError):
        observation.flux[0, 0, 0] = 0.0


def test_real_full_template_loader_is_optional_and_strict() -> None:
    template_path = EXTERNAL / "w77_1DRC_FULL.txt"
    if not template_path.is_file():
        pytest.skip("Smith et al. external template is not available")
    template = load_smith2024_wasp77ab_template(
        template_path,
        verify_sha256=True,
    )
    assert template.wavelength.size > 100_000
    assert template.planet_flux.size == template.wavelength.size
    assert template.stellar_flux.size == template.wavelength.size
    assert template.wavelength.flags.writeable is False
    assert template.metadata["template_md5"]
