"""Smoke-test one isolated emission-intercomparison runtime.

Run this script separately from the ROBERT, PICASO, and stable-pRT Conda
environments.  The external checks execute a small thermal calculation and
read the locally installed opacity data; they do not import ROBERT.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PICASO_REFERENCE = (
    REPOSITORY / "opacity_data/picaso_official/reference_v3_2"
)
DEFAULT_PICASO_DATABASE = (
    REPOSITORY
    / "opacity_data/picaso_official/reference/opacities"
    / "opacities_0.3_15_R15000.db"
)
DEFAULT_PRT_INPUT_DATA = REPOSITORY / "opacity_data/petitRADTRANS/input_data"


def _check_robert() -> dict[str, object]:
    from robert_exoplanets.diagnostics import planck_radiance_wavelength

    radiance = planck_radiance_wavelength(np.array([1.0, 5.0, 10.0]), 1000.0)
    return {
        "code": "ROBERT",
        "version": importlib.metadata.version("robert-exoplanets"),
        "finite_positive_planck_radiance": bool(
            np.isfinite(radiance).all() and np.all(radiance > 0.0)
        ),
    }


def _check_picaso(reference: Path, database: Path) -> dict[str, object]:
    if not reference.is_dir():
        raise FileNotFoundError(f"PICASO reference data not found: {reference}")
    if not database.is_file():
        raise FileNotFoundError(f"PICASO opacity database not found: {database}")
    cache_root = Path(tempfile.gettempdir()) / "robert-picaso-cache"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ["picaso_refdata"] = str(reference.resolve())

    import scipy
    from picaso import justdoit as jdi
    from picaso.fluxes import get_thermal_1d

    opacity = jdi.opannection(
        filename_db=str(database.resolve()),
        wave_range=[1.0, 1.1],
        resample=100,
        verbose=False,
    )
    wavenumber = np.linspace(9900.0, 10000.0, 8)
    pressure_cgs = np.array([0.0, 1.0e2, 1.0e4, 1.0e6, 1.0e8])
    temperature_k = np.full(5, 1000.0)
    optical_depth = np.full((4, 8), 0.2)
    point_intensity, _ = get_thermal_1d(
        5,
        wavenumber,
        8,
        2,
        1,
        temperature_k,
        optical_depth,
        np.full_like(optical_depth, 1.0e-10),
        np.zeros_like(optical_depth),
        pressure_cgs,
        np.array([[0.25], [0.75]]),
        np.zeros(8),
        1,
        np.zeros(8),
        0,
    )
    return {
        "code": "PICASO",
        "version": importlib.metadata.version("picaso"),
        "scipy_version": scipy.__version__,
        "queried_opacity_points": int(opacity.nwno),
        "finite_positive_thermal_intensity": bool(
            np.isfinite(point_intensity).all() and np.all(point_intensity > 0.0)
        ),
    }


def _check_petitradtrans(input_data: Path) -> dict[str, object]:
    if not input_data.is_dir():
        raise FileNotFoundError(f"pRT input data not found: {input_data}")
    os.environ.setdefault("OMPI_MCA_btl", "self")

    import petitRADTRANS
    from petitRADTRANS.radtrans import Radtrans

    pressure_bar = np.geomspace(1.0e-5, 100.0, 8)
    atmosphere = Radtrans(
        pressures=pressure_bar,
        wavelength_boundaries=np.array([1.0, 1.05]),
        line_species=["H2O__POKAZATEL"],
        scattering_in_emission=False,
        path_input_data=str(input_data.resolve()),
    )
    wavelength_cm, flux_cgs, _ = atmosphere.calculate_flux(
        temperatures=np.full(8, 1000.0),
        mass_fractions={"H2O__POKAZATEL": np.full(8, 1.0e-3)},
        mean_molar_masses=np.full(8, 2.3),
        reference_gravity=1500.0,
        frequencies_to_wavelengths=True,
    )
    return {
        "code": "petitRADTRANS",
        "version": petitRADTRANS.__version__,
        "wavelength_points": int(np.size(wavelength_cm)),
        "finite_positive_thermal_flux": bool(
            np.isfinite(flux_cgs).all() and np.all(flux_cgs > 0.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", choices=("robert", "picaso", "petitradtrans"))
    parser.add_argument(
        "--picaso-reference", type=Path, default=DEFAULT_PICASO_REFERENCE
    )
    parser.add_argument(
        "--picaso-database", type=Path, default=DEFAULT_PICASO_DATABASE
    )
    parser.add_argument("--prt-input-data", type=Path, default=DEFAULT_PRT_INPUT_DATA)
    args = parser.parse_args()

    if args.code == "robert":
        result = _check_robert()
    elif args.code == "picaso":
        result = _check_picaso(args.picaso_reference, args.picaso_database)
    else:
        result = _check_petitradtrans(args.prt_input_data)
    if not all(value is not False for value in result.values()):
        raise RuntimeError(f"environment smoke test failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
