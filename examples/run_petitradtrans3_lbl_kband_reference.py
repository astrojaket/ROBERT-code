"""Create the stable petitRADTRANS R=1e6 CO/H2O K-band oracle.

Run this file in the ``petitradtrans-stable`` Conda environment.  The opacity
files stay external to Git.  The script reads only the requested narrow HDF5
hyperslabs and refuses configurations whose conservative memory estimate is
above the configured ceiling.

The public composition inputs are H2O and CO volume mixing ratios (VMR).  The
complete VMR state is converted to mass fractions only at the private
``Radtrans.calculate_flux`` boundary.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import tempfile
from time import perf_counter
from typing import Iterator

# Set these limits before NumPy, BLAS, or the pRT extension modules load.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    os.environ[_thread_variable] = "3"
os.environ.setdefault("OMPI_MCA_btl", "self")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
try:  # noqa: E402
    import petitRADTRANS
    from petitRADTRANS.radtrans import Radtrans
except ModuleNotFoundError:  # Keep VMR helpers testable in the ROBERT env.
    petitRADTRANS = None
    Radtrans = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "external_data" / "petitRADTRANS" / "input_data"
DEFAULT_OUTPUT = (
    ROOT
    / "examples"
    / "outputs"
    / "lbl_kband"
    / "petitradtrans3_lbl_kband_reference.npz"
)
LINE_SPECIES = ("H2O__POKAZATEL", "CO__HITEMP")
DEFAULT_SPECIES_FILES = {
    "H2O__POKAZATEL": (
        "opacities/lines/line_by_line/H2O/1H2-16O/"
        "1H2-16O__POKAZATEL.R1e6_2.3-2.3mu.xsec.petitRADTRANS.h5"
    ),
    "CO__HITEMP": (
        "opacities/lines/line_by_line/CO/12C-16O/"
        "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
    ),
}
DEFAULT_MAX_MEMORY_GIB = 1.9
MAX_MEMORY_GIB = 2.0
# This fixed number ratio is the VMR representation of the former H2/He
# background mass ratio.  It preserves the old default physical composition
# after the private VMR-to-mass-fraction conversion at the pRT boundary.
BACKGROUND_VMR_RATIO = {"H2": 0.850289450888851, "He": 0.149710549111149}
MOLAR_MASSES = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O__POKAZATEL": 18.0,
    "CO__HITEMP": 28.0,
}
# These VMR defaults reproduce the former 0.001 H2O and 0.003 CO mass
# fractions, to floating-point precision, with the background ratio above.
DEFAULT_H2O_VMR = 1.2898479517314042e-4
DEFAULT_CO_VMR = 2.4875639069105653e-4
DEFAULT_WAVELENGTH_BOUNDS_MICRON = (2.29880, 2.30378)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument(
        "--max-memory-gib",
        type=_memory_gib_argument,
        default=DEFAULT_MAX_MEMORY_GIB,
    )
    parser.add_argument("--gravity-m-s2", type=float, default=15.0)
    parser.add_argument("--h2o-vmr", type=float, default=DEFAULT_H2O_VMR)
    parser.add_argument("--co-vmr", type=float, default=DEFAULT_CO_VMR)
    parser.add_argument("--temperature-top-k", type=float, default=900.0)
    parser.add_argument("--temperature-bottom-k", type=float, default=1800.0)
    parser.add_argument("--line-opacity-sampling", type=int, default=1)
    parser.add_argument(
        "--h2o-table",
        type=Path,
        default=Path(DEFAULT_SPECIES_FILES["H2O__POKAZATEL"]),
        help="Absolute path, or a path relative to --input-data.",
    )
    parser.add_argument(
        "--co-table",
        type=Path,
        default=Path(DEFAULT_SPECIES_FILES["CO__HITEMP"]),
        help="Absolute path, or a path relative to --input-data.",
    )
    parser.add_argument(
        "--wavelength-bounds-micron",
        type=float,
        nargs=2,
        default=DEFAULT_WAVELENGTH_BOUNDS_MICRON,
        metavar=("LOWER", "UPPER"),
    )
    return parser.parse_args()


def _validate_max_memory_gib(value: float) -> float:
    """Validate the strict process-memory ceiling in GiB."""

    memory_gib = float(value)
    if not np.isfinite(memory_gib) or not 0.0 < memory_gib < MAX_MEMORY_GIB:
        raise ValueError("max-memory-gib must be finite, positive, and below 2 GiB")
    return memory_gib


def _memory_gib_argument(value: str) -> float:
    """Parse a process-memory ceiling below the two-GiB hard limit."""

    try:
        return _validate_max_memory_gib(float(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _resolve_table_path(input_data: Path, value: Path) -> Path:
    path = value.expanduser()
    return path.resolve() if path.is_absolute() else (input_data / path).resolve()


@contextmanager
def _selected_input_data_view(
    table_paths: dict[str, Path],
) -> Iterator[Path]:
    """Expose one selected file per species to avoid pRT file ambiguity."""

    relative_directories = {
        "H2O__POKAZATEL": Path(
            "opacities/lines/line_by_line/H2O/1H2-16O"
        ),
        "CO__HITEMP": Path("opacities/lines/line_by_line/CO/12C-16O"),
    }
    with tempfile.TemporaryDirectory(prefix="robert-prt-lbl-input-") as temporary:
        root = Path(temporary)
        for species, source in table_paths.items():
            destination = root / relative_directories[species] / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
        yield root


def _complete_vmr_state(h2o_vmr: float, co_vmr: float) -> dict[str, float]:
    """Build the complete public VMR state used by the oracle."""

    h2o = float(h2o_vmr)
    co = float(co_vmr)
    active = h2o + co
    if not np.isfinite(active) or h2o < 0.0 or co < 0.0 or active >= 1.0:
        raise ValueError(
            "H2O and CO VMR values must be finite, non-negative, and sum below one"
        )
    background = 1.0 - active
    return {
        "H2": background * BACKGROUND_VMR_RATIO["H2"],
        "He": background * BACKGROUND_VMR_RATIO["He"],
        "H2O__POKAZATEL": h2o,
        "CO__HITEMP": co,
    }


def _vmr_to_mass_fractions(vmr: dict[str, float]) -> tuple[dict[str, float], float]:
    """Convert one complete VMR state at the private pRT boundary."""

    if set(vmr) != set(MOLAR_MASSES):
        raise ValueError("VMR state must contain exactly the pRT gas species")
    values = {name: float(value) for name, value in vmr.items()}
    total_vmr = sum(values.values())
    if (
        not np.isfinite(total_vmr)
        or abs(total_vmr - 1.0) > 2.0e-12
        or any(not np.isfinite(value) or value < 0.0 for value in values.values())
    ):
        raise ValueError("complete VMR state must be finite, non-negative, and sum to one")
    mean_molar_mass_amu = sum(
        values[name] * MOLAR_MASSES[name] for name in MOLAR_MASSES
    )
    if not np.isfinite(mean_molar_mass_amu) or mean_molar_mass_amu <= 0.0:
        raise ValueError("VMR state has an invalid mean molar mass")
    mass_fractions = {
        name: values[name] * MOLAR_MASSES[name] / mean_molar_mass_amu
        for name in MOLAR_MASSES
    }
    return mass_fractions, float(mean_molar_mass_amu)


def _table_preflight(
    path: Path,
    wavelength_bounds_micron: tuple[float, float],
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pRT line-by-line table: {path}")
    with h5py.File(path, "r") as handle:
        required = ("bin_edges", "xsecarr", "p", "t", "mol_mass")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path.name} is missing datasets: {', '.join(missing)}")
        wavenumber = np.asarray(handle["bin_edges"], dtype=float)
        lower, upper = wavelength_bounds_micron
        selected = (10000.0 / wavenumber >= lower) & (10000.0 / wavenumber <= upper)
        sample_count = int(np.count_nonzero(selected))
        if sample_count < 2:
            raise ValueError(f"{path.name} does not cover the requested wavelength range")
        cross_section = handle["xsecarr"]
        pressure_count, temperature_count, _ = cross_section.shape
        selected_bytes = (
            pressure_count
            * temperature_count
            * sample_count
            * np.dtype(cross_section.dtype).itemsize
        )
        return {
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "pressure_count": int(pressure_count),
            "temperature_count": int(temperature_count),
            "native_wavelength_count": int(wavenumber.size),
            "selected_wavelength_count": sample_count,
            "selected_cross_section_bytes": int(selected_bytes),
            "pressure_bar": [float(handle["p"][0]), float(handle["p"][-1])],
            "temperature_K": [float(np.min(handle["t"])), float(np.max(handle["t"]))],
            "wavelength_micron": [
                float(10000.0 / wavenumber[-1]),
                float(10000.0 / wavenumber[0]),
            ],
            "molar_mass_amu": float(handle["mol_mass"][0]),
        }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes. Linux reports KiB.
    if os.uname().sysname == "Darwin":
        return value
    return value * 1024


def main() -> dict[str, object]:
    args = _parse_args()
    if petitRADTRANS is None or Radtrans is None:
        raise ModuleNotFoundError(
            "petitRADTRANS is required; run this oracle in the petitradtrans-stable environment"
        )
    if args.layers < 2:
        raise ValueError("layers must be at least 2")
    max_memory_gib = _validate_max_memory_gib(args.max_memory_gib)
    if not np.isfinite(args.gravity_m_s2) or args.gravity_m_s2 <= 0.0:
        raise ValueError("gravity-m-s2 must be finite and positive")
    if (
        not np.isfinite(args.temperature_top_k)
        or not np.isfinite(args.temperature_bottom_k)
        or args.temperature_top_k <= 0.0
        or args.temperature_bottom_k <= 0.0
    ):
        raise ValueError("profile temperatures must be finite and positive")
    if args.line_opacity_sampling < 1:
        raise ValueError("line-opacity-sampling must be at least one")
    bounds = tuple(float(value) for value in args.wavelength_bounds_micron)
    if len(bounds) != 2 or bounds[0] <= 0.0 or bounds[0] >= bounds[1]:
        raise ValueError("wavelength bounds must be positive and increasing")

    input_data = args.input_data.expanduser().resolve()
    table_paths = {
        "H2O__POKAZATEL": _resolve_table_path(input_data, args.h2o_table),
        "CO__HITEMP": _resolve_table_path(input_data, args.co_table),
    }
    tables = {
        species: _table_preflight(path, bounds)
        for species, path in table_paths.items()
    }
    selected_bytes = sum(
        int(table["selected_cross_section_bytes"]) for table in tables.values()
    )
    selected_samples = max(
        int(table["selected_wavelength_count"]) for table in tables.values()
    )
    # This covers pRT opacity copies, interpolated arrays, RT work arrays, and
    # all full spectral axes. It is intentionally much larger than raw data.
    estimated_peak_bytes = int(
        24 * selected_bytes
        + 32 * args.layers * selected_samples * np.dtype(float).itemsize
        + 256 * 1024**2
    )
    max_memory_bytes = int(max_memory_gib * 1024**3)
    if estimated_peak_bytes > max_memory_bytes:
        raise MemoryError(
            "refusing pRT LBL run: conservative estimate "
            f"{estimated_peak_bytes / 1024**3:.2f} GiB exceeds "
            f"{args.max_memory_gib:.2f} GiB"
        )

    pressure_bar = np.geomspace(1.0e-5, 100.0, args.layers)
    log_pressure_fraction = (
        np.log10(pressure_bar) - np.log10(pressure_bar[0])
    ) / np.log10(pressure_bar[-1] / pressure_bar[0])
    temperature_K = args.temperature_top_k + (
        args.temperature_bottom_k - args.temperature_top_k
    ) * log_pressure_fraction
    volume_mixing_ratios = _complete_vmr_state(
        float(args.h2o_vmr),
        float(args.co_vmr),
    )
    mass_fractions, mean_molar_mass_amu = _vmr_to_mass_fractions(
        volume_mixing_ratios
    )
    mass_fraction_profiles = {
        name: np.full(args.layers, fraction, dtype=float)
        for name, fraction in mass_fractions.items()
    }

    start = perf_counter()
    with _selected_input_data_view(table_paths) as selected_input_data:
        atmosphere = Radtrans(
            pressures=pressure_bar,
            wavelength_boundaries=np.asarray(bounds),
            line_species=list(LINE_SPECIES),
            line_opacity_mode="lbl",
            line_by_line_opacity_sampling=args.line_opacity_sampling,
            scattering_in_emission=False,
            path_input_data=str(selected_input_data),
        )
        construction_seconds = perf_counter() - start

        start = perf_counter()
        wavelength_cm, flux_cgs_per_cm, _ = atmosphere.calculate_flux(
            temperatures=temperature_K,
            mass_fractions=mass_fraction_profiles,
            mean_molar_masses=np.full(args.layers, mean_molar_mass_amu),
            reference_gravity=args.gravity_m_s2 * 100.0,
            frequencies_to_wavelengths=True,
        )
        calculation_seconds = perf_counter() - start

    peak_rss_bytes = _peak_rss_bytes()
    if peak_rss_bytes > max_memory_bytes:
        raise MemoryError(
            f"pRT LBL run used {peak_rss_bytes / 1024**3:.2f} GiB RSS, "
            f"above the {args.max_memory_gib:.2f} GiB ceiling"
        )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "oracle": "petitRADTRANS stable line-by-line",
        "petitradtrans_version": petitRADTRANS.__version__,
        "line_opacity_mode": "lbl",
        "line_by_line_opacity_sampling": args.line_opacity_sampling,
        "native_resolving_power": 1_000_000,
        "line_species": list(LINE_SPECIES),
        "clear_atmosphere": True,
        "included_continua": [],
        "wavelength_bounds_micron": list(bounds),
        "n_layers": args.layers,
        "n_wavelength": int(np.asarray(wavelength_cm).size),
        "gravity_m_s2": args.gravity_m_s2,
        "volume_mixing_ratios": volume_mixing_ratios,
        "composition_provenance": {
            "input_representation": "volume mixing ratio (VMR)",
            "vmr_source": (
                "CLI --h2o-vmr and --co-vmr, with the fixed H2/He background "
                "VMR ratio"
            ),
            "pRT_boundary_conversion": {
                "formula": "MMW=sum_i(VMR_i*m_i); MMR_i=VMR_i*m_i/MMW",
                "mass_fractions": mass_fractions,
                "mean_molar_mass_amu": mean_molar_mass_amu,
                "role": "private deterministic conversion for Radtrans.calculate_flux",
            },
            "default_composition_note": (
                "The default VMR values reproduce the former H2O=0.001 and "
                "CO=0.003 mass-fraction state to floating-point precision. "
                "A regenerated NPZ checksum can still change because CLI and "
                "provenance metadata changed."
            ),
        },
        "mean_molar_mass_amu": mean_molar_mass_amu,
        "temperature_profile": {
            "coordinate": "linear in log10 pressure",
            "top_K": args.temperature_top_k,
            "bottom_K": args.temperature_bottom_k,
        },
        "tables": tables,
        "resource_policy": {
            "cpu_threads": 3,
            "max_memory_bytes": max_memory_bytes,
            "estimated_peak_bytes": estimated_peak_bytes,
            "measured_peak_rss_bytes": peak_rss_bytes,
        },
        "timings_seconds": {
            "construct_and_load": construction_seconds,
            "calculate_flux": calculation_seconds,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata)),
        pressure_bar=pressure_bar,
        temperature_K=temperature_K,
        wavelength_micron=np.asarray(wavelength_cm, dtype=float) * 1.0e4,
        flux_w_m2_m=np.asarray(flux_cgs_per_cm, dtype=float) * 0.1,
    )
    print(json.dumps(metadata, indent=2))
    print(f"oracle: {output}")
    return metadata


if __name__ == "__main__":
    main()
