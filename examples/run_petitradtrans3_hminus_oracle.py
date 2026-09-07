"""Generate a compact petitRADTRANS 3.3.3 H-minus emission oracle.

Run this file in the ``petitradtrans-stable`` environment.  The only large
input is the existing CO HITEMP R=1e6 HDF5 table.  CO has zero VMR in every
case.  It is used only as a physical wavelength-grid carrier.

The broad LRS run uses a temporary stride-1000 carrier derived by narrow
strided reads from that HDF5 file.  This is needed because pRT 3.3.3 reads the
full selected LBL slab before it applies its sampling stride.  The temporary
carrier is never committed.  The HRS run uses the original table and reads
only the narrow K-band slab.  The pRT frequency-bin edges are saved in each
NPZ archive.  ROBERT uses the converted wavelength edges for the identical
bound-free bin average.

The external pRT boundary needs mass fractions.  The scientific state remains
VMR-only: this script converts the documented VMR state to pRT mass fractions
and records the conversion, MMW, and all three H-minus species in metadata.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import tempfile
from time import perf_counter
from typing import Iterator, Mapping

# Keep the external numerical run to one process and one thread while other
# validation jobs run.  These assignments occur before NumPy and pRT import.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("OMPI_MCA_btl", "self")

import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CO_TABLE = (
    ROOT
    / "external_data"
    / "petitRADTRANS"
    / "input_data"
    / "opacities"
    / "lines"
    / "line_by_line"
    / "CO"
    / "12C-16O"
    / "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "outputs" / "hminus_continuum"

PRESSURE_CENTERS_BAR = np.geomspace(1.0e-5, 100.0, 80)
TEMPERATURE_REFERENCES_K = (2500.0, 3000.0, 3500.0)
GRAVITIES_M_S2 = (10.0, 30.0)
TEMPERATURE_GRADIENT_K_PER_DECADE = 100.0
LRS_BOUNDS_MICRON = (0.8, 5.2)
HRS_BOUNDS_MICRON = (2.2988, 2.30378)
LRS_CARRIER_STRIDE = 1000
MODES = ("off", "bound_free", "free_free", "total", "gray_cloud")
GRAY_OPACITY_CM2_G = 1.0e-2
MAX_PROCESS_BYTES = int(1.9 * 1024**3)
# Keep the configured ceiling strictly below 1 GiB.
MAX_OPACITY_BYTES = 1023 * 1024**2

# Atomic and molecular masses in amu.  The electron value is the pRT 3.3.3
# value.  These values are used only at the pRT external boundary.
MOLAR_MASSES_AMU: Mapping[str, float] = {
    "H2": 2.01588,
    "He": 4.002602,
    "CO__HITEMP": 28.0,
    "H-": 1.0005485799096196,
    "H": 1.00794,
    "e-": 5.485799096195737e-4,
}


@dataclass(frozen=True)
class VMRState:
    """One explicit, layer-constant VMR state for the benchmark."""

    mode: str
    hminus_vmr: float
    hydrogen_vmr: float
    electron_vmr: float
    co_vmr: float = 0.0
    background_h2_fraction: float = 0.85

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown H-minus mode: {self.mode}")
        values = (
            self.hminus_vmr,
            self.hydrogen_vmr,
            self.electron_vmr,
            self.co_vmr,
            self.background_h2_fraction,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("VMR state values must be finite")
        if any(value < 0.0 for value in values[:4]):
            raise ValueError("VMR state abundances must be non-negative")
        if not 0.0 <= self.background_h2_fraction <= 1.0:
            raise ValueError("background_h2_fraction must be in [0, 1]")
        if sum(values[:4]) >= 1.0:
            raise ValueError("trace VMRs must leave room for the H2/He background")

    @property
    def vmr(self) -> dict[str, float]:
        """Return the complete VMR state, including the H2/He fill."""

        trace_total = (
            self.hminus_vmr
            + self.hydrogen_vmr
            + self.electron_vmr
            + self.co_vmr
        )
        background = 1.0 - trace_total
        return {
            "H2": background * self.background_h2_fraction,
            "He": background * (1.0 - self.background_h2_fraction),
            "CO__HITEMP": self.co_vmr,
            "H-": self.hminus_vmr,
            "H": self.hydrogen_vmr,
            "e-": self.electron_vmr,
        }


BASE_HMINUS_VMR = 1.0e-6
BASE_HYDROGEN_VMR = 1.0e-2
BASE_ELECTRON_VMR = 1.0e-3


def vmr_state(mode: str) -> VMRState:
    """Return one of the fixed VMR states used by pRT and ROBERT.

    ``off`` retains the base VMR state but disables the pRT H-minus
    contributor.  The isolated component cases set only the VMRs needed for
    that component.  No equilibrium closure is used.
    """

    normalized = str(mode).strip().lower()
    if normalized in {"off", "total", "gray_cloud"}:
        return VMRState(
            normalized,
            BASE_HMINUS_VMR,
            BASE_HYDROGEN_VMR,
            BASE_ELECTRON_VMR,
        )
    if normalized == "bound_free":
        return VMRState(normalized, BASE_HMINUS_VMR, 0.0, 0.0)
    if normalized == "free_free":
        return VMRState(normalized, 0.0, BASE_HYDROGEN_VMR, BASE_ELECTRON_VMR)
    raise ValueError(f"unknown H-minus mode: {mode}")


def _vmr_to_mass_fractions(state: VMRState) -> tuple[dict[str, float], float]:
    """Convert one documented VMR state to pRT MMR at the external boundary."""

    vmr = state.vmr
    mean_molecular_weight = sum(
        vmr[species] * MOLAR_MASSES_AMU[species] for species in vmr
    )
    if not np.isfinite(mean_molecular_weight) or mean_molecular_weight <= 0.0:
        raise ValueError("VMR-to-MMR conversion produced an invalid MMW")
    mass_fractions = {
        species: value * MOLAR_MASSES_AMU[species] / mean_molecular_weight
        for species, value in vmr.items()
    }
    if not np.isclose(sum(mass_fractions.values()), 1.0, rtol=0.0, atol=2.0e-15):
        raise RuntimeError("VMR-to-MMR conversion did not conserve total mass")
    return mass_fractions, float(mean_molecular_weight)


def pressure_temperature_profile(reference_temperature_K: float) -> np.ndarray:
    """Return the shared linear-in-log-pressure thermal profile."""

    reference = float(reference_temperature_K)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference temperature must be finite and positive")
    profile = reference + TEMPERATURE_GRADIENT_K_PER_DECADE * np.log10(
        PRESSURE_CENTERS_BAR
    )
    if np.any(profile <= 0.0):
        raise ValueError("temperature profile must be positive")
    return profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--co-table", type=Path, default=DEFAULT_CO_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-process-gib", type=float, default=1.9)
    parser.add_argument(
        "--max-opacity-gib",
        type=float,
        default=MAX_OPACITY_BYTES / 1024**3,
        help="Strict opacity preparation guard. The value must be below 1 GiB.",
    )
    return parser.parse_args()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if os.uname().sysname == "Darwin" else value * 1024


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_h5py() -> object:
    import h5py

    return h5py


def _table_preflight(
    path: Path,
    bounds_micron: tuple[float, float],
) -> dict[str, object]:
    """Inspect a table and estimate one selected pRT opacity slab."""

    if not path.is_file():
        raise FileNotFoundError(f"missing CO pRT table: {path}")
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        required = ("bin_edges", "xsecarr", "p", "t", "mol_mass")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(missing)}")
        wavenumber = np.asarray(handle["bin_edges"], dtype=float)
        wavelength = 10000.0 / wavenumber
        lower, upper = bounds_micron
        selected = (wavelength >= lower) & (wavelength <= upper)
        count = int(np.count_nonzero(selected))
        if count < 2:
            raise ValueError(f"{path.name} does not cover {bounds_micron}")
        cross_section = handle["xsecarr"]
        pressure_count, temperature_count, _ = cross_section.shape
        selected_bytes = (
            pressure_count
            * temperature_count
            * count
            * np.dtype(cross_section.dtype).itemsize
        )
        return {
            "path": str(path),
            "file_bytes": int(path.stat().st_size),
            "pressure_count": int(pressure_count),
            "temperature_count": int(temperature_count),
            "native_wavelength_count": int(wavenumber.size),
            "selected_wavelength_count": count,
            "selected_cross_section_bytes": int(selected_bytes),
            "pressure_bar": [float(handle["p"][0]), float(handle["p"][-1])],
            "temperature_K": [
                float(np.min(handle["t"])),
                float(np.max(handle["t"])),
            ],
            "wavelength_micron": [
                float(np.min(wavelength)),
                float(np.max(wavelength)),
            ],
            "molar_mass_amu": float(handle["mol_mass"][0]),
        }


def _estimated_peak_bytes(preflight: Mapping[str, object]) -> int:
    """Return a conservative per-``Radtrans`` estimate."""

    selected_bytes = int(preflight["selected_cross_section_bytes"])
    selected_count = int(preflight["selected_wavelength_count"])
    return int(
        12 * selected_bytes
        + 64 * PRESSURE_CENTERS_BAR.size * selected_count * np.dtype(float).itemsize
        + 256 * 1024**2
    )


def _copy_strided_carrier(
    source: Path,
    destination: Path,
    *,
    stride: int,
) -> dict[str, object]:
    """Create a compact temporary pRT carrier with strided HDF5 reads."""

    if stride < 1:
        raise ValueError("carrier stride must be positive")
    h5py = _require_h5py()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as source_handle, h5py.File(destination, "w") as target:
        source_edges = source_handle["bin_edges"]
        start = 1 if source_edges.size > 1 and source_edges[0] == source_edges[1] else 0
        indices = np.arange(start, source_edges.size, stride, dtype=np.int64)
        if indices.size < 2:
            raise ValueError("strided carrier contains too few wavelength samples")
        target_edges = target.create_dataset(
            "bin_edges", data=np.asarray(source_edges[indices])
        )
        target_xsec = target.create_dataset(
            "xsecarr", data=np.asarray(source_handle["xsecarr"][:, :, indices])
        )
        for source_dataset, target_dataset in (
            (source_edges, target_edges),
            (source_handle["xsecarr"], target_xsec),
        ):
            for key, value in source_dataset.attrs.items():
                target_dataset.attrs[key] = value
        for name in ("p", "t", "mol_mass"):
            target_dataset = target.create_dataset(
                name, data=np.asarray(source_handle[name])
            )
            for key, value in source_handle[name].attrs.items():
                target_dataset.attrs[key] = value
        target.attrs["robert_source_table"] = str(source)
        target.attrs["robert_source_sha256"] = _file_sha256(source)
        target.attrs["robert_sampling_stride"] = int(stride)
    return {
        "path": str(destination),
        "source_path": str(source),
        "source_sampling_stride": int(stride),
        "source_read": "strided_hdf5_hyperslab",
        "temporary": True,
    }


@contextmanager
def _pRT_input_view(table_path: Path) -> Iterator[Path]:
    """Expose one selected CO table using pRT's normal directory layout."""

    with tempfile.TemporaryDirectory(prefix="robert-prt-hminus-input-") as temporary:
        root = Path(temporary)
        target = root / "opacities" / "lines" / "line_by_line" / "CO" / "12C-16O" / table_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(table_path)
        yield root


def _atomic_save_npz(path: Path, arrays: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _calculate_resolution(
    *,
    table_path: Path,
    bounds_micron: tuple[float, float],
    resolution_name: str,
    pRT_version: str,
    max_process_bytes: int,
    max_opacity_bytes: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Calculate one LRS or HRS archive in a bounded pRT process."""

    from petitRADTRANS import physical_constants as cst
    from petitRADTRANS.radtrans import Radtrans

    preflight = _table_preflight(table_path, bounds_micron)
    estimated_peak = _estimated_peak_bytes(preflight)
    selected_bytes = int(preflight["selected_cross_section_bytes"])
    if selected_bytes > max_opacity_bytes:
        raise MemoryError(
            f"refusing {resolution_name} pRT opacity slab: "
            f"{selected_bytes / 1024**3:.3f} GiB exceeds the "
            f"{max_opacity_bytes / 1024**3:.3f} GiB opacity ceiling"
        )
    if estimated_peak > max_process_bytes:
        raise MemoryError(
            f"refusing {resolution_name} pRT run: conservative estimate "
            f"{estimated_peak / 1024**3:.3f} GiB exceeds the "
            f"{max_process_bytes / 1024**3:.3f} GiB process ceiling"
        )

    start = perf_counter()
    with _pRT_input_view(table_path) as input_data:
        continuum_objects: dict[str, object] = {}
        for key, include_hminus in (("off", False), ("hminus", True)):
            continuum_objects[key] = Radtrans(
                pressures=PRESSURE_CENTERS_BAR,
                wavelength_boundaries=np.asarray(bounds_micron),
                line_species=["CO__HITEMP"],
                gas_continuum_contributors=["H-"] if include_hminus else [],
                line_opacity_mode="lbl",
                line_by_line_opacity_sampling=1,
                scattering_in_emission=False,
                path_input_data=str(input_data),
            )

        first_object = continuum_objects["off"]
        frequency_edges = np.asarray(first_object._frequency_bins_edges, dtype=float)
        frequency_centers = np.asarray(first_object._frequencies, dtype=float)
        wavelength_edges = 1.0e4 * float(cst.c) / frequency_edges
        if not (
            np.all(np.diff(wavelength_edges) > 0.0)
            or np.all(np.diff(wavelength_edges) < 0.0)
        ):
            raise RuntimeError("pRT wavelength bin edges are not monotonic")

        flux_by_case: list[np.ndarray] = []
        mass_fraction_metadata: dict[str, object] = {}
        wavelengths: np.ndarray | None = None
        for temperature_reference in TEMPERATURE_REFERENCES_K:
            temperatures = pressure_temperature_profile(temperature_reference)
            for gravity in GRAVITIES_M_S2:
                flux_by_mode: list[np.ndarray] = []
                for mode in MODES:
                    state = vmr_state(mode)
                    mass_fractions, mean_molecular_weight = _vmr_to_mass_fractions(state)
                    mass_profiles = {
                        species: np.full(PRESSURE_CENTERS_BAR.size, value, dtype=float)
                        for species, value in mass_fractions.items()
                    }
                    continuum_key = "hminus" if mode in {"bound_free", "free_free", "total"} else "off"
                    atmosphere = continuum_objects[continuum_key]
                    gray_opacity = GRAY_OPACITY_CM2_G if mode == "gray_cloud" else None
                    wavelength_cm, flux_cgs_per_cm, _ = atmosphere.calculate_flux(
                        temperatures=temperatures,
                        mass_fractions=mass_profiles,
                        mean_molar_masses=np.full(
                            PRESSURE_CENTERS_BAR.size,
                            mean_molecular_weight,
                            dtype=float,
                        ),
                        reference_gravity=float(gravity) * 100.0,
                        gray_opacity=gray_opacity,
                        frequencies_to_wavelengths=True,
                    )
                    wavelength = np.asarray(wavelength_cm, dtype=float) * 1.0e4
                    flux = np.asarray(flux_cgs_per_cm, dtype=float) * 0.1
                    if wavelengths is None:
                        wavelengths = wavelength
                    elif not np.allclose(wavelengths, wavelength, rtol=0.0, atol=0.0):
                        raise RuntimeError("pRT changed the wavelength grid between cases")
                    flux_by_mode.append(flux)
                    mass_fraction_metadata.setdefault(mode, {
                        "vmr": state.vmr,
                        "mass_fractions": mass_fractions,
                        "mean_molecular_weight_amu": mean_molecular_weight,
                    })
                flux_by_case.append(np.stack(flux_by_mode, axis=0))

    if wavelengths is None:
        raise RuntimeError("pRT returned no spectrum")
    flux_array = np.stack(flux_by_case, axis=0).reshape(
        len(TEMPERATURE_REFERENCES_K),
        len(GRAVITIES_M_S2),
        len(MODES),
        wavelengths.size,
    )
    measured_rss = _peak_rss_bytes()
    if measured_rss > max_process_bytes:
        raise MemoryError(
            f"{resolution_name} pRT run used {measured_rss / 1024**3:.3f} GiB RSS, "
            f"above the {max_process_bytes / 1024**3:.3f} GiB ceiling"
        )
    metadata: dict[str, object] = {
        "resolution_name": resolution_name,
        "petitradtrans_version": pRT_version,
        "line_opacity_mode": "lbl",
        "line_by_line_opacity_sampling": 1,
        "line_species": ["CO__HITEMP"],
        "co_zero_abundance_grid_carrier": True,
        "wavelength_bounds_micron": list(bounds_micron),
        "native_resolving_power": 1_000_000,
        "effective_resolving_power": 1_000_000
        if resolution_name == "hrs_native"
        else 1_000,
        "n_layers": int(PRESSURE_CENTERS_BAR.size),
        "n_wavelength": int(wavelengths.size),
        "pressure_centers_bar": PRESSURE_CENTERS_BAR.tolist(),
        "temperature_references_K": list(TEMPERATURE_REFERENCES_K),
        "temperature_gradient_K_per_decade": TEMPERATURE_GRADIENT_K_PER_DECADE,
        "gravities_m_s2": list(GRAVITIES_M_S2),
        "modes": list(MODES),
        "gray_opacity_cm2_g": GRAY_OPACITY_CM2_G,
        "vmr_states": {
            mode: vmr_state(mode).vmr
            for mode in MODES
        },
        "pRT_mass_fraction_states": mass_fraction_metadata,
        "hminus_formula": {
            "bound_free": (
                "Gray polynomial sigma_bf(lambda) in cm^2 per H- particle; "
                "pRT performs a wavelength-bin average below 16400 Angstrom"
            ),
            "free_free": (
                "Gray polynomial sigma_ff(lambda,T,p_e) in cm^2 per H particle; "
                "p_e is in dyn cm^-2 and the fit is zero below 2500 K"
            ),
            "source": "Gray (2008), pp. 155-156, as implemented by pRT 3.3.3",
            "hminus_species": "H-",
            "hydrogen_species": "H",
            "electron_species": "e-",
        },
        "vmr_to_mass_fraction": {
            "formula": "MMW=sum_i(VMR_i*m_i); MMR_i=VMR_i*m_i/MMW",
            "role": "private external pRT boundary conversion",
            "robert_inputs": "VMR only",
            "molar_masses_amu": dict(MOLAR_MASSES_AMU),
        },
        "frequency_bin_edges_source": "Radtrans._frequency_bins_edges",
        "frequency_centers_source": "Radtrans._frequencies",
        "frequency_bin_edges_unit": "Hz",
        "wavelength_bin_edges_unit": "micron",
        "resource_policy": {
            "process_threads": 1,
            "process_memory_ceiling_bytes": max_process_bytes,
            "opacity_memory_ceiling_bytes": max_opacity_bytes,
            "selected_opacity_bytes": selected_bytes,
            "estimated_peak_bytes": estimated_peak,
            "measured_peak_rss_bytes": measured_rss,
        },
        "table_preflight": preflight,
        "timing_seconds": {"pRT_calculate_flux": perf_counter() - start},
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "pressure_bar": np.asarray(PRESSURE_CENTERS_BAR),
        "temperature_reference_K": np.asarray(TEMPERATURE_REFERENCES_K),
        "temperature_profile_K": np.stack(
            [pressure_temperature_profile(value) for value in TEMPERATURE_REFERENCES_K]
        ),
        "gravity_m_s2": np.asarray(GRAVITIES_M_S2),
        "wavelength_micron": wavelengths,
        "frequency_centers_hz": frequency_centers,
        "frequency_bin_edges_hz": frequency_edges,
        "wavelength_bin_edges_micron": wavelength_edges,
        "flux_w_m2_m": flux_array,
    }
    return metadata, arrays


def main() -> dict[str, object]:
    """Generate LRS and HRS pRT archives without committing opacity data."""

    args = _parse_args()
    max_process_bytes = int(float(args.max_process_gib) * 1024**3)
    max_opacity_bytes = int(float(args.max_opacity_gib) * 1024**3)
    if max_process_bytes <= 0 or max_process_bytes > MAX_PROCESS_BYTES:
        raise ValueError("process ceiling must be positive and no greater than 1.9 GiB")
    if max_opacity_bytes <= 0 or max_opacity_bytes >= MAX_OPACITY_BYTES:
        raise ValueError("opacity ceiling must be positive and below 1 GiB")
    source = args.co_table.expanduser().resolve()
    source_sha256 = _file_sha256(source)

    # pRT imports are local so the VMR helpers remain testable in the ROBERT
    # environment, where the optional pRT package is not required.
    import petitRADTRANS  # noqa: PLC0415

    output_dir = args.output_dir.expanduser().resolve()
    overall_start = perf_counter()
    with tempfile.TemporaryDirectory(prefix="robert-prt-hminus-carrier-") as temporary:
        temporary_carrier = Path(temporary) / source.name
        carrier_metadata = _copy_strided_carrier(
            source,
            temporary_carrier,
            stride=LRS_CARRIER_STRIDE,
        )
        lrs_metadata, lrs_arrays = _calculate_resolution(
            table_path=temporary_carrier,
            bounds_micron=LRS_BOUNDS_MICRON,
            resolution_name="lrs",
            pRT_version=str(petitRADTRANS.__version__),
            max_process_bytes=max_process_bytes,
            max_opacity_bytes=max_opacity_bytes,
        )
        hrs_metadata, hrs_arrays = _calculate_resolution(
            table_path=source,
            bounds_micron=HRS_BOUNDS_MICRON,
            resolution_name="hrs_native",
            pRT_version=str(petitRADTRANS.__version__),
            max_process_bytes=max_process_bytes,
            max_opacity_bytes=max_opacity_bytes,
        )

    lrs_metadata["carrier"] = carrier_metadata
    lrs_metadata["source_table_sha256"] = source_sha256
    lrs_metadata["source_table_path"] = str(source)
    hrs_metadata["source_table_sha256"] = source_sha256
    hrs_metadata["source_table_path"] = str(source)
    _atomic_save_npz(output_dir / "petitradtrans3_hminus_lrs.npz", lrs_arrays)
    _atomic_save_npz(output_dir / "petitradtrans3_hminus_hrs_native.npz", hrs_arrays)

    report = {
        "schema_version": 1,
        "oracle": "petitRADTRANS 3.3.3 H-minus clear-emission oracle",
        "petitradtrans_version": str(petitRADTRANS.__version__),
        "source_table": {
            "path": str(source),
            "sha256": source_sha256,
            "role": "zero-abundance CO spectral-grid carrier",
            "file_bytes": int(source.stat().st_size),
        },
        "lrs": lrs_metadata,
        "hrs_native": hrs_metadata,
        "resource_policy": {
            "process_threads": 1,
            "process_memory_ceiling_bytes": max_process_bytes,
            "opacity_memory_ceiling_bytes": max_opacity_bytes,
            "measured_peak_rss_bytes": _peak_rss_bytes(),
        },
        "total_elapsed_seconds": perf_counter() - overall_start,
        "outputs": {
            "lrs": str(output_dir / "petitradtrans3_hminus_lrs.npz"),
            "hrs_native": str(output_dir / "petitradtrans3_hminus_hrs_native.npz"),
        },
    }
    _atomic_write_json(output_dir / "petitradtrans3_hminus_oracle.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
