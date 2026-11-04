"""Optional exo_k bridge for accuracy-preserving correlated-k binning."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from robert_exoplanets.core import (
    RobertConfigError,
    RobertCoverageError,
    RobertValidationError,
    SpectralGrid,
)

from .correlated_k import CorrelatedKTable
from .inspectors import file_sha256
from .kta import KtaHeader, read_kta_header
from .metadata import (
    spectral_grid_bin_bounds_in_unit,
    spectral_grid_values_in_unit,
)


def bin_correlated_k_table_with_exok(
    table: CorrelatedKTable,
    spectral_grid: SpectralGrid,
    *,
    num: int = 300,
    use_rebin: bool = False,
    remove_zeros: bool = True,
    zero_deltalog_min_value: float = 10.0,
) -> CorrelatedKTable:
    """Bin an exo_k-readable table using ``exo_k.Ktable.bin_down_cp``.

    ``exo_k`` re-sorts and recompresses the opacity distribution within every
    requested wavenumber bin. Individual g ordinates are not interpolated in
    wavelength.
    """

    if spectral_grid.bin_edges is None:
        raise RobertValidationError(
            "exo_k correlated-k binning requires spectral bin edges"
        )
    if int(num) < 2:
        raise RobertValidationError("exo_k binning num must be at least two")
    if not np.isfinite(zero_deltalog_min_value) or zero_deltalog_min_value <= 0.0:
        raise RobertValidationError(
            "exo_k zero_deltalog_min_value must be finite and positive"
        )
    if spectral_grid.bin_edges.ndim == 2:
        contiguous_edges = _contiguous_edges_for_grid(spectral_grid)
        if contiguous_edges is None:
            return _bin_exact_spectral_windows_with_exok(
                table,
                spectral_grid,
                num=int(num),
                use_rebin=bool(use_rebin),
                remove_zeros=bool(remove_zeros),
                zero_deltalog_min_value=float(zero_deltalog_min_value),
            )
        spectral_grid = SpectralGrid(
            values=spectral_grid.values,
            unit=spectral_grid.unit,
            name=spectral_grid.name,
            role=spectral_grid.role,
            bin_edges=contiguous_edges,
            metadata=spectral_grid.metadata,
        )
    source_path = str(table.metadata.get("source_path", ""))
    if not source_path or not Path(source_path).is_file():
        raise RobertConfigError(
            "exo_k binning requires an exo_k-readable source opacity file"
        )
    exok = _import_exok()
    try:
        native = exok.Ktable(
            filename=source_path,
            mol=table.species,
            p_unit="bar",
            kdata_unit="cm^2/molecule",
            remove_zeros=False,
        )
    except Exception as exc:
        raise RobertConfigError(
            f"exo_k could not load correlated-k table: {source_path}"
        ) from exc

    _validate_exok_source(native, table)
    native_kcoeff = table.kcoeff
    if np.asarray(native.wns)[0] > np.asarray(native.wns)[-1]:
        native_kcoeff = native_kcoeff[:, :, ::-1, :]
    native.kdata = np.array(native_kcoeff, dtype=float, copy=True)
    zero_mask = native.kdata == 0.0
    zero_count = int(np.sum(zero_mask))
    zero_floor: float | None = None
    if remove_zeros and zero_count:
        native.remove_zeros(deltalog_min_value=float(zero_deltalog_min_value))
        zero_floor = float(np.min(native.kdata[zero_mask]))
    native.ggrid = np.array(table.g_samples, dtype=float, copy=True)
    native.weights = np.array(table.g_weights, dtype=float, copy=True)
    native.gedges = np.concatenate(([0.0], np.cumsum(native.weights)))
    native.Ng = native.weights.size

    edge_grid = SpectralGrid(
        values=spectral_grid.bin_edges, unit=spectral_grid.unit, role="internal"
    )
    target_wavenumber_edges = np.sort(spectral_grid_values_in_unit(edge_grid, "cm^-1"))
    native_edges = np.asarray(native.wnedges, dtype=float)
    tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(native_edges))))
    if (
        target_wavenumber_edges[0] < np.min(native_edges) - tolerance
        or target_wavenumber_edges[-1] > np.max(native_edges) + tolerance
    ):
        raise RobertCoverageError(
            "requested spectral bins extend outside the native exo_k table"
        )

    try:
        binned = native.bin_down_cp(
            wnedges=target_wavenumber_edges,
            weights=table.g_weights,
            ggrid=table.g_samples,
            num=int(num),
            use_rebin=bool(use_rebin),
            remove_zeros=bool(remove_zeros),
        )
    except Exception as exc:
        raise RobertValidationError(
            "exo_k failed while binning the correlated-k distribution"
        ) from exc

    target_wavenumber = spectral_grid_values_in_unit(spectral_grid, "cm^-1")
    kcoeff = np.asarray(binned.kdata, dtype=float)
    exok_wavenumber = np.asarray(binned.wns, dtype=float)
    if target_wavenumber[0] > target_wavenumber[-1]:
        kcoeff = kcoeff[:, :, ::-1, :]
        exok_wavenumber = exok_wavenumber[::-1]
    if kcoeff.shape[2] != spectral_grid.size:
        raise RobertValidationError(
            "exo_k returned an unexpected number of spectral bins"
        )
    center_offset = float(np.max(np.abs(exok_wavenumber - target_wavenumber)))
    return CorrelatedKTable(
        species=table.species,
        pressure_bar=table.pressure_bar,
        temperature_K=table.temperature_K,
        wavenumber_cm_inverse=target_wavenumber,
        wavelength_micron=10000.0 / target_wavenumber,
        g_samples=np.asarray(binned.ggrid, dtype=float),
        g_weights=np.asarray(binned.weights, dtype=float),
        kcoeff=kcoeff,
        unit=str(binned.kdata_unit),
        metadata={
            **dict(table.metadata),
            "spectral_preparation": "exo_k_bin_down",
            "exo_k_version": str(getattr(exok, "__version__", "unknown")),
            "exo_k_num": str(int(num)),
            "exo_k_use_rebin": str(bool(use_rebin)).lower(),
            "exo_k_remove_zeros": str(bool(remove_zeros)).lower(),
            "exo_k_zero_deltalog_min_value": f"{float(zero_deltalog_min_value):.12g}",
            "exo_k_zeros_replaced": str(zero_count if remove_zeros else 0),
            "exo_k_zero_floor": "" if zero_floor is None else f"{zero_floor:.17g}",
            "exo_k_max_center_offset_cm-1": f"{center_offset:.12g}",
            "native_spectral_points": str(table.wavenumber_cm_inverse.size),
            "binned_spectral_points": str(spectral_grid.size),
        },
    )


def _contiguous_edges_for_grid(
    spectral_grid: SpectralGrid,
) -> np.ndarray | None:
    """Return 1D edges when per-bin bounds exactly form one continuous grid."""

    bounds = spectral_grid.bin_bounds
    assert bounds is not None
    if spectral_grid.size == 1:
        return np.asarray(bounds[0], dtype=float)
    increasing = bool(spectral_grid.values[0] < spectral_grid.values[-1])
    if increasing:
        if not np.array_equal(bounds[:-1, 1], bounds[1:, 0]):
            return None
        return np.concatenate((bounds[:1, 0], bounds[:, 1]))
    if not np.array_equal(bounds[:-1, 0], bounds[1:, 1]):
        return None
    return np.concatenate((bounds[:1, 1], bounds[:, 0]))


def _bin_exact_spectral_windows_with_exok(
    table: CorrelatedKTable,
    spectral_grid: SpectralGrid,
    *,
    num: int,
    use_rebin: bool,
    remove_zeros: bool,
    zero_deltalog_min_value: float,
) -> CorrelatedKTable:
    """Bin non-contiguous or overlapping windows independently with exo_k.

    The native opacity data are already loaded in ``table``. Each backend
    call receives only the contiguous native spectral slice that brackets its
    requested window. The backend therefore never copies the full source
    array once per reported bin.
    """

    source_path = str(table.metadata.get("source_path", ""))
    if not source_path or not Path(source_path).is_file():
        raise RobertConfigError(
            "exo_k binning requires an exo_k-readable source opacity file"
        )
    exok = _import_exok()
    source_wavenumber, source_edges, source_kcoeff = _ascending_native_source(
        exok,
        table,
        source_path,
    )
    target_bounds = spectral_grid_bin_bounds_in_unit(
        spectral_grid,
        "cm^-1",
    )
    target_centers = spectral_grid_values_in_unit(spectral_grid, "cm^-1")
    assert target_bounds is not None
    tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(source_edges))))
    if (
        np.min(target_bounds) < np.min(source_edges) - tolerance
        or np.max(target_bounds) > np.max(source_edges) + tolerance
    ):
        raise RobertCoverageError(
            "requested spectral bins extend outside the native exo_k table"
        )

    zero_count = 0
    zero_floor: float | None = None
    if remove_zeros:
        zero_count, minimum_positive = _zero_statistics(source_kcoeff)
        if zero_count and minimum_positive is None:
            raise RobertValidationError(
                "exo_k cannot replace zeros when the native table has no positive coefficients"
            )
        if zero_count and minimum_positive is not None:
            zero_floor = minimum_positive / (10.0**zero_deltalog_min_value)

    native = _empty_exok_table(exok, table, source_wavenumber, source_edges)
    output = np.empty(
        (
            table.pressure_bar.size,
            table.temperature_K.size,
            spectral_grid.size,
            table.g_samples.size,
        ),
        dtype=float,
    )
    center_offsets = np.empty(spectral_grid.size, dtype=float)
    for index, (lower, upper) in enumerate(target_bounds):
        first = max(0, int(np.searchsorted(source_edges, lower, side="right")) - 1)
        stop = min(
            source_wavenumber.size,
            int(np.searchsorted(source_edges, upper, side="left")),
        )
        if stop <= first:
            raise RobertCoverageError(
                f"requested spectral bin {index} contains no native opacity cells"
            )
        window = source_kcoeff[:, :, first:stop, :]
        if remove_zeros and zero_floor is not None and np.any(window == 0.0):
            window = np.array(window, dtype=float, copy=True)
            window[window == 0.0] = zero_floor
        native.kdata = window
        native.wns = source_wavenumber[first:stop]
        native.wnedges = source_edges[first : stop + 1]
        native.Nw = stop - first
        try:
            binned = native.bin_down_cp(
                wnedges=np.asarray([lower, upper], dtype=float),
                weights=table.g_weights,
                ggrid=table.g_samples,
                num=num,
                use_rebin=use_rebin,
                remove_zeros=remove_zeros,
            )
        except Exception as exc:
            raise RobertValidationError(
                f"exo_k failed while binning spectral window {index}"
            ) from exc
        if binned.kdata.shape[2] != 1:
            raise RobertValidationError(
                "exo_k returned an unexpected number of bins for a spectral window"
            )
        output[:, :, index, :] = np.asarray(binned.kdata[:, :, 0, :], dtype=float)
        center_offsets[index] = abs(
            0.5 * (lower + upper) - target_centers[index]
        )

    return CorrelatedKTable(
        species=table.species,
        pressure_bar=table.pressure_bar,
        temperature_K=table.temperature_K,
        wavenumber_cm_inverse=target_centers,
        wavelength_micron=spectral_grid_values_in_unit(spectral_grid, "micron"),
        g_samples=table.g_samples,
        g_weights=table.g_weights,
        kcoeff=output,
        unit=table.unit,
        metadata={
            **dict(table.metadata),
            "spectral_preparation": "exo_k_bin_down_exact_windows",
            "exo_k_version": str(getattr(exok, "__version__", "unknown")),
            "exo_k_num": str(num),
            "exo_k_use_rebin": str(use_rebin).lower(),
            "exo_k_remove_zeros": str(remove_zeros).lower(),
            "exo_k_zero_deltalog_min_value": f"{zero_deltalog_min_value:.12g}",
            "exo_k_zeros_replaced": str(zero_count),
            "exo_k_zero_floor": "" if zero_floor is None else f"{zero_floor:.17g}",
            "exo_k_max_center_offset_cm-1": f"{float(np.max(center_offsets)):.12g}",
            "native_spectral_points": str(table.wavenumber_cm_inverse.size),
            "binned_spectral_points": str(spectral_grid.size),
            "binned_spectral_windows": str(spectral_grid.size),
        },
    )


def _ascending_native_source(
    exok,
    table: CorrelatedKTable,
    source_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source centers, cell edges, and coefficients in increasing cm-1."""

    if Path(source_path).suffix.lower() == ".kta":
        header = _validate_kta_table_matches_source(table, source_path)
        # exo_k reads wavelength centers as float32 and forms wavenumber
        # centers and midpoint edges before its binary64 binning work for
        # irregular wavelength grids. Regular KTA grids use float64 values
        # made from wl_min + arange(Nw) * dwl in exo_k.
        if header.wavelength_step_micron >= 0.0:
            stored_wavelength = header.wavelength_min_micron + np.arange(
                header.n_wavelength,
                dtype=float,
            ) * header.wavelength_step_micron
            native_wavenumber = 10000.0 / stored_wavelength[::-1]
        else:
            native_wavelength = np.asarray(
                table.wavelength_micron,
                dtype=np.float32,
            )
            native_wavenumber = np.asarray(
                np.float32(10000.0) / native_wavelength,
                dtype=np.float32,
            )
        source_kcoeff = table.kcoeff
        if native_wavenumber[0] > native_wavenumber[-1]:
            native_wavenumber = native_wavenumber[::-1]
            source_kcoeff = source_kcoeff[:, :, ::-1, :]
        source_wavenumber = np.asarray(native_wavenumber, dtype=float)
        midpoint = 0.5 * (native_wavenumber[:-1] + native_wavenumber[1:])
        source_edges = np.asarray(
            np.concatenate(
                (native_wavenumber[:1], midpoint, native_wavenumber[-1:])
            ),
            dtype=float,
        )
    else:
        table_wavenumber = np.asarray(table.wavenumber_cm_inverse, dtype=float)
        if table_wavenumber[0] > table_wavenumber[-1]:
            source_wavenumber = table_wavenumber[::-1]
            source_kcoeff = table.kcoeff[:, :, ::-1, :]
        else:
            source_wavenumber = table_wavenumber
            source_kcoeff = table.kcoeff
        try:
            native = exok.Ktable(
                filename=source_path,
                mol=table.species,
                p_unit="bar",
                kdata_unit=table.unit,
                remove_zeros=False,
            )
        except Exception as exc:
            raise RobertConfigError(
                f"exo_k could not load correlated-k table: {source_path}"
            ) from exc
        _validate_exok_source(native, table)
        native_wavenumber = np.asarray(native.wns, dtype=float)
        source_edges = np.asarray(native.wnedges, dtype=float)
        if native_wavenumber[0] > native_wavenumber[-1]:
            native_wavenumber = native_wavenumber[::-1]
            source_edges = source_edges[::-1]
        if native_wavenumber.shape != source_wavenumber.shape or not np.allclose(
            native_wavenumber,
            source_wavenumber,
            rtol=2.0e-6,
            atol=1.0e-6,
        ):
            raise RobertValidationError(
                "exo_k and ROBERT disagree on the native spectral grid"
            )
        del native

    if np.any(np.diff(source_edges) <= 0.0):
        raise RobertValidationError("native exo_k spectral edges must increase")
    if source_edges.size != source_wavenumber.size + 1:
        raise RobertValidationError(
            "native exo_k spectral edges must bracket every source coefficient"
        )
    return source_wavenumber, source_edges, source_kcoeff


def _validate_kta_table_matches_source(
    table: CorrelatedKTable,
    source_path: str,
) -> KtaHeader:
    """Check cached table grids against a lightweight native KTA header."""

    header = read_kta_header(source_path, checksum=False)
    if table.kcoeff.shape != header.native_shape:
        raise RobertValidationError(
            "correlated-k table does not match its native KTA spectral dimensions"
        )
    checks = (
        (table.pressure_bar, header.pressure_bar, "pressure"),
        (table.temperature_K, header.temperature_K, "temperature"),
        (table.g_samples, header.g_samples, "g ordinate"),
        (table.g_weights, header.g_weights, "g weight"),
    )
    for candidate, reference, label in checks:
        reference_values = np.asarray(reference, dtype=float)
        if label == "g weight":
            weight_sum = float(np.sum(reference_values))
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                raise RobertValidationError(
                    "native KTA g weights must have a finite positive sum"
                )
            reference_values = reference_values / weight_sum
        if candidate.shape != reference.shape or not np.allclose(
            candidate,
            reference_values,
            rtol=1.0e-6,
            atol=1.0e-12,
        ):
            raise RobertValidationError(
                f"correlated-k table does not match its native KTA {label} grid"
            )
    candidate_wavenumber = np.asarray(table.wavenumber_cm_inverse, dtype=float)
    reference_wavenumber = np.asarray(header.wavenumber_cm_inverse, dtype=float)
    if candidate_wavenumber[0] > candidate_wavenumber[-1]:
        candidate_wavenumber = candidate_wavenumber[::-1]
    if reference_wavenumber[0] > reference_wavenumber[-1]:
        reference_wavenumber = reference_wavenumber[::-1]
    if candidate_wavenumber.shape != reference_wavenumber.shape or not np.allclose(
        candidate_wavenumber,
        reference_wavenumber,
        rtol=2.0e-6,
        atol=1.0e-6,
    ):
        raise RobertValidationError(
            "correlated-k table does not match its native KTA spectral grid"
        )
    return header


def _empty_exok_table(exok, table: CorrelatedKTable, wavenumber, edges):
    """Create a reusable exo_k table shell without loading or copying full data."""

    native = exok.Ktable()
    native.filename = str(table.metadata.get("source_path", ""))
    native.mol = table.species
    native.pgrid = table.pressure_bar
    native.logpgrid = np.log10(table.pressure_bar)
    native.tgrid = table.temperature_K
    native.wns = wavenumber
    native.wnedges = edges
    native.Np = table.pressure_bar.size
    native.Nt = table.temperature_K.size
    native.Nw = wavenumber.size
    native.Ng = table.g_samples.size
    native.Nx = None
    native.ggrid = table.g_samples
    native.weights = table.g_weights
    native.gedges = np.concatenate(([0.0], np.cumsum(table.g_weights)))
    native.kdata_unit = table.unit
    native.p_unit = "bar"
    native.kdata = table.kcoeff
    return native


def _zero_statistics(kcoeff: np.ndarray) -> tuple[int, float | None]:
    """Count zero coefficients and find the global positive minimum in chunks."""

    n_wavelength = kcoeff.shape[2]
    values_per_spectrum = kcoeff.shape[0] * kcoeff.shape[1] * kcoeff.shape[3]
    block = max(1, 1_000_000 // max(1, values_per_spectrum))
    zero_count = 0
    minimum_positive: float | None = None
    for start in range(0, n_wavelength, block):
        sample = kcoeff[:, :, start : start + block, :]
        zero_count += int(np.count_nonzero(sample == 0.0))
        positive = sample[sample > 0.0]
        if positive.size:
            current = float(np.min(positive))
            minimum_positive = (
                current
                if minimum_positive is None
                else min(minimum_positive, current)
            )
    return zero_count, minimum_positive


def load_correlated_k_table_with_exok(
    path: str | Path,
    *,
    species: str,
    nonfinite_policy: str = "raise",
    nonfinite_fill_value: float = 1.0e-300,
    remove_zeros: bool = True,
    zero_deltalog_min_value: float = 10.0,
) -> CorrelatedKTable:
    """Load any precomputed correlated-k format supported by ``exo_k``.

    The file may be an ExoMol/exo_k KTA or HDF5 product, or another Ktable
    format recognized by the installed exo_k version. Raw ExoMol line lists
    are not precomputed opacity tables and are outside this loader's scope.
    """

    table_path = Path(path).expanduser()
    if not table_path.is_file():
        raise RobertConfigError(
            f"correlated-k opacity file does not exist: {table_path}"
        )
    species_name = str(species).strip()
    if not species_name:
        raise RobertValidationError("correlated-k species must not be empty")
    normalized_policy = nonfinite_policy.strip().lower()
    if normalized_policy in {"strict", "error"}:
        normalized_policy = "raise"
    if normalized_policy not in {"raise", "floor"}:
        raise RobertValidationError("nonfinite_policy must be 'raise' or 'floor'")
    if not np.isfinite(nonfinite_fill_value) or nonfinite_fill_value <= 0.0:
        raise RobertValidationError("nonfinite_fill_value must be finite and positive")
    if not np.isfinite(zero_deltalog_min_value) or zero_deltalog_min_value <= 0.0:
        raise RobertValidationError(
            "zero_deltalog_min_value must be finite and positive"
        )

    exok = _import_exok()
    try:
        native = exok.Ktable(
            filename=str(table_path),
            mol=species_name,
            p_unit="bar",
            kdata_unit="cm^2/molecule",
            remove_zeros=False,
        )
    except Exception as exc:
        raise RobertConfigError(
            f"exo_k could not load correlated-k table: {table_path}"
        ) from exc
    kdata = np.array(native.kdata, dtype=float, copy=True)
    nonfinite = ~np.isfinite(kdata)
    n_nonfinite = int(np.sum(nonfinite))
    if n_nonfinite and normalized_policy == "raise":
        raise RobertValidationError(
            f"kcoeff contains {n_nonfinite} non-finite values; use nonfinite_policy='floor'"
        )
    if n_nonfinite:
        kdata[nonfinite] = float(nonfinite_fill_value)
    if np.any(kdata < 0.0):
        raise RobertValidationError("correlated-k coefficients must not be negative")
    native.kdata = kdata
    zero_mask = native.kdata == 0.0
    zero_count = int(np.sum(zero_mask))
    zero_floor: float | None = None
    if remove_zeros and zero_count:
        native.remove_zeros(deltalog_min_value=float(zero_deltalog_min_value))
        zero_floor = float(np.min(native.kdata[zero_mask]))

    wavenumber = np.asarray(native.wns, dtype=float)
    return CorrelatedKTable(
        species=species_name,
        pressure_bar=np.asarray(native.pgrid, dtype=float),
        temperature_K=np.asarray(native.tgrid, dtype=float),
        wavenumber_cm_inverse=wavenumber,
        wavelength_micron=10000.0 / wavenumber,
        g_samples=np.asarray(native.ggrid, dtype=float),
        g_weights=np.asarray(native.weights, dtype=float),
        kcoeff=np.asarray(native.kdata, dtype=float),
        unit=str(native.kdata_unit),
        metadata={
            "source_format": f"exo_k:{table_path.suffix.lower().lstrip('.') or 'unknown'}",
            "source_path": str(table_path),
            "checksum_sha256": file_sha256(table_path),
            "exo_k_version": str(getattr(exok, "__version__", "unknown")),
            "kcoeff_nonfinite_policy": normalized_policy,
            "kcoeff_nonfinite_fill_value": f"{float(nonfinite_fill_value):.17g}",
            "kcoeff_nonfinite_replaced": str(
                n_nonfinite if normalized_policy == "floor" else 0
            ),
            "exo_k_remove_zeros": str(bool(remove_zeros)).lower(),
            "exo_k_zero_deltalog_min_value": f"{float(zero_deltalog_min_value):.12g}",
            "exo_k_zeros_replaced": str(zero_count if remove_zeros else 0),
            "exo_k_zero_floor": "" if zero_floor is None else f"{zero_floor:.17g}",
        },
    )


def _import_exok():
    cache_directory = _ensure_writable_numba_cache()
    try:
        import exo_k
    except ImportError as exc:
        raise RobertConfigError(
            "exo_k is required for correlated-k spectral binning; install `robert-exoplanets[opacity]`"
        ) from exc
    except RuntimeError as exc:
        if "cannot cache function" in str(exc):
            raise RobertConfigError(
                "exo_k/Numba could not create its cache even after ROBERT selected "
                f"the writable directory {cache_directory}"
            ) from exc
        raise
    return exo_k


def _ensure_writable_numba_cache() -> Path:
    """Select a writable Numba cache before importing cache-enabled libraries."""

    candidates: list[Path] = []
    configured = os.environ.get("NUMBA_CACHE_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if xdg_cache:
        candidates.append(Path(xdg_cache).expanduser() / "robert-exoplanets" / "numba")
    elif sys.platform == "darwin":
        candidates.append(
            Path.home() / "Library" / "Caches" / "robert-exoplanets" / "numba"
        )
    elif os.name == "nt" and local_app_data:
        candidates.append(
            Path(local_app_data).expanduser() / "robert-exoplanets" / "numba"
        )
    else:
        candidates.append(Path.home() / ".cache" / "robert-exoplanets" / "numba")

    user_token = str(os.getuid()) if hasattr(os, "getuid") else str(os.getpid())
    candidates.append(
        Path(tempfile.gettempdir()) / f"robert-exoplanets-{user_token}" / "numba"
    )

    attempted: list[str] = []
    for raw_candidate in candidates:
        try:
            candidate = raw_candidate.resolve()
        except (OSError, RuntimeError):
            attempted.append(str(raw_candidate))
            continue
        if str(candidate) in attempted:
            continue
        attempted.append(str(candidate))
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=candidate,
                prefix=".robert-write-test-",
            ):
                pass
        except OSError:
            continue

        os.environ["NUMBA_CACHE_DIR"] = str(candidate)
        numba = sys.modules.get("numba")
        numba_config = getattr(numba, "config", None)
        if numba_config is not None:
            numba_config.CACHE_DIR = str(candidate)
        return candidate

    raise RobertConfigError(
        "ROBERT could not create a writable Numba cache directory; attempted "
        + ", ".join(attempted)
    )


def _validate_exok_source(native, table: CorrelatedKTable) -> None:
    checks = (
        (np.asarray(native.pgrid, dtype=float), table.pressure_bar, "pressure"),
        (np.asarray(native.tgrid, dtype=float), table.temperature_K, "temperature"),
        (np.asarray(native.ggrid, dtype=float), table.g_samples, "g ordinate"),
        (np.asarray(native.weights, dtype=float), table.g_weights, "g weight"),
    )
    for candidate, reference, label in checks:
        if candidate.shape != reference.shape or not np.allclose(
            candidate, reference, rtol=1.0e-6, atol=1.0e-12
        ):
            raise RobertValidationError(
                f"exo_k and ROBERT disagree on the native {label} grid"
            )
    native_wavenumber = np.asarray(native.wns, dtype=float)
    reference_wavenumber = table.wavenumber_cm_inverse
    if native_wavenumber[0] > native_wavenumber[-1]:
        native_wavenumber = native_wavenumber[::-1]
    if reference_wavenumber[0] > reference_wavenumber[-1]:
        reference_wavenumber = reference_wavenumber[::-1]
    if native_wavenumber.shape != reference_wavenumber.shape or not np.allclose(
        native_wavenumber,
        reference_wavenumber,
        rtol=2.0e-6,
        atol=1.0e-6,
    ):
        raise RobertValidationError(
            "exo_k and ROBERT disagree on the native spectral grid"
        )


__all__ = ["bin_correlated_k_table_with_exok", "load_correlated_k_table_with_exok"]
