"""Optional exo_k bridge for accuracy-preserving correlated-k binning."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robert_exoplanets.core import RobertConfigError, RobertCoverageError, RobertValidationError, SpectralGrid

from .correlated_k import CorrelatedKTable
from .inspectors import file_sha256
from .metadata import spectral_grid_values_in_unit


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
        raise RobertValidationError("exo_k correlated-k binning requires spectral bin edges")
    if int(num) < 2:
        raise RobertValidationError("exo_k binning num must be at least two")
    if not np.isfinite(zero_deltalog_min_value) or zero_deltalog_min_value <= 0.0:
        raise RobertValidationError("exo_k zero_deltalog_min_value must be finite and positive")
    source_path = str(table.metadata.get("source_path", ""))
    if not source_path or not Path(source_path).is_file():
        raise RobertConfigError("exo_k binning requires an exo_k-readable source opacity file")
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
        raise RobertConfigError(f"exo_k could not load correlated-k table: {source_path}") from exc

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

    edge_grid = SpectralGrid(values=spectral_grid.bin_edges, unit=spectral_grid.unit, role="internal")
    target_wavenumber_edges = np.sort(spectral_grid_values_in_unit(edge_grid, "cm^-1"))
    native_edges = np.asarray(native.wnedges, dtype=float)
    tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(native_edges))))
    if target_wavenumber_edges[0] < np.min(native_edges) - tolerance or target_wavenumber_edges[-1] > np.max(
        native_edges
    ) + tolerance:
        raise RobertCoverageError("requested spectral bins extend outside the native exo_k table")

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
        raise RobertValidationError("exo_k failed while binning the correlated-k distribution") from exc

    target_wavenumber = spectral_grid_values_in_unit(spectral_grid, "cm^-1")
    kcoeff = np.asarray(binned.kdata, dtype=float)
    exok_wavenumber = np.asarray(binned.wns, dtype=float)
    if target_wavenumber[0] > target_wavenumber[-1]:
        kcoeff = kcoeff[:, :, ::-1, :]
        exok_wavenumber = exok_wavenumber[::-1]
    if kcoeff.shape[2] != spectral_grid.size:
        raise RobertValidationError("exo_k returned an unexpected number of spectral bins")
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
        raise RobertConfigError(f"correlated-k opacity file does not exist: {table_path}")
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
        raise RobertValidationError("zero_deltalog_min_value must be finite and positive")

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
        raise RobertConfigError(f"exo_k could not load correlated-k table: {table_path}") from exc
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
            "kcoeff_nonfinite_replaced": str(n_nonfinite if normalized_policy == "floor" else 0),
            "exo_k_remove_zeros": str(bool(remove_zeros)).lower(),
            "exo_k_zero_deltalog_min_value": f"{float(zero_deltalog_min_value):.12g}",
            "exo_k_zeros_replaced": str(zero_count if remove_zeros else 0),
            "exo_k_zero_floor": "" if zero_floor is None else f"{zero_floor:.17g}",
        },
    )


def _import_exok():
    try:
        import exo_k
    except ImportError as exc:
        raise RobertConfigError(
            "exo_k is required for correlated-k spectral binning; install `robert-exoplanets[opacity]`"
        ) from exc
    except RuntimeError as exc:
        if "cannot cache function" in str(exc):
            raise RobertConfigError(
                "exo_k/Numba could not create its cache; set NUMBA_CACHE_DIR to a writable directory"
            ) from exc
        raise
    return exo_k


def _validate_exok_source(native, table: CorrelatedKTable) -> None:
    checks = (
        (np.asarray(native.pgrid, dtype=float), table.pressure_bar, "pressure"),
        (np.asarray(native.tgrid, dtype=float), table.temperature_K, "temperature"),
        (np.asarray(native.ggrid, dtype=float), table.g_samples, "g ordinate"),
        (np.asarray(native.weights, dtype=float), table.g_weights, "g weight"),
    )
    for candidate, reference, label in checks:
        if candidate.shape != reference.shape or not np.allclose(candidate, reference, rtol=1.0e-6, atol=1.0e-12):
            raise RobertValidationError(f"exo_k and ROBERT disagree on the native {label} grid")
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
        raise RobertValidationError("exo_k and ROBERT disagree on the native spectral grid")


__all__ = ["bin_correlated_k_table_with_exok", "load_correlated_k_table_with_exok"]
