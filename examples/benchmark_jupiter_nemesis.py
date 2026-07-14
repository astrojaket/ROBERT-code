"""Reproduce the official NEMESIS Jupiter CIRS forward model with ROBERT.

The official example is intentionally external: point ``--nemesis-case`` at
``Example_calculations/Jupiter_CIRS_nadir_thermal_emission`` in a NEMESIS
checkout and ``--nemesis-output`` at the ``cirstest.out`` produced by
``CIRSdrv_wave`` for that same case. ROBERT reads the legacy opacity/CIA files
and exact 71-layer driver state; it does not copy the large third-party tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import PressureGrid, SpectralGrid
from robert_exoplanets.opacity import CorrelatedKOpacityProvider
from robert_exoplanets.rt import (
    GasOpticalDepth,
    LayerOpticalDepth,
    cia_optical_depth,
    normal_emission_geometry,
    random_overlap_species_tau,
    rayleigh_scattering_optical_depth,
    read_cia_table,
    solve_emission_spectrum,
)


SPECIES = ("NH3", "PH3", "C2H2", "C2H6", "CH4_1", "CH4_2", "CH4_3")
GAS_COLUMN = {
    "NH3": 0,
    "PH3": 1,
    "C2H2": 2,
    "C2H6": 4,
    "CH4_1": 8,
    "CH4_2": 9,
    "CH4_3": 10,
}
KTA_FILE = {
    "NH3": "nh3ZERO_5-1500_2pt5.kta",
    "PH3": "ph3ZERO_5-1500_2pt5.kta",
    "C2H2": "c2h2ZERO_5-1500_2pt5.kta",
    "C2H6": "c2h6ZERO_5-1500_2pt5.kta",
    "CH4_1": "ch4ONE_5-1500_2pt5.kta",
    "CH4_2": "ch4TWO_5-1500_2pt5.kta",
    "CH4_3": "ch4THREE_5-1500_2pt5.kta",
}
FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


@dataclass(frozen=True)
class NemesisLayers:
    base_height_km: np.ndarray
    thickness_km: np.ndarray
    base_pressure_atm: np.ndarray
    base_temperature_k: np.ndarray
    total_column_cm2: np.ndarray
    pressure_atm: np.ndarray
    temperature_k: np.ndarray
    gas_column_cm2: np.ndarray
    gas_partial_pressure_atm: np.ndarray
    path_scale: np.ndarray


def run_benchmark(
    nemesis_case: Path,
    output: Path,
    repeats: int = 5,
    nemesis_output: Path | None = None,
    nemesis_seconds: float | None = None,
) -> dict[str, object]:
    case = nemesis_case.expanduser().resolve()
    _require_case(case)
    nemesis_output = (
        ((case / "cirstest.out") if nemesis_output is None else nemesis_output)
        .expanduser()
        .resolve()
    )
    if not nemesis_output.is_file():
        raise FileNotFoundError(
            "missing cirstest.out from a CIRSdrv_wave run on the supplied driver: "
            f"{nemesis_output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    layers = read_nemesis_driver(case / "cirstest.drv")
    wavenumber, nemesis_radiance = read_cirsdrv_native_forward(nemesis_output)
    spectral_grid = SpectralGrid.from_array(wavenumber, unit="cm^-1", role="opacity")
    pressure_grid = _pressure_grid(layers)
    atmosphere = _atmosphere(layers, pressure_grid)

    load_start = perf_counter()
    provider = CorrelatedKOpacityProvider.from_kta_paths(
        {name: case / "ktab" / KTA_FILE[name] for name in SPECIES},
        name="official-nemesis-jupiter-cirs",
        interpolation="log_pressure_temperature_nemesis_k_clip",
        spectral_coordinate="wavenumber_cm_inverse",
    )
    opacity_load_seconds = perf_counter() - load_start
    prepared = provider.prepare(spectral_grid, pressure_grid, SPECIES)
    evaluation_start = perf_counter()
    evaluated = provider.evaluate(atmosphere, prepared)
    opacity_evaluation_seconds = perf_counter() - evaluation_start
    gas_tau = _exact_driver_gas_optical_depth(atmosphere, evaluated, layers)

    cia_path = _cia_path(case)
    cia = cia_optical_depth(
        gas_tau,
        read_cia_table(cia_path, dnu=1.0),
        path_length_cm=layers.thickness_km * 1.0e5,
        normal_hydrogen=True,
        temperature_extrapolation="clip",
        spectral_extrapolation="zero",
        coefficient_interpolation="linear",
    )
    rayleigh = rayleigh_scattering_optical_depth(gas_tau)
    cia = _scale_layer_tau(cia, layers.path_scale)
    rayleigh = _scale_layer_tau(rayleigh, layers.path_scale)

    gas_only = _solve(gas_tau, wavenumber, ())
    gas_cia = _solve(gas_tau, wavenumber, (cia,))
    robert_radiance = _solve(gas_tau, wavenumber, (cia, rayleigh))
    timings = []
    for _ in range(max(1, repeats)):
        start = perf_counter()
        _solve(gas_tau, wavenumber, (cia, rayleigh))
        timings.append(perf_counter() - start)

    metrics = _metrics(nemesis_radiance, robert_radiance)
    passed = metrics["p95_absolute_fraction_bright"] < 0.01
    checksums = {
        path.name: _checksum(path)
        for path in [case / "cirstest.drv", cia_path, nemesis_output]
    }
    report: dict[str, object] = {
        "benchmark": "official NEMESIS Jupiter CIRS nadir thermal-emission driver forward model",
        "status": "pass" if passed else "fail",
        "pass_criterion": "p95 absolute fractional residual below 1% where NEMESIS radiance exceeds 1% of its peak",
        "nemesis_case": str(case),
        "nemesis_output": str(nemesis_output),
        "n_layers": int(layers.temperature_k.size),
        "n_spectral": int(wavenumber.size),
        "wavenumber_range_cm-1": [float(wavenumber[0]), float(wavenumber[-1])],
        "physics": {
            "geometry": "nadir, mu=1",
            "molecular_opacity": "same seven official NEMESIS KTA tables",
            "k_interpolation": "NEMESIS get_k convention: bilinear in log-pressure and temperature, log-k when all first-g corners are positive, otherwise linear-k; clipped",
            "gas_overlap": "random overlap with resorting/rebinning",
            "continuum": "official isotest.tab, normal-H2 CIA",
            "rayleigh": "H2/He extinction included",
            "cia_path": "explicit cirstest.drv layer thicknesses",
            "layer_state": "exact cirstest.drv columns, temperatures, and path scales",
            "bottom_boundary": "deepest-layer blackbody, matching zero NEMESIS surface temperature",
        },
        "metrics": metrics,
        "speed_seconds": {
            "opacity_load": opacity_load_seconds,
            "opacity_evaluation": opacity_evaluation_seconds,
            "forward_median": float(np.median(timings)),
            "forward_min": float(np.min(timings)),
            "forward_repeats": len(timings),
            "nemesis_cirsdrv_wave": nemesis_seconds,
        },
        "input_sha256": checksums,
    }
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez(
        output / "benchmark_arrays.npz",
        wavenumber_cm_inverse=wavenumber,
        nemesis_radiance_w_cm2_sr_cm=nemesis_radiance,
        robert_radiance_w_cm2_sr_cm=robert_radiance,
        robert_gas_only_w_cm2_sr_cm=gas_only,
        robert_gas_cia_w_cm2_sr_cm=gas_cia,
        layer_pressure_atm=layers.pressure_atm,
        layer_temperature_k=layers.temperature_k,
    )
    _plot(
        output / "jupiter_nemesis_benchmark.png",
        wavenumber,
        nemesis_radiance,
        robert_radiance,
        gas_only,
        gas_cia,
    )
    return report


def read_nemesis_driver(path: Path) -> NemesisLayers:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = (
        next(i for i, line in enumerate(lines) if "format of layer data" in line) + 4
    )
    (
        base_h,
        del_h,
        base_p,
        base_t,
        total,
        pressure,
        temperature,
        gas,
        partial_pressure,
    ) = ([] for _ in range(9))
    cursor = start
    for expected_index in range(1, 72):
        header = _numbers(lines[cursor])
        if len(header) != 9 or int(header[0]) != expected_index:
            raise RuntimeError(f"unexpected NEMESIS layer header at line {cursor + 1}")
        _, h0, dh, p0, t0, amount, p, t, _doppler = header
        cursor += 1
        absorber: list[float] = []
        while len(absorber) < 22:
            absorber.extend(_numbers(lines[cursor]))
            cursor += 1
        if len(absorber) != 22:
            raise RuntimeError("unexpected NEMESIS absorber-column record")
        cursor += 1  # one aerosol-continuum amount in this cloud-free case
        base_h.append(h0)
        del_h.append(dh)
        base_p.append(p0)
        base_t.append(t0)
        total.append(amount)
        pressure.append(p)
        temperature.append(t)
        gas.append(absorber[0::2])
        partial_pressure.append(absorber[1::2])
    path_header = _numbers(lines[cursor])
    if len(path_header) < 2 or int(path_header[0]) != 71:
        raise RuntimeError("NEMESIS path section was not found")
    cursor += 1
    scale_by_layer = np.empty(71)
    for _ in range(71):
        values = _numbers(lines[cursor])
        cursor += 1
        _path_index, layer_index, _emission_temperature, scale = values
        scale_by_layer[int(layer_index) - 1] = scale
    return NemesisLayers(
        *(
            np.asarray(item, dtype=float)
            for item in (
                base_h,
                del_h,
                base_p,
                base_t,
                total,
                pressure,
                temperature,
                gas,
                partial_pressure,
                scale_by_layer,
            )
        )
    )


def read_cirsdrv_native_forward(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the unconvolved native-grid block from ``CIRSdrv_wave`` output."""

    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if "!! nwave" in line),
        None,
    )
    if header_index is None:
        raise RuntimeError("NEMESIS CIRSdrv output does not contain an nwave block")
    n_wave = int(_numbers(lines[header_index])[0])
    label_index = next(
        (
            index
            for index in range(header_index + 1, len(lines))
            if lines[index].strip().startswith("Radiance [")
        ),
        None,
    )
    if label_index is None:
        raise RuntimeError("NEMESIS CIRSdrv native radiance block was not found")
    rows = [
        _numbers(line) for line in lines[label_index + 1 : label_index + 1 + n_wave]
    ]
    if len(rows) != n_wave or any(len(row) != 2 for row in rows):
        raise RuntimeError("NEMESIS CIRSdrv native radiance block is incomplete")
    values = np.asarray(rows, dtype=float)
    return values[:, 0], values[:, 1]


def _pressure_grid(layers: NemesisLayers) -> PressureGrid:
    final_edge = layers.pressure_atm[-1] ** 2 / layers.base_pressure_atm[-1]
    edges = np.concatenate((layers.base_pressure_atm, [final_edge]))
    return PressureGrid(
        edges=edges,
        centers=layers.pressure_atm,
        unit="bar",
        metadata={
            "native_pressure_convention": "NEMESIS atmosphere values treated in KTA pressure units"
        },
    )


def _atmosphere(layers: NemesisLayers, pressure_grid: PressureGrid) -> AtmosphereState:
    vmr = layers.gas_partial_pressure_atm / layers.pressure_atm[:, None]
    composition = {name: vmr[:, index] for name, index in GAS_COLUMN.items()}
    composition.update({"H2": vmr[:, 6], "He": vmr[:, 7], "CH4": vmr[:, 8]})
    return AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=layers.temperature_k,
        composition=composition,
        mean_molecular_weight=np.full(71, 2.299),
        metadata={"source": "official NEMESIS cirstest.drv"},
    )


def _exact_driver_gas_optical_depth(
    atmosphere: AtmosphereState, opacity: object, layers: NemesisLayers
) -> GasOpticalDepth:
    total_absorber_column_cm2 = np.sum(layers.gas_column_cm2, axis=1)
    species_columns_cm2 = np.stack(
        [
            layers.gas_partial_pressure_atm[:, GAS_COLUMN[name]]
            / layers.pressure_atm
            * total_absorber_column_cm2
            for name in SPECIES
        ]
    )
    species_tau = opacity.kcoeff * species_columns_cm2[:, :, None, None]
    species_tau *= layers.path_scale[None, :, None, None]
    total_tau = random_overlap_species_tau(
        species_tau, opacity.prepared.g_weights, cutoff=0.0
    )
    edges = atmosphere.pressure_grid.edges
    return GasOpticalDepth(
        atmosphere=atmosphere,
        opacity=opacity,
        species=SPECIES,
        gravity_m_s2=np.full(71, 24.0),
        layer_pressure_thickness_pa=np.abs(np.diff(edges)) * 101325.0,
        layer_column_density_molecules_m2=layers.total_column_cm2 * 1.0e4,
        species_column_density_molecules_m2=species_columns_cm2 * 1.0e4,
        species_tau=species_tau,
        total_tau=total_tau,
        metadata={
            "gas_combination": "random_overlap",
            "column_model": "exact_nemesis_driver",
            "path_scale": "cirstest.drv",
        },
    )


def _scale_layer_tau(layer: LayerOpticalDepth, scale: np.ndarray) -> LayerOpticalDepth:
    return LayerOpticalDepth(
        name=layer.name,
        tau=layer.tau * scale[:, None],
        spectral_grid=layer.spectral_grid,
        pressure_grid=layer.pressure_grid,
        kind=layer.kind,
        phase_function_moments=layer.phase_function_moments,
        metadata={**dict(layer.metadata), "path_scale": "cirstest.drv"},
    )


def _solve(
    gas_tau: GasOpticalDepth,
    wavenumber: np.ndarray,
    additional: tuple[LayerOpticalDepth, ...],
) -> np.ndarray:
    spectrum = solve_emission_spectrum(
        gas_tau,
        geometry=normal_emission_geometry(),
        bottom_boundary="blackbody",
        additional_optical_depths=additional,
        thermal_integration_backend="auto",
    )
    return np.asarray(spectrum.values) * 1.0e-6 / wavenumber**2


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    residual = candidate - reference
    bright = reference >= 0.01 * np.max(reference)
    fractional = np.abs(residual[bright] / reference[bright])
    return {
        "rmse_w_cm-2_sr-1_cm": float(np.sqrt(np.mean(residual**2))),
        "median_absolute_fraction_bright": float(np.median(fractional)),
        "p95_absolute_fraction_bright": float(np.percentile(fractional, 95.0)),
        "maximum_absolute_fraction_bright": float(np.max(fractional)),
        "reduced_chi_square_at_5_percent": float(
            np.mean((residual / (0.05 * reference)) ** 2)
        ),
    }


def _plot(
    path: Path,
    wn: np.ndarray,
    nemesis: np.ndarray,
    robert: np.ndarray,
    gas: np.ndarray,
    gas_cia: np.ndarray,
) -> None:
    scale = 1.0e9
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax.plot(wn, nemesis * scale, color="#17131f", lw=1.5, label="NEMESIS CIRSdrv_wave")
    ax.plot(
        wn, robert * scale, color="#7251b5", lw=1.1, label="ROBERT full matched physics"
    )
    ax.plot(
        wn,
        gas_cia * scale,
        color="#b8a1df",
        lw=0.8,
        alpha=0.8,
        label="ROBERT gas + CIA",
    )
    ax.plot(
        wn, gas * scale, color="#d8ccef", lw=0.8, alpha=0.8, label="ROBERT gas only"
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"Radiance (nW cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)")
    ax.set_title("Official Jupiter CIRS benchmark: ROBERT versus NEMESIS")
    ax.legend(ncol=2, frameon=False)
    fractional = (
        100.0 * (robert - nemesis) / np.maximum(nemesis, 0.01 * np.max(nemesis))
    )
    residual_ax.axhline(0.0, color="#17131f", lw=0.8)
    residual_ax.plot(wn, fractional, color="#7251b5", lw=0.9)
    residual_ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    residual_ax.set_ylabel("Residual (%)")
    for axis in (ax, residual_ax):
        axis.grid(alpha=0.15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _numbers(line: str) -> list[float]:
    return [float(token.replace("D", "E")) for token in FLOAT.findall(line)]


def _checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_case(case: Path) -> None:
    required = [
        case / "cirstest.drv",
        _cia_path(case),
        *(case / "ktab" / KTA_FILE[name] for name in SPECIES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing official NEMESIS benchmark inputs: " + ", ".join(missing)
        )


def _cia_path(case: Path) -> Path:
    local = case / "isotest.tab"
    if local.is_file():
        return local
    return case.parents[1] / "raddata" / "isotest.tab"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemesis-case", type=Path, required=True)
    parser.add_argument("--nemesis-output", type=Path)
    parser.add_argument("--nemesis-seconds", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/outputs/jupiter_nemesis_benchmark"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    return parser


if __name__ == "__main__":
    result = run_benchmark(**vars(_parser().parse_args()))
    print(json.dumps(result, indent=2))
