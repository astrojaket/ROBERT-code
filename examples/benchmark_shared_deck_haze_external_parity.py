"""Benchmark the shared finite deck/haze model against stable petitRADTRANS.

Four aerosol states are evaluated in both geometries: clear, deck only, haze
only, and deck plus haze. ROBERT and pRT independently evaluate their molecular
and continuum opacities; comparison therefore emphasizes the differential
cloud effect, while clear-spectrum residuals diagnose opacity differences.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    OpacitySamplingProvider,
    ParameterizedDeckHazeCloudModel,
    PressureGrid,
    assemble_gas_optical_depth,
    cia_optical_depth,
    gauss_legendre_disk_geometry,
    hydrostatic_path_geometry,
    load_nemesispy_cia_table,
    planck_radiance_wavelength,
    rayleigh_scattering_optical_depth,
    solve_absorption_transmission,
    solve_emission_spectrum,
)
from robert_exoplanets.diagnostics.benchmark_style import (
    REFERENCE_COLOR,
    ROBERT_COLOR,
)

try:
    from examples.benchmark_official_picaso_molecular_cloud_parity import (
        EXOMOL,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _inverse_square_hydrostatic_profiles,
        _make_science_contract,
    )
except ModuleNotFoundError:
    from benchmark_official_picaso_molecular_cloud_parity import (
        EXOMOL,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _inverse_square_hydrostatic_profiles,
        _make_science_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_petitradtrans3_shared_deck_haze.py")
DEFAULT_INPUT = ROOT / "opacity_data" / "petitRADTRANS" / "input_data"
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "shared_deck_haze_external_parity"
CASES = ("clear", "deck", "haze", "deck_haze")


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--petitradtrans-python",
        type=Path,
        default=Path("/Users/jaketaylor/miniforge3/envs/petitradtrans-stable/bin/python"),
    )
    parser.add_argument("--petitradtrans-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--opacity-stride", type=int, default=15)
    parser.add_argument("--output-bins", type=int, default=180)
    parser.add_argument("--convergence-layers", default="48,80,120,160")
    args = parser.parse_args()
    convergence_layers = tuple(
        dict.fromkeys(int(value) for value in args.convergence_layers.split(",") if value)
    )
    return run(
        args.petitradtrans_python,
        args.petitradtrans_input,
        args.output_dir,
        layers=args.layers,
        opacity_stride=args.opacity_stride,
        output_bins=args.output_bins,
        convergence_layers=convergence_layers,
    )


def run(
    petitradtrans_python: Path,
    petitradtrans_input: Path,
    output_dir: Path,
    *,
    layers: int,
    opacity_stride: int,
    output_bins: int,
    convergence_layers: tuple[int, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_counts = tuple(dict.fromkeys((*convergence_layers, layers)))
    results = {}
    for n_layer in layer_counts:
        contract = _contract(n_layer)
        started = perf_counter()
        robert = _evaluate_robert(contract, opacity_stride)
        robert_seconds = perf_counter() - started
        with tempfile.TemporaryDirectory(prefix="robert-deck-haze-") as scratch:
            contract_path = Path(scratch) / "contract.npz"
            external_path = Path(scratch) / "petitradtrans.npz"
            np.savez_compressed(contract_path, **contract)
            _run_external(
                petitradtrans_python,
                petitradtrans_input,
                contract_path,
                external_path,
            )
            with np.load(external_path, allow_pickle=False) as archive:
                external = {name: np.array(archive[name], copy=True) for name in archive.files}
        compact = _compact(robert, external, contract, output_bins)
        metrics = _metrics(compact)
        results[str(n_layer)] = {
            "metrics": metrics,
            "robert_seconds": robert_seconds,
            "petitradtrans": json.loads(str(external["metadata_json"])),
        }
        if n_layer == layers:
            np.savez_compressed(
                output_dir / "shared_deck_haze_external_parity_spectra.npz",
                **compact,
            )
            _plot(
                output_dir / "shared_deck_haze_external_parity.png",
                compact,
            )

    primary = results[str(layers)]["metrics"]
    report = {
        "schema_version": 1,
        "benchmark": "shared_finite_deck_power_law_haze_emission_transmission",
        "interpretation": (
            "ROBERT and stable pRT evaluate independent molecular/CIA opacity databases. "
            "Differential aerosol effects are the primary parity observable."
        ),
        "physical_contract": {
            "cases": list(CASES),
            "deck_top_pressure_bar": 1.0e-3,
            "deck_integrated_vertical_optical_depth": 0.3,
            "deck_single_scattering_albedo": 0.0,
            "haze_mass_extinction_cm2_g_at_1um": 1.0e-3,
            "haze_slope": -4.0,
            "haze_single_scattering_albedo": 1.0,
            "pressure_layers": list(layer_counts),
            "gas_species": list(SPECIES),
        },
        "sampling": {
            "robert_exomolop_stride": opacity_stride,
            "petitradtrans_correlated_k_resolution": 1000,
            "comparison_bins": output_bins,
            "primary_layers": layers,
        },
        "external_comparison_targets_not_truth_criteria": {
            "independent_emission_cloud_effect_rms_ppm": 10.0,
            "independent_transmission_cloud_effect_rms_ppm": 20.0,
            "interpretation": (
                "Targets identify differences to explain; pRT is an independent "
                "reference, not the implementation specification."
            ),
        },
        "primary_metrics": primary,
        "vertical_convergence": results,
    }
    (output_dir / "shared_deck_haze_external_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def _contract(n_layer: int) -> dict[str, np.ndarray]:
    contract = _make_science_contract(96, 24, n_layer)
    contract.update(
        deck_top_pressure_bar=np.array(1.0e-3),
        deck_optical_depth=np.array(0.3),
        haze_mass_extinction_cm2_g=np.array(1.0e-3),
        haze_reference_wavelength_micron=np.array(1.0),
        haze_slope=np.array(-4.0),
    )
    return contract


def _evaluate_robert(contract: dict[str, np.ndarray], opacity_stride: int):
    pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    pressure_layer = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure_layer,
        unit="bar",
        name="shared deck haze parity",
    )
    composition = {
        name: np.asarray(contract["gas_vmr"][:, index], dtype=float)
        for index, name in enumerate(SPECIES)
    }
    composition["H2"] = np.full(pressure_layer.size, float(contract["h2_vmr"]))
    composition["He"] = np.asarray(contract["he_vmr"], dtype=float)
    mean_molecular_weight = sum(
        composition[name] * MOLECULAR_WEIGHTS[name] for name in composition
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=0.5
        * (contract["temperature_level_k"][:-1] + contract["temperature_level_k"][1:]),
        temperature_edges=contract["temperature_level_k"],
        composition=composition,
        mean_molecular_weight=mean_molecular_weight,
    )
    hydrostatic = _inverse_square_hydrostatic_profiles(contract)
    provider = OpacitySamplingProvider.from_exomol_paths(
        {name: EXOMOL / f"{name}.h5" for name in SPECIES},
        interpolation="log_pressure_temperature_log_xsec_clip",
        checksum=False,
    )
    spectral_grid = provider.native_spectral_grid(
        sampling=opacity_stride,
        wavelength_bounds_micron=(1.0, 12.0),
        name=f"ExoMolOP stride {opacity_stride}",
    )
    prepared = provider.prepare(spectral_grid, pressure_grid, SPECIES)
    gas = assemble_gas_optical_depth(
        atmosphere,
        provider.evaluate(atmosphere, prepared),
        gravity_m_s2=hydrostatic["column_gravity_m_s2"],
    )
    base = (
        cia_optical_depth(
            gas,
            load_nemesispy_cia_table(),
            temperature_extrapolation="clip",
            spectral_extrapolation="zero",
        ),
        rayleigh_scattering_optical_depth(gas),
    )
    cloud_model = ParameterizedDeckHazeCloudModel()
    cloud = cloud_model.evaluate(
        gas,
        {
            "log_cloud_top_pressure_bar": np.log10(float(contract["deck_top_pressure_bar"])),
            "log_cloud_optical_depth": np.log10(float(contract["deck_optical_depth"])),
            "log_haze_mass_extinction": np.log10(
                float(contract["haze_mass_extinction_cm2_g"])
            ),
            "haze_slope": float(contract["haze_slope"]),
        },
    )
    geometry_atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=contract["temperature_level_k"][1:],
        temperature_edges=contract["temperature_level_k"],
        composition=composition,
        mean_molecular_weight=hydrostatic["mean_molecular_weight_level"][1:],
    )
    path_geometry = hydrostatic_path_geometry(
        geometry_atmosphere,
        gravity_m_s2=hydrostatic["geometry_gravity_m_s2"],
        reference_radius_m=float(contract["planet_radius_m"]),
        reference_pressure=float(contract["reference_pressure_bar"]),
        reference_pressure_unit="bar",
    )
    output: dict[str, np.ndarray] = {"wavelength_micron": spectral_grid.values}
    additions = {
        "clear": base,
        "deck": (*base, cloud[0]),
        "haze": (*base, cloud[1]),
        "deck_haze": (*base, *cloud),
    }
    for case, extra in additions.items():
        emission = solve_emission_spectrum(
            gas,
            geometry=gauss_legendre_disk_geometry(6),
            additional_optical_depths=extra,
            multiple_scattering_backend="sh4",
            planet_radius_m=float(contract["planet_radius_m"]),
            star_radius_m=float(contract["star_radius_m"]),
            star_temperature_k=float(contract["star_temperature_k"]),
        )
        transmission = solve_absorption_transmission(
            gas,
            path_geometry,
            star_radius_m=float(contract["star_radius_m"]),
            additional_optical_depths=extra,
            impact_quadrature_order=8,
        )
        output[f"{case}_eclipse_depth"] = np.asarray(emission.values)
        output[f"{case}_transit_depth"] = np.asarray(transmission.transit_depth.values)
    output["bottom_radius_m"] = np.array(path_geometry.bottom_radius_m)
    output["top_radius_m"] = np.array(path_geometry.top_radius_m)
    return output


def _run_external(python: Path, input_data: Path, contract: Path, output: Path) -> None:
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "prt-mpl")
    environment["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "prt-numba")
    subprocess.run(
        [
            str(python),
            str(RUNNER),
            str(contract),
            str(output),
            "--input-data",
            str(input_data),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def _compact(robert, external, contract, output_bins: int):
    edges = np.geomspace(1.0, 12.0, output_bins + 1)
    wavelength = np.sqrt(edges[:-1] * edges[1:])
    output = {"wavelength_micron": wavelength}
    robert_wavelength = np.asarray(robert["wavelength_micron"])
    prt_wavelength = np.asarray(external["wavelength_cm"]) * 1.0e4
    area_ratio = (float(contract["planet_radius_m"]) / float(contract["star_radius_m"])) ** 2
    stellar_flux = np.pi * planck_radiance_wavelength(
        prt_wavelength, float(contract["star_temperature_k"])
    )
    for case in CASES:
        values = {
            "robert_emission": robert[f"{case}_eclipse_depth"],
            "robert_transmission": robert[f"{case}_transit_depth"],
            "petitradtrans_emission": (
                np.asarray(external[f"{case}_flux_cgs_per_cm"]) * 0.1 / stellar_flux * area_ratio
            ),
            "petitradtrans_transmission": (
                np.asarray(external[f"{case}_transit_radius_cm"]) * 1.0e-2
                / float(contract["star_radius_m"])
            ) ** 2,
        }
        for name, value in values.items():
            native = robert_wavelength if name.startswith("robert") else prt_wavelength
            output[f"{case}_{name}"] = _bin(native, value, edges)
    return output


def _bin(wavelength, values, edges):
    order = np.argsort(wavelength)
    wavelength = np.asarray(wavelength)[order]
    values = np.asarray(values)[order]
    centers = np.sqrt(edges[:-1] * edges[1:])
    result = np.empty(centers.size)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (wavelength >= left) & (wavelength < right)
        result[index] = (
            np.mean(values[selected])
            if np.any(selected)
            else np.interp(centers[index], wavelength, values)
        )
    return result


def _metrics(compact):
    output = {}
    for case in CASES:
        robert_emission = compact[f"{case}_robert_emission"]
        prt_emission = compact[f"{case}_petitradtrans_emission"]
        robert_transmission = compact[f"{case}_robert_transmission"]
        prt_transmission = compact[f"{case}_petitradtrans_transmission"]
        emission_effect_difference = (
            (robert_emission - compact["clear_robert_emission"])
            - (prt_emission - compact["clear_petitradtrans_emission"])
        ) * 1.0e6
        transmission_effect_difference = (
            (robert_transmission - compact["clear_robert_transmission"])
            - (prt_transmission - compact["clear_petitradtrans_transmission"])
        ) * 1.0e6
        output[case] = {
            "emission_absolute_rms_difference_ppm": _rms(
                (robert_emission - prt_emission) * 1.0e6
            ),
            "transmission_absolute_rms_difference_ppm": _rms(
                (robert_transmission - prt_transmission) * 1.0e6
            ),
            "emission_effect_rms_difference_ppm": _rms(emission_effect_difference),
            "transmission_effect_rms_difference_ppm": _rms(
                transmission_effect_difference
            ),
            "emission_effect_max_abs_difference_ppm": float(
                np.max(np.abs(emission_effect_difference))
            ),
            "transmission_effect_max_abs_difference_ppm": float(
                np.max(np.abs(transmission_effect_difference))
            ),
        }
    return output


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _plot(path: Path, compact) -> None:
    wavelength = compact["wavelength_micron"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for column, geometry in enumerate(("emission", "transmission")):
        top = axes[0, column]
        residual = axes[1, column]
        scale = 1.0e6
        for case, alpha in zip(CASES, (0.45, 0.65, 0.8, 1.0), strict=True):
            robert = compact[f"{case}_robert_{geometry}"] * scale
            prt = compact[f"{case}_petitradtrans_{geometry}"] * scale
            top.plot(wavelength, prt, color=REFERENCE_COLOR, alpha=alpha, lw=1.0)
            top.plot(wavelength, robert, color=ROBERT_COLOR, alpha=alpha, lw=1.0)
            effect = (
                (compact[f"{case}_robert_{geometry}"] - compact[f"clear_robert_{geometry}"])
                - (
                    compact[f"{case}_petitradtrans_{geometry}"]
                    - compact[f"clear_petitradtrans_{geometry}"]
                )
            ) * scale
            residual.plot(wavelength, effect, lw=1.0, alpha=alpha, label=case)
        top.set_title(f"{geometry.capitalize()} spectra")
        top.set_ylabel("Depth (ppm)")
        residual.axhline(0.0, color="0.5", lw=0.7)
        residual.set_ylabel("Cloud-effect difference (ppm)")
        residual.set_xlabel("Wavelength (micron)")
        residual.legend(frameon=False, fontsize=8, ncol=2)
        top.set_xscale("log")
        residual.set_xscale("log")
    figure.suptitle("Shared deck/haze: ROBERT (purple) vs pRT 3.3.3 (black)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
