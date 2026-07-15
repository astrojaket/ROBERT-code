"""Independent matched-optical-depth audit of transmission geometry choices.

PICASO evaluates the molecular and aerosol layer optical depths once. ROBERT
then reuses those exact arrays while varying one pressure-radius or annulus
integration convention at a time. This separates geometry differences from
opacity-database and cloud-parameterization differences.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robert-mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import (
    AtmosphereState,
    EvaluatedCorrelatedKOpacity,
    LayerOpticalDepth,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    hydrostatic_path_geometry,
    inverse_square_hydrostatic_path_geometry,
    solve_absorption_transmission,
)

try:
    from examples.benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _inverse_square_hydrostatic_profiles,
    )
    from examples.benchmark_shared_deck_haze_external_parity import (
        CASES,
        ROOT,
        _bin,
        _contract,
        _rms,
    )
    from examples.benchmark_shared_deck_haze_picaso import _run_external
except ModuleNotFoundError:
    from benchmark_official_picaso_molecular_cloud_parity import (
        DEFAULT_DATABASE,
        DEFAULT_REFERENCE,
        MOLECULAR_WEIGHTS,
        SPECIES,
        _inverse_square_hydrostatic_profiles,
    )
    from benchmark_shared_deck_haze_external_parity import (
        CASES,
        ROOT,
        _bin,
        _contract,
        _rms,
    )
    from benchmark_shared_deck_haze_picaso import _run_external


DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "transmission_geometry_audit"
IMPACT_ORDERS = (2, 4, 6, 8, 12, 16, 24)
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
ATOMIC_MASS_KG = 1.66053906660e-27


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picaso-python",
        type=Path,
        default=Path("/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python"),
    )
    parser.add_argument("--picaso-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--picaso-database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--opacity-stride", type=int, default=5)
    parser.add_argument("--output-bins", type=int, default=180)
    args = parser.parse_args()
    report = run(
        args.picaso_python,
        args.picaso_reference,
        args.picaso_database,
        args.output_dir,
        layers=args.layers,
        opacity_stride=args.opacity_stride,
        output_bins=args.output_bins,
    )
    print(json.dumps(report, indent=2))
    return report


def run(
    picaso_python: Path,
    picaso_reference: Path,
    picaso_database: Path,
    output_dir: Path,
    *,
    layers: int,
    opacity_stride: int,
    output_bins: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _contract(layers)
    contract_path = output_dir / "matched_physical_contract.npz"
    picaso_path = output_dir / "picaso_evaluated_optical_depths.npz"
    np.savez_compressed(contract_path, **contract)
    started = perf_counter()
    _run_external(
        picaso_python,
        picaso_reference,
        picaso_database,
        contract_path,
        picaso_path,
        opacity_stride,
    )
    external_seconds = perf_counter() - started
    with np.load(picaso_path, allow_pickle=False) as archive:
        picaso = {name: np.array(archive[name], copy=True) for name in archive.files}

    state = _matched_state(contract, picaso)
    baseline = state["picaso_level_inverse_square"]
    geometries = _geometry_variants(contract, state)
    edges = np.geomspace(1.0, 12.0, output_bins + 1)
    picaso_depth = {
        case: _bin(
            picaso["wavelength_micron"], picaso[f"{case}_transit_depth"], edges
        )
        for case in CASES
    }

    variant_spectra = {}
    variant_metrics = {}
    for name, geometry in geometries.items():
        spectra = _solve_cases(state, picaso, geometry, impact_order=8, edges=edges)
        variant_spectra[name] = spectra
        variant_metrics[name] = {
            "geometry": _geometry_metrics(geometry, baseline),
            "spectra": _comparison_metrics(spectra, picaso_depth),
        }

    algorithm_spectra = _algorithm_variants(
        state,
        picaso,
        baseline,
        edges=edges,
    )
    algorithm_metrics = {
        name: _comparison_metrics(spectra, picaso_depth)
        for name, spectra in algorithm_spectra.items()
    }

    impact_spectra = {}
    impact_metrics = {}
    finest = _solve_cases(
        state, picaso, baseline, impact_order=max(IMPACT_ORDERS), edges=edges
    )
    for order in IMPACT_ORDERS:
        spectra = _solve_cases(
            state, picaso, baseline, impact_order=order, edges=edges
        )
        impact_spectra[str(order)] = spectra
        impact_metrics[str(order)] = {
            case: {
                "absolute_to_order24_rms_ppm": _rms(
                    (spectra[case] - finest[case]) * 1.0e6
                ),
                "cloud_effect_to_order24_rms_ppm": _rms(
                    (
                        (spectra[case] - spectra["clear"])
                        - (finest[case] - finest["clear"])
                    )
                    * 1.0e6
                ),
            }
            for case in CASES
        }

    report = {
        "schema_version": 1,
        "benchmark": "matched_optical_depth_transmission_geometry_audit",
        "interpretation": (
            "PICASO evaluated gas, Rayleigh, and aerosol layer optical depths are "
            "held fixed. Residuals therefore diagnose hydrostatic radius mapping, "
            "reference anchoring, and annulus integration rather than opacity "
            "databases."
        ),
        "resolution": {
            "layers": layers,
            "opacity_stride": opacity_stride,
            "comparison_bins": output_bins,
            "impact_orders": list(IMPACT_ORDERS),
        },
        "picaso_metadata": json.loads(str(picaso["metadata_json"])),
        "external_seconds": external_seconds,
        "geometry_variants": variant_metrics,
        "transmission_algorithm_variants": algorithm_metrics,
        "impact_quadrature_convergence": impact_metrics,
    }
    (output_dir / "transmission_geometry_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _save_spectra(
        output_dir / "transmission_geometry_audit_spectra.npz",
        np.sqrt(edges[:-1] * edges[1:]),
        picaso_depth,
        variant_spectra,
        algorithm_spectra,
        impact_spectra,
    )
    _plot(
        output_dir / "transmission_geometry_audit.png",
        np.sqrt(edges[:-1] * edges[1:]),
        picaso_depth,
        variant_spectra,
        impact_metrics,
        geometries,
        baseline,
    )
    _plot_algorithms(
        output_dir / "transmission_algorithm_audit.png",
        np.sqrt(edges[:-1] * edges[1:]),
        picaso_depth,
        algorithm_spectra,
        algorithm_metrics,
    )
    return report


def _matched_state(contract, picaso):
    pressure_edges = np.asarray(contract["pressure_edges_bar"], dtype=float)
    pressure = np.sqrt(pressure_edges[:-1] * pressure_edges[1:])
    pressure_grid = PressureGrid(
        edges=pressure_edges,
        centers=pressure,
        unit="bar",
        name="matched PICASO optical-depth geometry audit",
    )
    wavelength = np.asarray(picaso["wavelength_micron"], dtype=float)
    spectral_grid = SpectralGrid.from_array(
        wavelength, unit="micron", role="opacity", name="PICASO evaluated opacity"
    )
    composition = {
        name: np.asarray(contract["gas_vmr"][:, index], dtype=float)
        for index, name in enumerate(SPECIES)
    }
    composition["H2"] = np.full(pressure.size, float(contract["h2_vmr"]))
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
    prepared = PreparedCorrelatedKOpacity(
        provider_name="PICASO matched evaluated opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key=f"geometry-audit-{pressure.size}-{wavelength.size}",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure.size, wavelength.size, 1)),
        unit="m^2/molecule",
    )
    gas = assemble_gas_optical_depth(
        atmosphere, opacity, gravity_m_s2=float(contract["gravity_m_s2"])
    )
    gas = replace(
        gas,
        species_tau=None,
        total_tau=(
            np.asarray(picaso["clear_gas_tau"], dtype=float)
            + np.asarray(picaso["clear_rayleigh_tau"], dtype=float)
        )[:, :, None],
        metadata={
            **dict(gas.metadata),
            "source": "PICASO evaluated taugas + tauray",
        },
    )
    hydrostatic = _inverse_square_hydrostatic_profiles(contract)
    picaso_state = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=contract["temperature_level_k"][1:],
        temperature_edges=contract["temperature_level_k"],
        composition=composition,
        mean_molecular_weight=hydrostatic["mean_molecular_weight_level"][1:],
    )
    reconstructed = hydrostatic_path_geometry(
        picaso_state,
        gravity_m_s2=hydrostatic["geometry_gravity_m_s2"],
        reference_radius_m=float(contract["planet_radius_m"]),
        reference_pressure=float(contract["reference_pressure_bar"]),
        reference_pressure_unit="bar",
    )
    pressure_level_pa = pressure_edges * 1.0e5
    pressure_center_pa = pressure * 1.0e5
    edge_radius = np.asarray(picaso["picaso_level_radius_m"], dtype=float)
    scale_height = (edge_radius[:-1] - edge_radius[1:]) / np.log(
        pressure_level_pa[1:] / pressure_level_pa[:-1]
    )
    center_radius = edge_radius[:-1] - scale_height * np.log(
        pressure_center_pa / pressure_level_pa[:-1]
    )
    baseline = replace(
        reconstructed,
        scale_height_m=scale_height,
        edge_radius_m=edge_radius,
        center_radius_m=center_radius,
        metadata={
            **dict(reconstructed.metadata),
            "geometry_source": "PICASO exported level radii",
        },
    )
    return {
        "gas": gas,
        "atmosphere": atmosphere,
        "picaso_state": picaso_state,
        "picaso_gravity": hydrostatic["geometry_gravity_m_s2"],
        "hydrostatic": hydrostatic,
        "contract": contract,
        "picaso": picaso,
        "picaso_level_inverse_square": baseline,
        "picaso_level_reconstructed_inverse_square": reconstructed,
        "star_radius_m": float(contract["star_radius_m"]),
    }


def _geometry_variants(contract, state):
    baseline = state["picaso_level_inverse_square"]
    reference_pressure = float(contract["reference_pressure_bar"])
    reference_radius = float(contract["planet_radius_m"])
    reference_gravity = float(contract["gravity_m_s2"])
    variants = {
        "picaso_level_inverse_square": baseline,
        "picaso_level_reconstructed_inverse_square": state[
            "picaso_level_reconstructed_inverse_square"
        ],
        "picaso_level_constant_gravity": hydrostatic_path_geometry(
            state["picaso_state"],
            gravity_m_s2=reference_gravity,
            reference_radius_m=reference_radius,
            reference_pressure=reference_pressure,
            reference_pressure_unit="bar",
        ),
        "layer_center_inverse_square": inverse_square_hydrostatic_path_geometry(
            state["atmosphere"],
            reference_radius_m=reference_radius,
            reference_pressure=reference_pressure,
            reference_gravity_m_s2=reference_gravity,
            reference_pressure_unit="bar",
        ),
        "layer_center_constant_gravity": hydrostatic_path_geometry(
            state["atmosphere"],
            gravity_m_s2=reference_gravity,
            reference_radius_m=reference_radius,
            reference_pressure=reference_pressure,
            reference_pressure_unit="bar",
        ),
    }
    for pressure_bar in (1.0, 0.1):
        variants[f"remapped_anchor_{pressure_bar:g}bar"] = replace(
            baseline,
            reference_radius_m=baseline.radius_at_pressure(pressure_bar * 1.0e5),
            reference_pressure_pa=pressure_bar * 1.0e5,
            metadata={
                **dict(baseline.metadata),
                "reference_anchor_remapped_without_changing_geometry": "true",
            },
        )
    correct_radius_at_1bar = baseline.radius_at_pressure(1.0e5)
    radius_shift = reference_radius - correct_radius_at_1bar
    variants["misanchored_same_radius_at_1bar"] = replace(
        baseline,
        reference_radius_m=reference_radius,
        reference_pressure_pa=1.0e5,
        edge_radius_m=baseline.edge_radius_m + radius_shift,
        center_radius_m=baseline.center_radius_m + radius_shift,
        metadata={
            **dict(baseline.metadata),
            "reference_anchor_misassigned": "same numerical radius moved from 10 to 1 bar",
        },
    )
    return variants


def _solve_cases(state, picaso, geometry, *, impact_order, edges):
    output = {}
    wavelength = np.asarray(picaso["wavelength_micron"], dtype=float)
    for case in CASES:
        additional = ()
        if case != "clear":
            additional = (
                LayerOpticalDepth(
                    name=f"PICASO {case} cloud extinction",
                    tau=np.asarray(picaso[f"{case}_cloud_tau"], dtype=float),
                    spectral_grid=state["gas"].spectral_grid,
                    pressure_grid=state["gas"].pressure_grid,
                ),
            )
        result = solve_absorption_transmission(
            state["gas"],
            geometry,
            star_radius_m=state["star_radius_m"],
            additional_optical_depths=additional,
            impact_quadrature_order=impact_order,
        )
        output[case] = _bin(wavelength, result.transit_depth.values, edges)
    return output


def _algorithm_variants(state, picaso, geometry, *, edges):
    """Separate shell extinction and annulus-integration conventions."""

    output = {
        "robert_uniform_shell_gauss": {},
        "picaso_density_shell_gauss": {},
        "uniform_shell_level_rectangle": {},
        "picaso_density_level_rectangle": {},
    }
    wavelength = np.asarray(picaso["wavelength_micron"], dtype=float)
    for case in CASES:
        vertical_tau = _case_vertical_tau(state, picaso, case)
        density_tau = _picaso_density_equivalent_vertical_tau(
            state,
            geometry,
            vertical_tau,
        )
        output["robert_uniform_shell_gauss"][case] = _solve_vertical_tau(
            state,
            geometry,
            vertical_tau,
            edges,
        )
        output["picaso_density_shell_gauss"][case] = _solve_vertical_tau(
            state,
            geometry,
            density_tau,
            edges,
        )
        output["uniform_shell_level_rectangle"][case] = _bin(
            wavelength,
            _solve_level_rectangle(
                vertical_tau,
                geometry,
                star_radius_m=state["star_radius_m"],
                extinction_model="uniform_shell",
                state=state,
            ),
            edges,
        )
        output["picaso_density_level_rectangle"][case] = _bin(
            wavelength,
            _solve_level_rectangle(
                vertical_tau,
                geometry,
                star_radius_m=state["star_radius_m"],
                extinction_model="picaso_density",
                state=state,
            ),
            edges,
        )
    return output


def _case_vertical_tau(state, picaso, case):
    tau = np.asarray(state["gas"].total_tau[:, :, 0], dtype=float).copy()
    if case != "clear":
        tau += np.asarray(picaso[f"{case}_cloud_tau"], dtype=float)
    return tau


def _picaso_density_m_inverse(state):
    picaso = state["picaso"]
    pressure_level_pa = np.asarray(picaso["picaso_level_pressure_pa"], dtype=float)
    temperature_level = np.asarray(
        picaso["picaso_level_temperature_k"], dtype=float
    )
    mmw_layer_kg = (
        np.asarray(picaso["picaso_layer_mmw_amu"], dtype=float) * ATOMIC_MASS_KG
    )
    column_mass_kg_m2 = np.asarray(
        picaso["picaso_layer_column_mass_kg_m2"], dtype=float
    )
    outer_number_density_m3 = pressure_level_pa[:-1] / (
        BOLTZMANN_CONSTANT_J_K * temperature_level[:-1]
    )
    return mmw_layer_kg * outer_number_density_m3 / column_mass_kg_m2


def _picaso_density_equivalent_vertical_tau(state, geometry, vertical_tau):
    shell_thickness = np.abs(np.diff(geometry.edge_radius_m))
    return vertical_tau * (
        _picaso_density_m_inverse(state) * shell_thickness
    )[:, None]


def _solve_vertical_tau(state, geometry, vertical_tau, edges):
    gas = replace(
        state["gas"],
        total_tau=np.asarray(vertical_tau, dtype=float)[:, :, None],
    )
    result = solve_absorption_transmission(
        gas,
        geometry,
        star_radius_m=state["star_radius_m"],
        impact_quadrature_order=8,
    )
    return _bin(
        state["gas"].spectral_grid.values,
        result.transit_depth.values,
        edges,
    )


def _solve_level_rectangle(
    vertical_tau,
    geometry,
    *,
    star_radius_m,
    extinction_model,
    state,
):
    """Independent vectorized form of PICASO's Brown (2001) level sum."""

    radius = np.asarray(geometry.edge_radius_m, dtype=float)
    shell_thickness = np.abs(np.diff(radius))
    if extinction_model == "uniform_shell":
        extinction_m_inverse = vertical_tau / shell_thickness[:, None]
    elif extinction_model == "picaso_density":
        extinction_m_inverse = vertical_tau * _picaso_density_m_inverse(state)[:, None]
    else:  # pragma: no cover - private benchmark contract
        raise ValueError(f"unknown extinction model {extinction_model!r}")

    transmission = np.ones((radius.size, vertical_tau.shape[1]), dtype=float)
    for tangent_index in range(1, radius.size):
        impact = radius[tangent_index]
        outer = np.sqrt(
            np.maximum(radius[:tangent_index] ** 2 - impact**2, 0.0)
        )
        inner = np.sqrt(
            np.maximum(radius[1 : tangent_index + 1] ** 2 - impact**2, 0.0)
        )
        segment = outer - inner
        slant_tau = 2.0 * np.einsum(
            "l,ls->s",
            segment,
            extinction_m_inverse[:tangent_index],
            optimize=True,
        )
        transmission[tangent_index] = np.exp(-slant_tau)

    dz = np.zeros_like(radius)
    dz[1:] = radius[:-1] - radius[1:]
    effective_radius_squared = radius[-1] ** 2 + 2.0 * np.sum(
        (1.0 - transmission) * (radius * dz)[:, None],
        axis=0,
    )
    return effective_radius_squared / float(star_radius_m) ** 2


def _comparison_metrics(spectra, reference):
    output = {}
    for case in CASES:
        absolute = (spectra[case] - reference[case]) * 1.0e6
        effect = (
            (spectra[case] - spectra["clear"])
            - (reference[case] - reference["clear"])
        ) * 1.0e6
        output[case] = {
            "absolute_rms_difference_ppm": _rms(absolute),
            "absolute_median_difference_ppm": float(np.median(absolute)),
            "absolute_max_abs_difference_ppm": float(np.max(np.abs(absolute))),
            "cloud_effect_rms_difference_ppm": _rms(effect),
            "cloud_effect_max_abs_difference_ppm": float(np.max(np.abs(effect))),
        }
    return output


def _geometry_metrics(geometry, baseline):
    edge_difference = geometry.edge_radius_m - baseline.edge_radius_m
    return {
        "reference_pressure_bar": geometry.reference_pressure_pa / 1.0e5,
        "reference_radius_m": geometry.reference_radius_m,
        "bottom_radius_difference_m": geometry.bottom_radius_m - baseline.bottom_radius_m,
        "top_radius_difference_m": geometry.top_radius_m - baseline.top_radius_m,
        "edge_radius_rms_difference_m": _rms(edge_difference),
        "edge_radius_max_abs_difference_m": float(np.max(np.abs(edge_difference))),
        "metadata": dict(geometry.metadata),
    }


def _save_spectra(path, wavelength, picaso, variants, algorithms, impact):
    arrays = {"wavelength_micron": wavelength}
    for case, values in picaso.items():
        arrays[f"picaso_{case}"] = values
    for name, spectra in variants.items():
        for case, values in spectra.items():
            arrays[f"geometry_{name}_{case}"] = values
    for name, spectra in algorithms.items():
        for case, values in spectra.items():
            arrays[f"algorithm_{name}_{case}"] = values
    for order, spectra in impact.items():
        for case, values in spectra.items():
            arrays[f"impact_{order}_{case}"] = values
    np.savez_compressed(path, **arrays)


def _plot_algorithms(path, wavelength, picaso, algorithms, metrics):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for name, spectra in algorithms.items():
        axes[0].plot(
            wavelength,
            (spectra["clear"] - picaso["clear"]) * 1.0e6,
            lw=1.0,
            label=name,
        )
        axes[1].plot(
            wavelength,
            (
                (spectra["deck"] - spectra["clear"])
                - (picaso["deck"] - picaso["clear"])
            )
            * 1.0e6,
            lw=1.0,
            label=name,
        )
    names = list(algorithms)
    x = np.arange(len(names))
    width = 0.19
    for case_index, case in enumerate(CASES):
        axes[2].bar(
            x + (case_index - 1.5) * width,
            [metrics[name][case]["absolute_rms_difference_ppm"] for name in names],
            width,
            label=case,
        )
    axes[0].set_title("Clear residual")
    axes[0].set_ylabel("ROBERT hybrid - PICASO (ppm)")
    axes[1].set_title("Deck cloud-effect residual")
    axes[1].set_ylabel("Effect difference (ppm)")
    axes[2].set_title("Absolute RMS residual")
    axes[2].set_ylabel("RMS (ppm)")
    axes[2].set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=6)
    for axis in axes[:2]:
        axis.set_xscale("log")
        axis.set_xlabel("Wavelength (micron)")
        axis.axhline(0.0, color="0.6", lw=0.6)
    axes[0].legend(frameon=False, fontsize=6)
    axes[2].legend(frameon=False, fontsize=6)
    figure.suptitle("Transmission algorithm decomposition at fixed optical depth")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot(path, wavelength, picaso, variants, impact_metrics, geometries, baseline):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    selected = (
        "picaso_level_inverse_square",
        "layer_center_inverse_square",
        "layer_center_constant_gravity",
        "misanchored_same_radius_at_1bar",
    )
    for name in selected:
        axes[0, 0].plot(
            wavelength,
            (variants[name]["clear"] - picaso["clear"]) * 1.0e6,
            lw=1.0,
            label=name,
        )
        axes[0, 1].plot(
            wavelength,
            (
                (variants[name]["deck"] - variants[name]["clear"])
                - (picaso["deck"] - picaso["clear"])
            )
            * 1.0e6,
            lw=1.0,
            label=name,
        )
    orders = np.asarray(IMPACT_ORDERS)
    for case in CASES:
        axes[1, 0].plot(
            orders,
            [
                impact_metrics[str(order)][case]["absolute_to_order24_rms_ppm"]
                for order in orders
            ],
            marker="o",
            label=case,
        )
    pressure = baseline.pressure_grid.edges
    for name in selected[1:]:
        axes[1, 1].plot(
            pressure,
            geometries[name].edge_radius_m - baseline.edge_radius_m,
            label=name,
        )
    axes[0, 0].set_title("Clear spectrum residual")
    axes[0, 0].set_ylabel("ROBERT - PICASO (ppm)")
    axes[0, 1].set_title("Deck cloud-effect residual")
    axes[0, 1].set_ylabel("Effect difference (ppm)")
    axes[1, 0].set_title("Impact quadrature convergence")
    axes[1, 0].set_xlabel("Gauss-Legendre order per shell annulus")
    axes[1, 0].set_ylabel("RMS to order 24 (ppm)")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_title("Pressure-radius convention")
    axes[1, 1].set_xlabel("Pressure (bar)")
    axes[1, 1].set_ylabel("Edge radius difference (m)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].invert_xaxis()
    for axis in axes.flat:
        axis.axhline(0.0, color="0.6", lw=0.6)
        axis.legend(frameon=False, fontsize=7)
    axes[0, 0].set_xscale("log")
    axes[0, 1].set_xscale("log")
    axes[0, 0].set_xlabel("Wavelength (micron)")
    axes[0, 1].set_xlabel("Wavelength (micron)")
    figure.suptitle("Matched-optical-depth transmission geometry audit")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
