"""Continuous non-isothermal, P/T-opacity, and sharp-cloud transmission benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
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
    HydrostaticPathGeometry,
    PreparedCorrelatedKOpacity,
    PressureGrid,
    SpectralGrid,
    assemble_gas_optical_depth,
    solve_absorption_transmission,
)

try:
    from examples.benchmark_shared_deck_haze_external_parity import ROOT, _rms
except ModuleNotFoundError:
    from benchmark_shared_deck_haze_external_parity import ROOT, _rms


BOLTZMANN_CONSTANT_J_K = 1.380649e-23
ATOMIC_MASS_KG = 1.66053906660e-27
GRID_MODES = ("aligned", "misaligned")
CASES = ("clear", "cloud")
DEFAULT_LAYERS = (20, 40, 80, 160, 320)
RUNNER = Path(__file__).with_name("run_picaso_continuous_pt_cloud_reference.py")
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "continuous_pt_cloud_reference"


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--picaso-python",
        type=Path,
        default=Path("/Users/jaketaylor/opt/anaconda3/envs/picaso/bin/python"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--layers",
        default=",".join(str(value) for value in DEFAULT_LAYERS),
    )
    parser.add_argument("--reference-order", type=int, default=1024)
    args = parser.parse_args()
    report = run(
        args.picaso_python,
        args.output_dir,
        layer_counts=tuple(
            dict.fromkeys(int(value) for value in args.layers.split(",") if value)
        ),
        reference_order=args.reference_order,
    )
    print(json.dumps(report, indent=2))
    return report


def run(
    picaso_python: Path,
    output_dir: Path,
    *,
    layer_counts: tuple[int, ...],
    reference_order: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _base_contract(layer_counts)
    started = perf_counter()
    reference = {
        case: _continuous_transit_depth(
            contract,
            reference_order,
            cloudy=case == "cloud",
        )
        for case in CASES
    }
    reference_seconds = perf_counter() - started
    check_order = max(64, reference_order // 2)
    reference_check = {
        case: _continuous_transit_depth(
            contract,
            check_order,
            cloudy=case == "cloud",
        )
        for case in CASES
    }

    gridded = _prepare_gridded_contract(contract, layer_counts)
    external_contract = {**contract, **gridded}
    contract_path = output_dir / "continuous_pt_cloud_contract.npz"
    picaso_path = output_dir / "picaso_continuous_pt_cloud.npz"
    np.savez_compressed(contract_path, **external_contract)
    _run_picaso(picaso_python, contract_path, picaso_path)
    with np.load(picaso_path, allow_pickle=False) as archive:
        picaso = {name: np.array(archive[name], copy=True) for name in archive.files}

    spectra: dict[str, np.ndarray] = {
        "reference_clear": reference["clear"],
        "reference_cloud": reference["cloud"],
    }
    metrics = {
        code: {mode: {} for mode in GRID_MODES}
        for code in ("robert", "picaso")
    }
    for mode in GRID_MODES:
        for layers in layer_counts:
            suffix = f"{mode}_{layers}"
            pressure_level_pa = gridded[f"pressure_level_pa_{suffix}"]
            radius_level_m = gridded[f"radius_level_m_{suffix}"]
            robert_case = {}
            for case in CASES:
                vertical_tau = gridded[f"vertical_tau_{case}_{suffix}"]
                robert_case[case] = _evaluate_robert(
                    contract,
                    pressure_level_pa,
                    radius_level_m,
                    vertical_tau,
                )
                spectra[f"robert_{case}_{suffix}"] = robert_case[case]
                spectra[f"picaso_{case}_{suffix}"] = np.asarray(
                    picaso[f"transit_depth_{case}_{suffix}"], dtype=float
                )
            for code, values in (
                ("robert", robert_case),
                (
                    "picaso",
                    {
                        case: spectra[f"picaso_{case}_{suffix}"]
                        for case in CASES
                    },
                ),
            ):
                metrics[code][mode][str(layers)] = _metrics(values, reference)

    report = {
        "schema_version": 1,
        "benchmark": "continuous_nonisothermal_pt_opacity_cloud_boundary",
        "interpretation": (
            "The continuous atmosphere, P/T-dependent gas opacity, and physical "
            "cloud top are fixed. Aligned and misaligned grids differ only in "
            "whether that cloud top is a pressure edge."
        ),
        "contract": {
            key: _json_scalar(value)
            for key, value in contract.items()
            if np.asarray(value).ndim == 0
        },
        "layer_counts": list(layer_counts),
        "continuous_reference": {
            "quadrature_order": reference_order,
            "check_order": check_order,
            "clear_check_rms_ppm": _rms(
                (reference_check["clear"] - reference["clear"]) * 1.0e6
            ),
            "cloud_check_rms_ppm": _rms(
                (reference_check["cloud"] - reference["cloud"]) * 1.0e6
            ),
            "cloud_effect_check_rms_ppm": _rms(
                (
                    (reference_check["cloud"] - reference_check["clear"])
                    - (reference["cloud"] - reference["clear"])
                )
                * 1.0e6
            ),
            "seconds": reference_seconds,
        },
        "metrics": metrics,
        "cloud_effect_convergence_order_fit": {
            code: {
                mode: _convergence_order(layer_counts, metrics[code][mode])
                for mode in GRID_MODES
            }
            for code in ("robert", "picaso")
        },
        "picaso_metadata": json.loads(str(picaso["metadata_json"])),
    }
    (output_dir / "continuous_pt_cloud_reference.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "continuous_pt_cloud_reference_spectra.npz",
        wavelength_micron=contract["wavelength_micron"],
        **spectra,
    )
    _plot(
        output_dir / "continuous_pt_cloud_reference.png",
        contract,
        reference,
        spectra,
        metrics,
        layer_counts,
    )
    return report


def _base_contract(layer_counts):
    wavelength = np.geomspace(1.0, 12.0, 120)
    return {
        "wavelength_micron": wavelength,
        "layer_counts": np.asarray(layer_counts, dtype=int),
        "pressure_bottom_bar": np.array(10.0),
        "pressure_top_bar": np.array(1.0e-9),
        "planet_radius_m": np.array(75_567_044.0),
        "star_radius_m": np.array(695_700_000.0),
        "gravity_m_s2": np.array(8.42),
        "mean_molecular_weight_amu": np.array(2.3),
        "temperature_bottom_k": np.array(1600.0),
        "temperature_top_k": np.array(800.0),
        "temperature_dip_k": np.array(250.0),
        "cloud_top_pressure_bar": np.array(2.0e-3),
        "cloud_cross_section_m2": np.array(2.0e-28),
        "opacity_reference_pressure_bar": np.array(0.1),
        "opacity_reference_temperature_k": np.array(1100.0),
        "band_center_micron": np.array([1.4, 1.9, 2.7, 4.3, 6.3, 9.5]),
        "band_log_width": np.array([0.08, 0.09, 0.12, 0.11, 0.14, 0.17]),
        "band_amplitude_m2": np.array(
            [3.0e-28, 9.0e-28, 2.0e-27, 1.2e-27, 3.0e-27, 8.0e-28]
        ),
        "band_pressure_exponent": np.array([0.03, -0.04, 0.08, 0.12, -0.02, 0.06]),
        "band_temperature_exponent": np.array([-1.0, 1.5, 0.8, -0.5, 2.0, 1.0]),
    }


def _temperature_from_u(u, contract):
    u = np.asarray(u, dtype=float)
    bottom = float(contract["temperature_bottom_k"])
    top = float(contract["temperature_top_k"])
    dip = float(contract["temperature_dip_k"])
    return bottom + (top - bottom) * u - dip * np.sin(np.pi * u)


def _temperature_integral_u(u, contract):
    u = np.asarray(u, dtype=float)
    bottom = float(contract["temperature_bottom_k"])
    delta = float(contract["temperature_top_k"]) - bottom
    dip = float(contract["temperature_dip_k"])
    return (
        bottom * u
        + 0.5 * delta * u**2
        + dip / np.pi * (np.cos(np.pi * u) - 1.0)
    )


def _pressure_log_span(contract):
    return np.log(
        float(contract["pressure_bottom_bar"])
        / float(contract["pressure_top_bar"])
    )


def _inverse_radius_from_u(u, contract):
    radius_bottom = float(contract["planet_radius_m"])
    gravitational_parameter = (
        float(contract["gravity_m_s2"]) * radius_bottom**2
    )
    coefficient = BOLTZMANN_CONSTANT_J_K / (
        float(contract["mean_molecular_weight_amu"])
        * ATOMIC_MASS_KG
        * gravitational_parameter
    )
    return (
        1.0 / radius_bottom
        - _pressure_log_span(contract)
        * coefficient
        * _temperature_integral_u(u, contract)
    )


def _radius_from_pressure(pressure_pa, contract):
    pressure_bottom_pa = float(contract["pressure_bottom_bar"]) * 1.0e5
    u = np.log(pressure_bottom_pa / np.asarray(pressure_pa, dtype=float)) / (
        _pressure_log_span(contract)
    )
    return 1.0 / _inverse_radius_from_u(u, contract)


def _state_at_radius(radius_m, contract):
    radius = np.asarray(radius_m, dtype=float)
    radius_bottom = float(contract["planet_radius_m"])
    radius_top = 1.0 / _inverse_radius_from_u(1.0, contract)
    u = np.clip((radius - radius_bottom) / (radius_top - radius_bottom), 0.0, 1.0)
    gravitational_parameter = (
        float(contract["gravity_m_s2"]) * radius_bottom**2
    )
    derivative_coefficient = (
        -_pressure_log_span(contract)
        * BOLTZMANN_CONSTANT_J_K
        / (
            float(contract["mean_molecular_weight_amu"])
            * ATOMIC_MASS_KG
            * gravitational_parameter
        )
    )
    target = 1.0 / radius
    for _ in range(7):
        residual = _inverse_radius_from_u(u, contract) - target
        derivative = derivative_coefficient * _temperature_from_u(u, contract)
        u = np.clip(u - residual / derivative, 0.0, 1.0)
    pressure_pa = (
        float(contract["pressure_bottom_bar"])
        * 1.0e5
        * np.exp(-_pressure_log_span(contract) * u)
    )
    return pressure_pa, _temperature_from_u(u, contract)


def _gas_cross_section(pressure_pa, temperature_k, contract):
    pressure = np.asarray(pressure_pa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    wavelength = np.asarray(contract["wavelength_micron"], dtype=float)
    profiles = np.exp(
        -0.5
        * (
            (
                np.log(wavelength)[None, :]
                - np.log(np.asarray(contract["band_center_micron"]))[:, None]
            )
            / np.asarray(contract["band_log_width"])[:, None]
        )
        ** 2
    )
    pressure_ratio = pressure / (
        float(contract["opacity_reference_pressure_bar"]) * 1.0e5
    )
    temperature_ratio = temperature / float(
        contract["opacity_reference_temperature_k"]
    )
    factors = (
        pressure_ratio[..., None]
        ** np.asarray(contract["band_pressure_exponent"])
        * temperature_ratio[..., None]
        ** np.asarray(contract["band_temperature_exponent"])
        * np.asarray(contract["band_amplitude_m2"])
    )
    continuum = (
        1.5e-31
        * pressure_ratio**0.03
        * temperature_ratio**0.5
    )
    return continuum[..., None] + np.einsum("...b,bw->...w", factors, profiles)


def _cloud_cross_section(contract):
    wavelength = np.asarray(contract["wavelength_micron"], dtype=float)
    return float(contract["cloud_cross_section_m2"]) * wavelength**-1.0


def _continuous_transit_depth(contract, order, *, cloudy):
    radius_bottom = float(contract["planet_radius_m"])
    radius_top = _radius_from_pressure(
        float(contract["pressure_top_bar"]) * 1.0e5,
        contract,
    )
    nodes, weights = np.polynomial.legendre.leggauss(order)
    cloud_top_pa = float(contract["cloud_top_pressure_bar"]) * 1.0e5
    cloud_sigma = _cloud_cross_section(contract)
    cloud_radius = _radius_from_pressure(cloud_top_pa, contract)
    intervals = (
        ((radius_bottom, cloud_radius), (cloud_radius, radius_top))
        if cloudy
        else ((radius_bottom, radius_top),)
    )
    blocked_area = np.full(
        np.asarray(contract["wavelength_micron"]).size,
        radius_bottom**2,
        dtype=float,
    )
    for lower_impact, upper_impact in intervals:
        half_width = 0.5 * (upper_impact - lower_impact)
        impact = 0.5 * (upper_impact + lower_impact) + half_width * nodes
        half_chord = np.sqrt(radius_top**2 - impact**2)
        slant_tau = np.empty(
            (order, np.asarray(contract["wavelength_micron"]).size)
        )
        chunk = 16
        for start in range(0, order, chunk):
            stop = min(start + chunk, order)
            local_half_chord = half_chord[start:stop]
            coordinate = 0.5 * local_half_chord[:, None] * (
                nodes[None, :] + 1.0
            )
            radius = np.sqrt(impact[start:stop, None] ** 2 + coordinate**2)
            pressure, temperature = _state_at_radius(radius, contract)
            number_density = pressure / (BOLTZMANN_CONSTANT_J_K * temperature)
            cross_section = _gas_cross_section(pressure, temperature, contract)
            slant_tau[start:stop] = local_half_chord[:, None] * np.einsum(
                "j,ij,ijw->iw",
                weights,
                number_density,
                cross_section,
                optimize=True,
            )
            if cloudy:
                cloud_half_chord = np.sqrt(
                    np.maximum(cloud_radius**2 - impact[start:stop] ** 2, 0.0)
                )
                cloud_coordinate = 0.5 * cloud_half_chord[:, None] * (
                    nodes[None, :] + 1.0
                )
                cloud_path_radius = np.sqrt(
                    impact[start:stop, None] ** 2 + cloud_coordinate**2
                )
                cloud_pressure, cloud_temperature = _state_at_radius(
                    cloud_path_radius, contract
                )
                cloud_number_density = cloud_pressure / (
                    BOLTZMANN_CONSTANT_J_K * cloud_temperature
                )
                cloud_slant_column = cloud_half_chord * np.sum(
                    weights[None, :] * cloud_number_density,
                    axis=1,
                )
                slant_tau[start:stop] += (
                    cloud_slant_column[:, None] * cloud_sigma
                )
        blocked_area += half_width * np.sum(
            weights[:, None]
            * 2.0
            * impact[:, None]
            * (1.0 - np.exp(-slant_tau)),
            axis=0,
        )
    return blocked_area / float(contract["star_radius_m"]) ** 2


def _pressure_edges(contract, layers, mode):
    top = float(contract["pressure_top_bar"])
    bottom = float(contract["pressure_bottom_bar"])
    cloud = float(contract["cloud_top_pressure_bar"])
    if mode == "misaligned":
        return np.geomspace(top, bottom, layers + 1)
    fraction = np.log(cloud / top) / np.log(bottom / top)
    layers_above = int(np.clip(round(layers * fraction), 1, layers - 1))
    return np.concatenate(
        (
            np.geomspace(top, cloud, layers_above + 1),
            np.geomspace(cloud, bottom, layers - layers_above + 1)[1:],
        )
    )


def _prepare_gridded_contract(contract, layer_counts):
    output = {}
    for mode in GRID_MODES:
        for layers in layer_counts:
            suffix = f"{mode}_{layers}"
            pressure_level_pa = _pressure_edges(contract, layers, mode) * 1.0e5
            pressure_layer_pa = np.sqrt(
                pressure_level_pa[:-1] * pressure_level_pa[1:]
            )
            radius_level_m = _radius_from_pressure(pressure_level_pa, contract)
            radius_layer_m = _radius_from_pressure(pressure_layer_pa, contract)
            output[f"pressure_level_pa_{suffix}"] = pressure_level_pa
            output[f"radius_level_m_{suffix}"] = radius_level_m
            output[f"temperature_level_k_{suffix}"] = _state_at_radius(
                radius_level_m, contract
            )[1]
            output[f"gravity_layer_m_s2_{suffix}"] = float(
                contract["gravity_m_s2"]
            ) * (float(contract["planet_radius_m"]) / radius_layer_m) ** 2
            output[f"vertical_tau_clear_{suffix}"] = _layer_vertical_tau(
                radius_level_m, contract, cloudy=False
            )
            output[f"vertical_tau_cloud_{suffix}"] = _layer_vertical_tau(
                radius_level_m, contract, cloudy=True
            )
    return output


def _layer_vertical_tau(radius_level_m, contract, *, cloudy, order=32):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    outer = np.asarray(radius_level_m[:-1], dtype=float)
    inner = np.asarray(radius_level_m[1:], dtype=float)
    half_width = 0.5 * (outer - inner)
    radius = 0.5 * (outer + inner)[:, None] + half_width[:, None] * nodes
    pressure, temperature = _state_at_radius(radius, contract)
    number_density = pressure / (BOLTZMANN_CONSTANT_J_K * temperature)
    cross_section = _gas_cross_section(pressure, temperature, contract)
    vertical_tau = half_width[:, None] * np.einsum(
        "j,ij,ijw->iw",
        weights,
        number_density,
        cross_section,
        optimize=True,
    )
    if cloudy:
        cloud_radius = _radius_from_pressure(
            float(contract["cloud_top_pressure_bar"]) * 1.0e5,
            contract,
        )
        cloud_outer = np.minimum(outer, cloud_radius)
        cloud_width = np.maximum(cloud_outer - inner, 0.0)
        cloud_half_width = 0.5 * cloud_width
        cloud_radius_nodes = (
            0.5 * (cloud_outer + inner)[:, None]
            + cloud_half_width[:, None] * nodes
        )
        cloud_pressure, cloud_temperature = _state_at_radius(
            cloud_radius_nodes, contract
        )
        cloud_number_density = cloud_pressure / (
            BOLTZMANN_CONSTANT_J_K * cloud_temperature
        )
        cloud_vertical_column = cloud_half_width * np.sum(
            weights[None, :] * cloud_number_density,
            axis=1,
        )
        vertical_tau += cloud_vertical_column[:, None] * _cloud_cross_section(
            contract
        )
    return vertical_tau


def _evaluate_robert(
    contract,
    pressure_level_pa,
    radius_level_m,
    vertical_tau,
):
    pressure_edges_bar = np.asarray(pressure_level_pa) / 1.0e5
    pressure_centers_bar = np.sqrt(
        pressure_edges_bar[:-1] * pressure_edges_bar[1:]
    )
    layers = pressure_centers_bar.size
    pressure_grid = PressureGrid(
        edges=pressure_edges_bar,
        centers=pressure_centers_bar,
        unit="bar",
        name="continuous non-isothermal P/T cloud reference",
    )
    center_radius = _radius_from_pressure(pressure_centers_bar * 1.0e5, contract)
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=_state_at_radius(center_radius, contract)[1],
        composition={"absorber": np.ones(layers)},
        mean_molecular_weight=float(contract["mean_molecular_weight_amu"]),
    )
    wavelength = np.asarray(contract["wavelength_micron"], dtype=float)
    spectral_grid = SpectralGrid.from_array(
        wavelength,
        unit="micron",
        role="opacity",
        name="analytic P/T-dependent opacity",
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="analytic P/T-dependent opacity",
        spectral_grid=spectral_grid,
        pressure_grid=pressure_grid,
        species=("absorber",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key=f"continuous-pt-cloud-{layers}-{pressure_edges_bar[1]:.8e}",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, layers, wavelength.size, 1)),
        unit="m^2/molecule",
    )
    gas = assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=float(contract["gravity_m_s2"]),
    )
    gas = replace(
        gas,
        species_tau=None,
        total_tau=np.asarray(vertical_tau)[:, :, None],
        metadata={
            **dict(gas.metadata),
            "source": "exact continuous layer-integrated vertical optical depth",
        },
    )
    scale_height = (radius_level_m[:-1] - radius_level_m[1:]) / np.log(
        pressure_level_pa[1:] / pressure_level_pa[:-1]
    )
    gravity = float(contract["gravity_m_s2"]) * (
        float(contract["planet_radius_m"]) / center_radius
    ) ** 2
    geometry = HydrostaticPathGeometry(
        pressure_grid=pressure_grid,
        reference_radius_m=float(contract["planet_radius_m"]),
        reference_pressure_pa=float(contract["pressure_bottom_bar"]) * 1.0e5,
        gravity_m_s2=gravity,
        scale_height_m=scale_height,
        edge_radius_m=radius_level_m,
        center_radius_m=center_radius,
        metadata={
            "path_model": "analytic_nonisothermal_inverse_square",
            "gravity_model": "exact_inverse_square",
        },
    )
    return np.asarray(
        solve_absorption_transmission(
            gas,
            geometry,
            star_radius_m=float(contract["star_radius_m"]),
            impact_quadrature_order=8,
        ).transit_depth.values,
        dtype=float,
    )


def _metrics(values, reference):
    clear_residual = (values["clear"] - reference["clear"]) * 1.0e6
    cloud_residual = (values["cloud"] - reference["cloud"]) * 1.0e6
    effect_residual = (
        (values["cloud"] - values["clear"])
        - (reference["cloud"] - reference["clear"])
    ) * 1.0e6
    return {
        "clear_rms_ppm": _rms(clear_residual),
        "cloud_absolute_rms_ppm": _rms(cloud_residual),
        "cloud_effect_rms_ppm": _rms(effect_residual),
        "cloud_effect_median_ppm": float(np.median(effect_residual)),
        "cloud_effect_max_abs_ppm": float(np.max(np.abs(effect_residual))),
    }


def _convergence_order(layer_counts, metrics):
    layers = np.asarray(layer_counts, dtype=float)
    errors = np.asarray(
        [metrics[str(value)]["cloud_effect_rms_ppm"] for value in layer_counts]
    )
    slope = np.polyfit(np.log(layers), np.log(errors), 1)[0]
    return float(-slope)


def _run_picaso(python, contract_path, output_path):
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "picaso-mpl")
    environment["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "picaso-numba")
    subprocess.run(
        [str(python), str(RUNNER), str(contract_path), str(output_path)],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def _json_scalar(value):
    scalar = np.asarray(value).item()
    return int(scalar) if isinstance(scalar, np.integer) else float(scalar)


def _plot(path, contract, reference, spectra, metrics, layer_counts):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    pressure = np.geomspace(
        float(contract["pressure_top_bar"]),
        float(contract["pressure_bottom_bar"]),
        300,
    )
    radius = _radius_from_pressure(pressure * 1.0e5, contract)
    temperature = _state_at_radius(radius, contract)[1]
    axes[0, 0].plot(temperature, pressure, color="black")
    axes[0, 0].axhline(
        float(contract["cloud_top_pressure_bar"]), color="tab:gray", ls="--"
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Continuous thermal structure")
    axes[0, 0].set_xlabel("Temperature (K)")
    axes[0, 0].set_ylabel("Pressure (bar)")
    wavelength = np.asarray(contract["wavelength_micron"])
    axes[0, 1].plot(
        wavelength,
        (reference["cloud"] - reference["clear"]) * 1.0e6,
        color="black",
    )
    axes[0, 1].set_title("Continuous cloud effect")
    axes[0, 1].set_ylabel("Transit-depth effect (ppm)")
    comparison_layers = 80 if 80 in layer_counts else layer_counts[len(layer_counts) // 2]
    colors = {
        ("robert", "aligned"): "tab:blue",
        ("robert", "misaligned"): "tab:cyan",
        ("picaso", "aligned"): "tab:orange",
        ("picaso", "misaligned"): "tab:red",
    }
    for code in ("robert", "picaso"):
        for mode in GRID_MODES:
            suffix = f"{mode}_{comparison_layers}"
            model_effect = (
                spectra[f"{code}_cloud_{suffix}"]
                - spectra[f"{code}_clear_{suffix}"]
            )
            reference_effect = reference["cloud"] - reference["clear"]
            label = f"{code.upper()} {mode}"
            axes[1, 0].plot(
                wavelength,
                (model_effect - reference_effect) * 1.0e6,
                color=colors[(code, mode)],
                lw=1.0,
                label=label,
            )
            axes[1, 1].plot(
                layer_counts,
                [
                    metrics[code][mode][str(value)]["cloud_effect_rms_ppm"]
                    for value in layer_counts
                ],
                color=colors[(code, mode)],
                marker="o",
                label=label,
            )
    axes[1, 0].axhline(0.0, color="0.6", lw=0.6)
    axes[1, 0].set_title(f"Cloud-effect residual at {comparison_layers} layers")
    axes[1, 0].set_ylabel("Model - continuous effect (ppm)")
    axes[1, 1].set_title("Cloud-boundary convergence")
    axes[1, 1].set_xlabel("Pressure layers")
    axes[1, 1].set_ylabel("Cloud-effect RMS (ppm)")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_xscale("log")
        axis.set_xlabel("Wavelength (micron)")
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].legend(frameon=False, fontsize=7)
    figure.suptitle("Continuous non-isothermal P/T-opacity cloud benchmark")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
