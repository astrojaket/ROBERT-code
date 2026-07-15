"""Run PICASO's transmission kernel on an analytic continuous-atmosphere contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np


BOLTZMANN_CONSTANT_ERG_K = 1.380649e-16
ATOMIC_MASS_G = 1.66053906660e-24


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.contract, args.output)


def evaluate(contract_path: Path, output_path: Path) -> None:
    from picaso.fluxes import get_transit_1d

    with np.load(contract_path, allow_pickle=False) as archive:
        contract = {name: np.array(archive[name], copy=True) for name in archive.files}

    wavelength = np.asarray(contract["wavelength_micron"], dtype=float)
    cross_section_cm2 = np.asarray(contract["cross_section_m2"], dtype=float) * 1.0e4
    layer_counts = np.asarray(contract["layer_counts"], dtype=int)
    pressure_bottom_pa = float(contract["pressure_bottom_bar"]) * 1.0e5
    pressure_top_pa = float(contract["pressure_top_bar"]) * 1.0e5
    radius_bottom_m = float(contract["planet_radius_m"])
    gravity_bottom_m_s2 = float(contract["gravity_m_s2"])
    temperature_k = float(contract["temperature_k"])
    mean_molecular_weight = float(contract["mean_molecular_weight_amu"])
    star_radius_cm = float(contract["star_radius_m"]) * 100.0
    coefficient_m = _hydrostatic_coefficient_m(contract)
    particle_mass_g = mean_molecular_weight * ATOMIC_MASS_G

    outputs: dict[str, np.ndarray] = {"wavelength_micron": wavelength}
    for layers in layer_counts:
        pressure_level_pa = np.geomspace(
            pressure_top_pa,
            pressure_bottom_pa,
            int(layers) + 1,
        )
        radius_level_m = _radius_at_pressure(
            pressure_level_pa,
            pressure_bottom_pa=pressure_bottom_pa,
            radius_bottom_m=radius_bottom_m,
            coefficient_m=coefficient_m,
        )
        pressure_layer_pa = np.sqrt(
            pressure_level_pa[:-1] * pressure_level_pa[1:]
        )
        radius_layer_m = _radius_at_pressure(
            pressure_layer_pa,
            pressure_bottom_pa=pressure_bottom_pa,
            radius_bottom_m=radius_bottom_m,
            coefficient_m=coefficient_m,
        )
        gravity_layer_m_s2 = gravity_bottom_m_s2 * (
            radius_bottom_m / radius_layer_m
        ) ** 2
        column_mass_g_cm2 = (
            np.diff(pressure_level_pa) / gravity_layer_m_s2
        ) * 0.1
        vertical_tau = (
            column_mass_g_cm2[:, None]
            / particle_mass_g
            * cross_section_cm2[None, :]
        )
        radius_level_cm = radius_level_m * 100.0
        layer_thickness_cm = np.zeros(radius_level_cm.size, dtype=float)
        layer_thickness_cm[1:] = radius_level_cm[:-1] - radius_level_cm[1:]
        transit_depth = get_transit_1d(
            radius_level_cm,
            layer_thickness_cm,
            radius_level_cm.size,
            wavelength.size,
            star_radius_cm,
            np.full(int(layers), mean_molecular_weight),
            BOLTZMANN_CONSTANT_ERG_K,
            ATOMIC_MASS_G,
            pressure_level_pa * 10.0,
            np.full(radius_level_cm.size, temperature_k),
            column_mass_g_cm2,
            vertical_tau,
        )
        outputs[f"transit_depth_{int(layers)}"] = np.asarray(
            transit_depth,
            dtype=float,
        )

    metadata = {
        "picaso_version": importlib.metadata.version("picaso"),
        "solver": "picaso.fluxes.get_transit_1d",
        "layer_counts": layer_counts.tolist(),
        "opacity_contract": "analytic_cross_section_times_P_over_kT",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **outputs,
    )


def _hydrostatic_coefficient_m(contract) -> float:
    boltzmann = 1.380649e-23
    atomic_mass = 1.66053906660e-27
    return (
        float(contract["mean_molecular_weight_amu"])
        * atomic_mass
        * float(contract["gravity_m_s2"])
        * float(contract["planet_radius_m"]) ** 2
        / (boltzmann * float(contract["temperature_k"]))
    )


def _radius_at_pressure(
    pressure_pa,
    *,
    pressure_bottom_pa,
    radius_bottom_m,
    coefficient_m,
):
    return 1.0 / (
        1.0 / radius_bottom_m
        + np.log(np.asarray(pressure_pa, dtype=float) / pressure_bottom_pa)
        / coefficient_m
    )


if __name__ == "__main__":
    main()
