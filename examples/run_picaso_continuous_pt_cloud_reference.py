"""Run PICASO's transmission kernel for the continuous P/T/cloud benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np


BOLTZMANN_CONSTANT_ERG_K = 1.380649e-16
ATOMIC_MASS_G = 1.66053906660e-24
GRID_MODES = ("aligned", "misaligned")


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
    layer_counts = np.asarray(contract["layer_counts"], dtype=int)
    star_radius_cm = float(contract["star_radius_m"]) * 100.0
    mean_molecular_weight = float(contract["mean_molecular_weight_amu"])
    outputs: dict[str, np.ndarray] = {"wavelength_micron": wavelength}

    for mode in GRID_MODES:
        for layers in layer_counts:
            suffix = f"{mode}_{int(layers)}"
            pressure_level_pa = np.asarray(
                contract[f"pressure_level_pa_{suffix}"], dtype=float
            )
            radius_level_m = np.asarray(
                contract[f"radius_level_m_{suffix}"], dtype=float
            )
            temperature_level_k = np.asarray(
                contract[f"temperature_level_k_{suffix}"], dtype=float
            )
            gravity_layer_m_s2 = np.asarray(
                contract[f"gravity_layer_m_s2_{suffix}"], dtype=float
            )
            column_mass_g_cm2 = (
                np.diff(pressure_level_pa) / gravity_layer_m_s2
            ) * 0.1
            radius_level_cm = radius_level_m * 100.0
            layer_thickness_cm = np.zeros(radius_level_cm.size, dtype=float)
            layer_thickness_cm[1:] = radius_level_cm[:-1] - radius_level_cm[1:]
            for case in ("clear", "cloud"):
                vertical_tau = np.asarray(
                    contract[f"vertical_tau_{case}_{suffix}"], dtype=float
                )
                outputs[f"transit_depth_{case}_{suffix}"] = np.asarray(
                    get_transit_1d(
                        radius_level_cm,
                        layer_thickness_cm,
                        radius_level_cm.size,
                        wavelength.size,
                        star_radius_cm,
                        np.full(int(layers), mean_molecular_weight),
                        BOLTZMANN_CONSTANT_ERG_K,
                        ATOMIC_MASS_G,
                        pressure_level_pa * 10.0,
                        temperature_level_k,
                        column_mass_g_cm2,
                        vertical_tau,
                    ),
                    dtype=float,
                )

    metadata = {
        "picaso_version": importlib.metadata.version("picaso"),
        "solver": "picaso.fluxes.get_transit_1d",
        "layer_counts": layer_counts.tolist(),
        "grid_modes": list(GRID_MODES),
        "vertical_optical_depth_contract": "exact_continuous_layer_integral",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **outputs,
    )


if __name__ == "__main__":
    main()
