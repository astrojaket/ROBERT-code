"""Derive H2/MMW posteriors and compare the L 98-59 b run with Figure 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robert_exoplanets import weighted_quantile


MOLECULAR_MASS_AMU = {
    "SO2": 64.066,
    "H2S": 34.0809,
    "CO2": 44.0095,
    "H2": 2.01588,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    result_directory = args.result_directory.expanduser().resolve()
    output_directory = (
        args.output_directory.expanduser().resolve()
        if args.output_directory is not None
        else result_directory.parent / "plots" / result_directory.name
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    result = json.loads((result_directory / "result.json").read_text())
    arrays = np.load(result_directory / result["arrays"])
    samples = np.asarray(arrays["samples"], dtype=float)
    weights = np.asarray(arrays["weights"], dtype=float)
    weights /= np.sum(weights)
    columns = {name: samples[:, index] for index, name in enumerate(result["parameter_names"])}

    vmr = {
        "SO2": np.power(10.0, columns["log_SO2"]),
        "H2S": np.power(10.0, columns["log_H2S"]),
        "CO2": np.power(10.0, columns["log_CO2"]),
    }
    vmr["H2"] = 1.0 - sum(vmr.values())
    if np.any(vmr["H2"] <= 0.0):
        raise ValueError("posterior samples violate four-gas composition closure")
    mean_molecular_weight = sum(
        vmr[name] * MOLECULAR_MASS_AMU[name] for name in MOLECULAR_MASS_AMU
    )

    quantile_levels = (0.025, 0.16, 0.5, 0.84, 0.975)
    derived = {
        "log_H2": np.log10(vmr["H2"]),
        "mean_molecular_weight_amu": mean_molecular_weight,
    }
    quantities = {
        "log_SO2": columns["log_SO2"],
        "log_H2S": columns["log_H2S"],
        "log_CO2": columns["log_CO2"],
        "log_H2": derived["log_H2"],
        "mean_molecular_weight_amu": mean_molecular_weight,
        "temperature_K": columns["temperature"],
    }
    quantiles = {
        name: weighted_quantile(values, weights, quantile_levels).tolist()
        for name, values in quantities.items()
    }
    comparison = {
        "robert_run": {
            "configuration": "four-gas CLR; fixed mass and 1-bar lower boundary",
            "live_points": 50,
            "mpi_processes": 3,
            "log_evidence": result["log_evidence"],
            "log_evidence_error": result["log_evidence_error"],
            "quantile_levels": list(quantile_levels),
            "quantiles": quantiles,
            "h2_vmr_97_5_percent_upper": float(
                weighted_quantile(vmr["H2"], weights, (0.975,))[0]
            ),
            "co2_vmr_97_5_percent_upper": float(
                weighted_quantile(vmr["CO2"], weights, (0.975,))[0]
            ),
            "mmw_2_5_percent_lower_amu": quantiles[
                "mean_molecular_weight_amu"
            ][0],
        },
        "poseidon_figure_8": {
            "configuration": "paper 11-parameter CLR; 2,000 live points",
            "log_evidence": 1895.3,
            "chi_squared": 186.0,
            "degrees_of_freedom": 207,
            "h2_vmr_97_5_percent_upper": 0.24,
            "co2_vmr_97_5_percent_upper": 0.84,
            "mmw_2_5_percent_lower_amu": 20.1,
            "qualitative_result": "SO2-rich atmosphere favoured",
            "source": "Bello-Arufe et al. (2025), Figure 8 and Section 4",
        },
        "comparability_note": (
            "The data and principal CLR composition treatment match, but evidence "
            "values are not directly comparable because ROBERT uses the requested "
            "four-gas model, fixed planet mass, fixed 1-bar lower boundary, R1000 "
            "opacity, and 50 rather than 2,000 live points."
        ),
    }
    summary_path = output_directory / "l98_59b_figure8_comparison.json"
    summary_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    _plot_comparison(quantities, weights, quantiles, output_directory)
    print(json.dumps(comparison, indent=2, sort_keys=True))


def _plot_comparison(
    quantities: dict[str, np.ndarray],
    weights: np.ndarray,
    quantiles: dict[str, list[float]],
    output_directory: Path,
) -> None:
    panels = (
        ("log_SO2", r"$\log_{10}(X_{\rm SO_2})$"),
        ("log_H2S", r"$\log_{10}(X_{\rm H_2S})$"),
        ("log_CO2", r"$\log_{10}(X_{\rm CO_2})$"),
        ("log_H2", r"$\log_{10}(X_{\rm H_2})$"),
        ("mean_molecular_weight_amu", r"Mean molecular weight (amu)"),
        ("temperature_K", r"Isothermal temperature (K)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.5))
    for axis, (name, label) in zip(axes.flat, panels, strict=True):
        axis.hist(
            quantities[name],
            bins=35,
            weights=weights,
            density=True,
            color="#365c8d",
            alpha=0.8,
        )
        low, _, median, _, high = quantiles[name]
        axis.axvline(median, color="black", linewidth=1.2)
        axis.axvspan(low, high, color="#f4a261", alpha=0.24, label="ROBERT 95%")
        axis.set_xlabel(label)
        axis.set_yticks([])
    axes[0, 2].axvline(
        np.log10(0.84),
        color="#b22222",
        linestyle="--",
        label="POSEIDON 95% upper",
    )
    axes[1, 0].axvline(
        np.log10(0.24),
        color="#b22222",
        linestyle="--",
        label="POSEIDON 95% upper",
    )
    axes[1, 1].axvline(
        20.1,
        color="#b22222",
        linestyle="--",
        label="POSEIDON 95% lower",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("L 98-59 b: ROBERT four-gas CLR posterior")
    figure.tight_layout()
    figure.savefig(output_directory / "l98_59b_figure8_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
