"""Generate a synthetic transmission spectrum with spot+facula contamination.

The default blackbody spectra make the example runnable without external data
and are a controlled approximation only. Use ``--spectrum-model phoenix`` for
the science path after setting ``PYSYN_CDBS`` to an STScI Synphot reference
root containing ``grid/phoenix``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from robert_exoplanets import (
    SpectralGrid,
    Spectrum,
    Star,
    StellarHeterogeneityDefinition,
    prepare_stellar_contamination_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectrum-model",
        choices=("blackbody", "phoenix"),
        default="blackbody",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/outputs/stellar_contamination_transmission"),
    )
    args = parser.parse_args()

    wavelength = np.geomspace(0.6, 12.0, 400)
    grid = SpectralGrid.from_array(wavelength, unit="micron", role="rt_native")
    star = Star(
        name="synthetic G star",
        radius_m=6.96e8,
        effective_temperature_k=5200.0,
        log_g_cgs=4.5,
        metallicity_dex=0.0,
    )
    contamination = prepare_stellar_contamination_model(
        star,
        grid,
        spectrum_model=args.spectrum_model,
        heterogeneities=(
            StellarHeterogeneityDefinition(
                name="cool_spot",
                kind="spot",
                temperature_k=4200.0,
                covering_fraction_parameter="f_spot",
            ),
            StellarHeterogeneityDefinition(
                name="hot_facula",
                kind="facula",
                temperature_k=6000.0,
                covering_fraction_parameter="f_fac",
            ),
        ),
    )
    uncontaminated_values = (
        0.008
        + 2.5e-4 * np.exp(-0.5 * ((wavelength - 1.4) / 0.12) ** 2)
        + 1.7e-4 * np.exp(-0.5 * ((wavelength - 4.3) / 0.25) ** 2)
    )
    uncontaminated = Spectrum(
        spectral_grid=grid,
        values=uncontaminated_values,
        unit="transit_depth",
        observable="transit_depth",
        metadata={"example": "synthetic_nonconstant_planet_spectrum"},
    )
    parameters = {"f_spot": 0.18, "f_fac": 0.12}
    result = contamination.evaluate(parameters)
    observed = contamination.apply(uncontaminated, parameters)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "stellar_contamination_transmission.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "wavelength_micron",
                "uncontaminated_transit_depth",
                "contamination_factor",
                "observed_transit_depth",
            )
        )
        writer.writerows(
            zip(
                wavelength,
                uncontaminated.values,
                result.contamination_factor.values,
                observed.values,
                strict=True,
            )
        )
    summary = {
        "spectrum_model": args.spectrum_model,
        "blackbody_scope": (
            "controlled approximation; not the stellar-atmosphere validation standard"
            if args.spectrum_model == "blackbody"
            else "not applicable"
        ),
        "parameters": parameters,
        "minimum_contamination_factor": float(
            np.min(result.contamination_factor.values)
        ),
        "maximum_contamination_factor": float(
            np.max(result.contamination_factor.values)
        ),
        "application_order": "native transit depth before instrument response",
        "csv": csv_path.name,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
