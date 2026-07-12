"""Download the minimal petitRADTRANS 4 data used by ROBERT benchmarks."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from petitRADTRANS.radtrans import Radtrans  # noqa: E402

INPUT_DATA = (
    Path(__file__).resolve().parents[1]
    / "opacity_data"
    / "petitRADTRANS"
    / "input_data"
)


def main() -> None:
    INPUT_DATA.mkdir(parents=True, exist_ok=True)
    atmosphere = Radtrans(
        pressures=np.geomspace(1.0e-6, 100.0, 80),
        wavelength_boundaries=np.array([2.8, 5.2]),
        line_species=["H2O__POKAZATEL"],
        gas_continuum_contributors=[
            "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
            "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
        ],
        scattering_in_emission=False,
        path_input_data=str(INPUT_DATA),
    )
    print(f"petitRADTRANS input data: {INPUT_DATA}")
    print(f"loaded pressures: {atmosphere.pressures.size}")
    print(f"loaded frequencies: {atmosphere.frequencies.size}")


if __name__ == "__main__":
    main()
