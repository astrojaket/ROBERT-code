"""Create every pRT oracle used by the CO/H2O K-band science grid.

Run this file in the ``petitradtrans-stable`` Conda environment. It starts one
oracle process at a time. Each process uses no more than three numerical
threads and has a strict process-memory ceiling below two GiB. The full H2O
and CO tables stay in ignored external storage.
"""

from __future__ import annotations

import argparse
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    os.environ[_thread_variable] = "3"
os.environ.setdefault("OMPI_MCA_btl", "self")


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = ROOT / "examples" / "run_petitradtrans3_lbl_kband_reference.py"
DEFAULT_INPUT = ROOT / "external_data" / "petitRADTRANS" / "input_data"
DEFAULT_OUTPUT = ROOT / "examples" / "outputs" / "lbl_kband"
H2O_TABLE = Path(
    "opacities/lines/line_by_line/H2O/1H2-16O/"
    "1H2-16O__POKAZATEL.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)

# These are the complete-state VMR values that preserve the former physical
# H2/He-background cases at the pRT boundary.  The reference runner converts
# the complete VMR state to pRT mass fractions only inside Radtrans.
BASELINE_H2O_VMR = 1.2898479517314042e-4
BASELINE_CO_VMR = 2.4875639069105653e-4
LOW_ABUNDANCE_H2O_VMR = 1.2856398911042743e-5
LOW_ABUNDANCE_CO_VMR = 2.479448361415386e-5
H2O_ONLY_H2O_VMR = 1.286295006797922e-4
H2O_ONLY_CO_VMR = 0.0
CO_ONLY_H2O_VMR = 0.0
CO_ONLY_CO_VMR = 2.4853900555007373e-4
BACKGROUND_ONLY_H2O_VMR = 0.0
BACKGROUND_ONLY_CO_VMR = 0.0

DEFAULT_MAX_MEMORY_GIB = 1.9
MAX_MEMORY_GIB = 2.0
CO_TABLE = Path(
    "opacities/lines/line_by_line/CO/12C-16O/"
    "12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-memory-gib",
        type=_memory_gib_argument,
        default=DEFAULT_MAX_MEMORY_GIB,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate oracle files that already exist.",
    )
    return parser.parse_args()


def _validate_max_memory_gib(value: float) -> float:
    """Validate the strict process-memory ceiling in GiB."""

    memory_gib = float(value)
    if not isfinite(memory_gib) or not 0.0 < memory_gib < MAX_MEMORY_GIB:
        raise ValueError("max-memory-gib must be finite, positive, and below 2 GiB")
    return memory_gib


def _memory_gib_argument(value: str) -> float:
    """Parse a process-memory ceiling below the two-GiB hard limit."""

    try:
        return _validate_max_memory_gib(float(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _vmr_arguments(h2o_vmr: float, co_vmr: float) -> tuple[str, ...]:
    """Return explicit H2O and CO VMR arguments for one oracle case."""

    return (
        "--h2o-vmr",
        repr(float(h2o_vmr)),
        "--co-vmr",
        repr(float(co_vmr)),
    )


def _case_arguments() -> tuple[tuple[str, tuple[str, ...]], ...]:
    narrow = ("--wavelength-bounds-micron", "2.2988", "2.30378")
    common = (
        "--layers",
        "80",
        "--temperature-top-k",
        "900",
        "--temperature-bottom-k",
        "1800",
        "--gravity-m-s2",
        "15",
    )
    baseline = (*common, *_vmr_arguments(BASELINE_H2O_VMR, BASELINE_CO_VMR))
    low_abundance = (
        *common,
        *_vmr_arguments(LOW_ABUNDANCE_H2O_VMR, LOW_ABUNDANCE_CO_VMR),
    )
    h2o_only = (*common, *_vmr_arguments(H2O_ONLY_H2O_VMR, H2O_ONLY_CO_VMR))
    co_only = (*common, *_vmr_arguments(CO_ONLY_H2O_VMR, CO_ONLY_CO_VMR))
    background_only = (
        *common,
        *_vmr_arguments(BACKGROUND_ONLY_H2O_VMR, BACKGROUND_ONLY_CO_VMR),
    )
    return (
        ("petitradtrans3_lbl_kband_grid_baseline.npz", (*baseline, *narrow)),
        (
            "petitradtrans3_lbl_kband_grid_isothermal.npz",
            (
                *baseline,
                "--temperature-top-k",
                "1300",
                "--temperature-bottom-k",
                "1300",
                *narrow,
            ),
        ),
        (
            "petitradtrans3_lbl_kband_grid_inversion.npz",
            (
                *baseline,
                "--temperature-top-k",
                "1800",
                "--temperature-bottom-k",
                "900",
                *narrow,
            ),
        ),
        (
            "petitradtrans3_lbl_kband_grid_low_abundance.npz",
            (*low_abundance, *narrow),
        ),
        (
            "petitradtrans3_lbl_kband_grid_high_gravity.npz",
            (*baseline, "--gravity-m-s2", "30", *narrow),
        ),
        (
            "petitradtrans3_lbl_kband_grid_h2o_only.npz",
            (*h2o_only, *narrow),
        ),
        (
            "petitradtrans3_lbl_kband_grid_co_only.npz",
            (*co_only, *narrow),
        ),
        (
            "petitradtrans3_lbl_kband_grid_background_only.npz",
            (*background_only, *narrow),
        ),
        (
            "petitradtrans3_lbl_kband_medium_reference_80.npz",
            (
                *baseline,
                "--wavelength-bounds-micron",
                "2.29",
                "2.32",
            ),
        ),
        (
            "petitradtrans3_lbl_kband_medium_reference_160.npz",
            (
                *baseline,
                "--layers",
                "160",
                "--wavelength-bounds-micron",
                "2.29",
                "2.32",
            ),
        ),
        (
            "petitradtrans3_lbl_kband_wide_reference.npz",
            (
                *baseline,
                "--wavelength-bounds-micron",
                "2.29",
                "2.35",
            ),
        ),
    )


def main() -> None:
    args = _parse_args()
    memory_gib = _validate_max_memory_gib(args.max_memory_gib)
    input_data = args.input_data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for relative_path in (H2O_TABLE, CO_TABLE):
        if not (input_data / relative_path).is_file():
            raise FileNotFoundError(input_data / relative_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    completed = 0
    skipped = 0
    for filename, case_arguments in _case_arguments():
        output = output_dir / filename
        if output.is_file() and not args.overwrite:
            print(f"skip existing oracle: {output}")
            skipped += 1
            continue
        command = [
            sys.executable,
            str(REFERENCE_SCRIPT),
            "--input-data",
            str(input_data),
            "--output",
            str(output),
            "--h2o-table",
            str(H2O_TABLE),
            "--co-table",
            str(CO_TABLE),
            "--max-memory-gib",
            f"{memory_gib:.12g}",
            *case_arguments,
        ]
        print(f"create oracle: {filename}")
        subprocess.run(command, check=True, env=environment)
        completed += 1
    print(f"oracle grid complete: created={completed}, skipped={skipped}")


if __name__ == "__main__":
    main()
