"""Benchmark optional JAX conservative RORR against ROBERT references.

The fixture uses the six WASP-69b F322W2 correlated-k tables, their native
16-point g quadrature, 80 layers, and all cached observation bins. JIT compile
time is reported separately from warmed calls. Set ``JAX_ENABLE_X64=1``;
float32 is intentionally rejected by the science-grade JAX backend.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "robert-numba-cache")
)

import numpy as np

from robert_exoplanets.rt.jax_random_overlap import (
    jax_random_overlap_device_info,
    jax_random_overlap_species_tau,
)
from robert_exoplanets.rt.random_overlap import random_overlap_species_tau
from retrieve_wasp69b_nircam_clear import SPECIES, _load_table


def _median_seconds(call, repeats: int) -> float:
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        elapsed.append(perf_counter() - start)
    return float(np.median(elapsed))


def run(repeats: int = 7) -> dict[str, object]:
    tables = [_load_table("f322w2", species) for species in SPECIES]
    n_layers = 80
    sigma_band = np.stack(
        [
            table.kcoeff[
                table.pressure_bar.size // 2,
                table.temperature_K.size // 2,
            ]
            for table in tables
        ],
        axis=0,
    )
    sigma = np.repeat(sigma_band[:, None, :, :], n_layers, axis=1)
    layer_column = np.geomspace(1.0e20, 1.0e27, n_layers)
    vmr = np.stack(
        [np.full(n_layers, 10.0 ** (-3.0 - index)) for index in range(len(SPECIES))]
    )
    tau = np.ascontiguousarray(
        sigma * 1.0e-4 * vmr[:, :, None, None] * layer_column[None, :, None, None]
    )
    weights = np.asarray(tables[0].g_weights, dtype=float)

    reference = random_overlap_species_tau(tau, weights, backend="numpy")
    random_overlap_species_tau(tau, weights, backend="numba")
    compile_start = perf_counter()
    jax_result = jax_random_overlap_species_tau(tau, weights, platform="cpu")
    jax_compile_and_first_call = perf_counter() - compile_start

    numba_result = random_overlap_species_tau(tau, weights, backend="numba")
    difference = jax_result - reference
    scale = np.maximum(np.abs(reference), 1.0e-300)
    return {
        "schema_version": 1,
        "fixture": "WASP-69b F322W2 correlated-k random overlap",
        "species": list(SPECIES),
        "shape": list(tau.shape),
        "jax_devices": list(jax_random_overlap_device_info("cpu")),
        "precision": "float64",
        "repeats": repeats,
        "jax_compile_and_first_call_seconds": jax_compile_and_first_call,
        "warmed_seconds": {
            "numpy_reference": _median_seconds(
                lambda: random_overlap_species_tau(tau, weights, backend="numpy"),
                repeats,
            ),
            "numba": _median_seconds(
                lambda: random_overlap_species_tau(tau, weights, backend="numba"),
                repeats,
            ),
            "jax_host_roundtrip": _median_seconds(
                lambda: jax_random_overlap_species_tau(
                    tau, weights, platform="cpu"
                ),
                repeats,
            ),
        },
        "jax_vs_numpy": {
            "max_abs": float(np.max(np.abs(difference))),
            "max_relative": float(np.max(np.abs(difference) / scale)),
            "rms_relative": float(np.sqrt(np.mean((difference / scale) ** 2))),
        },
        "numba_vs_numpy": {
            "max_abs": float(np.max(np.abs(numba_result - reference))),
            "max_relative": float(
                np.max(np.abs(numba_result - reference) / scale)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run(args.repeats)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
