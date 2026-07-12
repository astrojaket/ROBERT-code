"""Benchmark correlated-k random overlap across molecule and grid counts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from robert_exoplanets import random_overlap_species_tau

OUTPUT = Path(__file__).resolve().parent / "outputs" / "random_overlap_benchmark.json"


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def _case(
    *,
    n_species: int,
    n_layers: int,
    n_wavelength: int,
    n_g: int,
    repeats: int,
    warmups: int,
    backend: str,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * n_species + n_wavelength)
    species_tau = np.sort(
        np.exp(
            rng.uniform(
                -20.0,
                5.0,
                size=(n_species, n_layers, n_wavelength, n_g),
            )
        ),
        axis=-1,
    )
    weights = np.arange(1.0, n_g + 1.0)
    weights /= np.sum(weights)

    small_tau = species_tau[:, : min(2, n_layers), : min(3, n_wavelength), :]
    reference = random_overlap_species_tau(small_tau, weights, backend="numpy")
    candidate = random_overlap_species_tau(small_tau, weights, backend=backend)
    np.testing.assert_allclose(candidate, reference, rtol=2.0e-12, atol=2.0e-12)

    for _ in range(warmups):
        random_overlap_species_tau(species_tau, weights, backend=backend)
    durations = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = random_overlap_species_tau(species_tau, weights, backend=backend)
        durations.append(perf_counter() - started)
    if result is None:  # pragma: no cover - repeats is validated by the CLI
        raise RuntimeError("random-overlap benchmark did not run")

    median = float(np.median(durations))
    return {
        "n_species": n_species,
        "n_layers": n_layers,
        "n_wavelength": n_wavelength,
        "n_g": n_g,
        "input_mib": species_tau.nbytes / 2**20,
        "durations_s": durations,
        "median_s": median,
        "layer_wavelength_points_per_second": n_layers * n_wavelength / median,
        "checksum": float(np.sum(result)),
        "numpy_reference_max_abs_difference": float(
            np.max(np.abs(candidate - reference))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", type=_positive_csv, default=(1, 3, 6))
    parser.add_argument("--wavelengths", type=_positive_csv, default=(280, 1591))
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--g-ordinates", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "numpy", "numba"), default="auto")
    parser.add_argument("--seed", type=int, default=1928)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.layers < 1 or args.g_ordinates < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error(
            "layers, g ordinates, and repeats must be positive; warmups must be non-negative"
        )

    try:
        import numba

        numba_version = numba.__version__
        numba_threads = numba.get_num_threads()
    except ImportError:
        numba_version = None
        numba_threads = None

    cases = [
        _case(
            n_species=n_species,
            n_layers=args.layers,
            n_wavelength=n_wavelength,
            n_g=args.g_ordinates,
            repeats=args.repeats,
            warmups=args.warmups,
            backend=args.backend,
            seed=args.seed,
        )
        for n_wavelength in args.wavelengths
        for n_species in args.species
    ]
    report = {
        "schema_version": 1,
        "benchmark": "correlated_k_random_overlap_scaling",
        "backend": args.backend,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": numba_version,
        "numba_threads": numba_threads,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
