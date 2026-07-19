"""Compare ROBERT's CLR transform with POSEIDON's reference implementation.

The POSEIDON function is extracted from its source AST so this comparison does
not import POSEIDON's optional sampler, opacity, or MPI dependencies.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from robert_exoplanets import centered_log_ratio_prior_transform


def _poseidon_clr_function(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "CLR_Prior"
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["CLR_Prior"], sha256(source.encode("utf-8")).hexdigest()


def compare(poseidon_root: Path, *, samples: int, seed: int) -> dict[str, object]:
    source = poseidon_root / "POSEIDON" / "retrieval.py"
    if not source.is_file():
        raise FileNotFoundError(f"POSEIDON retrieval source not found: {source}")
    poseidon_clr, source_sha = _poseidon_clr_function(source)
    rng = np.random.default_rng(seed)
    cases: dict[str, object] = {}
    for n_free in (1, 2, 3, 6):
        cube = rng.random((samples, n_free))
        maximum_absolute_difference = 0.0
        accepted = 0
        for row in cube:
            poseidon = np.asarray(poseidon_clr(row, -12.0), dtype=float)
            robert = centered_log_ratio_prior_transform(
                row, lower_log10_vmr=-12.0
            )
            difference = float(np.max(np.abs(robert - poseidon[1:])))
            maximum_absolute_difference = max(
                maximum_absolute_difference, difference
            )
            accepted += int(not np.all(robert == -50.0))
        cases[str(n_free)] = {
            "total_composition_categories": n_free + 1,
            "samples": samples,
            "accepted": accepted,
            "maximum_absolute_log10_vmr_difference": maximum_absolute_difference,
        }
    commit = subprocess.run(
        ["git", "-C", str(poseidon_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "comparison": "ROBERT centered_log_ratio_prior_transform vs POSEIDON CLR_Prior",
        "poseidon_commit": commit,
        "poseidon_retrieval_source_sha256": source_sha,
        "lower_log10_vmr": -12.0,
        "seed": seed,
        "cases": cases,
        "all_cases_identical": all(
            case["maximum_absolute_log10_vmr_difference"] <= 2.0e-14
            for case in cases.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poseidon-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.poseidon_root, samples=args.samples, seed=args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["all_cases_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
