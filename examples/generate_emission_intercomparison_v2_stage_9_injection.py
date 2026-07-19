#!/usr/bin/env python3
"""Generate one approved Stage-9 native injection mean on Glamdring."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_intercomparison_v2_stage_9_native import (  # noqa: E402
    build_native_forward,
    load_common_contract,
    truth_parameters,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (  # noqa: E402
    FRAMEWORKS,
    SCENARIOS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("scenario", choices=tuple(item.name for item in SCENARIOS))
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--approved",
        action="store_true",
        help="required acknowledgement that the user approved native forward execution",
    )
    args = parser.parse_args()
    if not args.approved:
        parser.error("--approved is required before executing a native injection")
    if os.environ.get("STAGE9_CLUSTER") != "glamdring":
        raise RuntimeError(
            "Stage-9 native injection generation is restricted to Glamdring"
        )

    project = args.project_root.expanduser().resolve()
    common_path = project / "contracts" / "common_contract.json"
    common = load_common_contract(common_path)
    forward = build_native_forward(args.framework, common, args.scenario)
    truth = truth_parameters(common, args.scenario)
    eclipse = forward.eclipse_depth(truth)
    if not np.all(np.isfinite(eclipse)):
        raise RuntimeError("native injection contains non-finite eclipse depths")
    destination = (
        project / "injections" / args.framework / args.scenario / "native_mean.npz"
    )
    if destination.exists():
        raise RuntimeError(
            f"refusing to overwrite an existing injection: {destination}"
        )
    metadata = {
        "schema_version": "1.0",
        "stage": 9,
        "track": "track_b_native_retrieval",
        "framework": args.framework,
        "scenario": args.scenario,
        "truth_parameters": truth,
        "common_contract_sha256": _sha256(common_path),
        "python": os.path.realpath(sys.executable),
        "platform": platform.platform(),
        "numpy_version": importlib.metadata.version("numpy"),
        "framework_version": importlib.metadata.version(
            {
                "robert": "robert-exoplanets",
                "picaso": "picaso",
                "petitradtrans": "petitRADTRANS",
            }[args.framework]
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        wavelength_micron=forward.r100_centers,
        eclipse_depth=eclipse,
        native_wavelength_micron=forward.last_native_wavelength,
        native_flux_w_m2_m=forward.last_native_flux,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    checksum = _sha256(destination)
    destination.with_suffix(".sha256").write_text(
        f"{checksum}  {destination.name}\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
