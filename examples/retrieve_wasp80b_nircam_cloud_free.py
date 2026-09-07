"""Run the current WASP-80b NIRCam retrieval configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

if __package__:
    from .retrieve_wasp69b_nircam_cloud_free import _run_current_configuration
else:
    from retrieve_wasp69b_nircam_cloud_free import _run_current_configuration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configurations" / "targets/WASP-80b/wasp80b_cloud_free_nircam_pg14_R1000.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the strict YAML workflow for WASP-80b NIRCam modes."""

    _run_current_configuration(default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == "__main__":
    main()
