"""Run the reusable Mie-cloud scenarios with the WASP-80b target config."""

from __future__ import annotations

import os

os.environ["ROBERT_TARGET_CONFIG"] = "examples.wasp80b_target"

from retrieve_wasp69b_mie_cloud import main


if __name__ == "__main__":
    main()
