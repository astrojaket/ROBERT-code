"""Run the reusable NIRCam clear scenario with the WASP-80b target config."""

from __future__ import annotations

import os

os.environ["ROBERT_TARGET_CONFIG"] = "examples.wasp80b_target"

from retrieve_wasp69b_nircam_clear import main


if __name__ == "__main__":
    main()
