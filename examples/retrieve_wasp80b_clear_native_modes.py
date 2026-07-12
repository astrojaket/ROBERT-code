"""Run the reusable native-mode clear scenario with the WASP-80b config."""

from __future__ import annotations

import os

os.environ["ROBERT_TARGET_CONFIG"] = "examples.wasp80b_target"

from retrieve_wasp69b_clear_native_modes import main


if __name__ == "__main__":
    main()
