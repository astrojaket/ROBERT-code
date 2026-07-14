"""Tests for the external NEMESIS Jupiter benchmark harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def test_cirsdrv_native_forward_parser_reads_first_spectral_block(
    tmp_path: Path,
) -> None:
    module = _benchmark_module()
    output = tmp_path / "cirstest.out"
    output.write_text(
        """           1      !! npath
           3      !! nwave
 XLIM, YLIM :
   5.0 7.0 1.0e-9 3.0e-9
 XLABEL, YLABEL :
 Wavenumbers [cm-1]
 Radiance [W cm-2 sr-2 (cm-1)-1]
   5.0 1.0e-9
   6.0 2.0e-9
   7.0 3.0e-9
           2      !! nconv
   5.0 9.0e-9
   7.0 9.0e-9
""",
        encoding="utf-8",
    )

    wavenumber, radiance = module.read_cirsdrv_native_forward(output)

    np.testing.assert_allclose(wavenumber, [5.0, 6.0, 7.0])
    np.testing.assert_allclose(radiance, [1.0e-9, 2.0e-9, 3.0e-9])


def _benchmark_module():
    path = Path(__file__).parents[1] / "examples" / "benchmark_jupiter_nemesis.py"
    spec = importlib.util.spec_from_file_location("benchmark_jupiter_nemesis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
