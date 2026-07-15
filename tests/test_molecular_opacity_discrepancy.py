from __future__ import annotations

import numpy as np

from examples.benchmark_picaso_robert_molecular_opacity_discrepancy import (
    _ppm_metrics,
    _species_metrics,
)


def test_ppm_metrics_report_scale_and_offset() -> None:
    metrics = _ppm_metrics(np.array([-3.0, 4.0]))

    assert metrics == {"rms_ppm": np.sqrt(12.5), "median_ppm": 0.5, "max_abs_ppm": 4.0}


def test_species_metrics_ignore_transparent_tail_for_active_summary() -> None:
    wavelength = np.array([1.0, 2.0, 3.0])
    robert = np.array([1.0, 0.5, 1.0e-20])
    picaso = np.array([1.0, 0.25, 1.0e-30])

    metrics = _species_metrics(wavelength, robert, picaso)

    assert metrics["active_bins"] == 2
    assert np.isclose(metrics["active_max_abs_log10_tau_difference_dex"], np.log10(2.0))
