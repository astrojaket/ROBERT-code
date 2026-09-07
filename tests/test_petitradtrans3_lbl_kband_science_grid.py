"""Direct tests for the VMR-first pRT K-band science-grid runner."""

from __future__ import annotations

import numpy as np
import pytest

from examples import run_petitradtrans3_lbl_kband_reference as reference
from examples import run_petitradtrans3_lbl_kband_science_grid as science_grid


def _flag_value(arguments: tuple[str, ...], flag: str) -> float:
    """Return one numeric command-line value from a case argument tuple."""

    index = arguments.index(flag)
    return float(arguments[index + 1])


def test_science_grid_cases_use_explicit_vmr_arguments_only() -> None:
    old_flags = {"--h2o-mass-fraction", "--co-mass-fraction"}
    expected = {
        "baseline": (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
        ),
        "isothermal": (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
        ),
        "inversion": (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
        ),
        "low_abundance": (
            science_grid.LOW_ABUNDANCE_H2O_VMR,
            science_grid.LOW_ABUNDANCE_CO_VMR,
        ),
        "high_gravity": (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
        ),
        "h2o_only": (
            science_grid.H2O_ONLY_H2O_VMR,
            science_grid.H2O_ONLY_CO_VMR,
        ),
        "co_only": (
            science_grid.CO_ONLY_H2O_VMR,
            science_grid.CO_ONLY_CO_VMR,
        ),
        "background_only": (
            science_grid.BACKGROUND_ONLY_H2O_VMR,
            science_grid.BACKGROUND_ONLY_CO_VMR,
        ),
        "medium_reference_80": (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
        ),
        "medium_reference_160": (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
        ),
        "wide_reference": (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
        ),
    }

    cases = science_grid._case_arguments()
    assert len(cases) == len(expected)
    for filename, arguments in cases:
        assert old_flags.isdisjoint(arguments)
        assert arguments.count("--h2o-vmr") == 1
        assert arguments.count("--co-vmr") == 1
        case_name = filename.removeprefix("petitradtrans3_lbl_kband_grid_")
        case_name = case_name.removeprefix("petitradtrans3_lbl_kband_")
        case_name = case_name.removesuffix(".npz")
        assert case_name in expected
        expected_h2o, expected_co = expected[case_name]
        assert _flag_value(arguments, "--h2o-vmr") == pytest.approx(expected_h2o)
        assert _flag_value(arguments, "--co-vmr") == pytest.approx(expected_co)


@pytest.mark.parametrize(
    ("h2o_vmr", "co_vmr", "target_h2o_mmr", "target_co_mmr"),
    [
        (
            science_grid.BASELINE_H2O_VMR,
            science_grid.BASELINE_CO_VMR,
            0.001,
            0.003,
        ),
        (
            science_grid.LOW_ABUNDANCE_H2O_VMR,
            science_grid.LOW_ABUNDANCE_CO_VMR,
            0.0001,
            0.0003,
        ),
        (
            science_grid.H2O_ONLY_H2O_VMR,
            science_grid.H2O_ONLY_CO_VMR,
            0.001,
            0.0,
        ),
        (
            science_grid.CO_ONLY_H2O_VMR,
            science_grid.CO_ONLY_CO_VMR,
            0.0,
            0.003,
        ),
        (
            science_grid.BACKGROUND_ONLY_H2O_VMR,
            science_grid.BACKGROUND_ONLY_CO_VMR,
            0.0,
            0.0,
        ),
    ],
)
def test_boundary_conversion_reproduces_former_mass_fraction_targets(
    h2o_vmr: float,
    co_vmr: float,
    target_h2o_mmr: float,
    target_co_mmr: float,
) -> None:
    vmr = reference._complete_vmr_state(h2o_vmr, co_vmr)
    mass_fractions, _ = reference._vmr_to_mass_fractions(vmr)

    np.testing.assert_allclose(sum(vmr.values()), 1.0, rtol=0.0, atol=2.0e-16)
    assert mass_fractions["H2O__POKAZATEL"] == pytest.approx(
        target_h2o_mmr,
        rel=0.0,
        abs=5.0e-15,
    )
    assert mass_fractions["CO__HITEMP"] == pytest.approx(
        target_co_mmr,
        rel=0.0,
        abs=5.0e-15,
    )


@pytest.mark.parametrize("value", (0.0, -0.1, 2.0, 2.1, float("nan"), float("inf")))
def test_memory_policy_rejects_non_positive_non_finite_and_two_gib_values(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="below 2 GiB"):
        science_grid._validate_max_memory_gib(value)
    with pytest.raises(ValueError, match="below 2 GiB"):
        reference._validate_max_memory_gib(value)


def test_memory_policy_defaults_below_two_gib_and_parser_rejects_two_gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["science-grid"])
    science_args = science_grid._parse_args()
    assert science_args.max_memory_gib == pytest.approx(1.9)

    monkeypatch.setattr("sys.argv", ["reference", "--max-memory-gib", "2.0"])
    with pytest.raises(SystemExit):
        reference._parse_args()

    monkeypatch.setattr("sys.argv", ["science-grid", "--max-memory-gib", "2.0"])
    with pytest.raises(SystemExit):
        science_grid._parse_args()
