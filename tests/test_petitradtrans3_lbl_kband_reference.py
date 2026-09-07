"""Direct tests for the VMR-first pRT K-band oracle helpers."""

from __future__ import annotations

import pytest

from examples import run_petitradtrans3_lbl_kband_reference as reference


def test_default_vmr_state_reproduces_former_default_boundary_composition() -> None:
    vmr = reference._complete_vmr_state(
        reference.DEFAULT_H2O_VMR,
        reference.DEFAULT_CO_VMR,
    )
    mass_fractions, mean_molar_mass = reference._vmr_to_mass_fractions(vmr)

    assert sum(vmr.values()) == pytest.approx(1.0)
    assert mean_molar_mass == pytest.approx(2.3217263131165278)
    assert mass_fractions["H2"] == pytest.approx(0.738)
    assert mass_fractions["He"] == pytest.approx(0.258)
    assert mass_fractions["H2O__POKAZATEL"] == pytest.approx(0.001)
    assert mass_fractions["CO__HITEMP"] == pytest.approx(0.003)


def test_complete_vmr_state_scales_only_the_fixed_background_complement() -> None:
    vmr = reference._complete_vmr_state(1.0e-3, 2.0e-3)

    assert vmr["H2O__POKAZATEL"] == pytest.approx(1.0e-3)
    assert vmr["CO__HITEMP"] == pytest.approx(2.0e-3)
    assert vmr["H2"] / vmr["He"] == pytest.approx(
        reference.BACKGROUND_VMR_RATIO["H2"] / reference.BACKGROUND_VMR_RATIO["He"]
    )
    assert sum(vmr.values()) == pytest.approx(1.0)


def test_private_boundary_conversion_requires_a_complete_vmr_state() -> None:
    with pytest.raises(ValueError, match="exactly the pRT gas species"):
        reference._vmr_to_mass_fractions({"H2": 1.0})
    with pytest.raises(ValueError, match="sum to one"):
        reference._vmr_to_mass_fractions(
            {
                "H2": 0.7,
                "He": 0.2,
                "H2O__POKAZATEL": 0.001,
                "CO__HITEMP": 0.003,
            }
        )


def test_cli_exposes_vmr_inputs_and_rejects_old_mass_fraction_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_petitradtrans3_lbl_kband_reference.py",
            "--h2o-vmr",
            "1e-4",
            "--co-vmr",
            "2e-4",
        ],
    )
    arguments = reference._parse_args()
    assert arguments.h2o_vmr == pytest.approx(1.0e-4)
    assert arguments.co_vmr == pytest.approx(2.0e-4)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_petitradtrans3_lbl_kband_reference.py",
            "--h2o-mass-fraction",
            "1e-3",
        ],
    )
    with pytest.raises(SystemExit):
        reference._parse_args()
