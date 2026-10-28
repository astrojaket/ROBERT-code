"""Tests for the opt-in FastChem condensate diagnostics adapter."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets.atmosphere import fastchem_condensates
from robert_exoplanets.atmosphere.fastchem_condensates import (
    FastChemCondensateDiagnostics,
)
from robert_exoplanets.core import (
    PressureGrid,
    RobertConfigError,
    RobertValidationError,
)


class _FakeInput:
    def __init__(self) -> None:
        self.pressure: list[float] = []
        self.temperature: list[float] = []
        self.equilibrium_condensation = False
        self.rainout_condensation = False


class _FakeOutput:
    def __init__(self) -> None:
        self.number_densities: list[list[float]] = []
        self.number_densities_cond: list[list[float]] = []
        self.element_cond_degree: list[list[float]] = []
        self.element_conserved: list[list[int]] = []
        self.fastchem_flag: list[int] = []
        self.nb_chemistry_iterations: list[int] = []
        self.nb_cond_iterations: list[int] = []
        self.nb_iterations: list[int] = []


class _FakeFastChem:
    condensate_symbols = ("Fe(s,l)", "Mg2SiO4(s,l)")
    condensate_names = ("Iron", "Forsterite")
    gas_species_symbols = ("H2", "He")
    gas_species_weights = (2.01588, 4.002602)
    element_symbols = ("H", "Fe", "Mg", "Si", "O")
    element_weights = (1.00784, 55.845, 24.305, 28.085, 15.999)

    def __init__(
        self,
        *,
        return_flag: int = 0,
        layer_flags: list[int] | None = None,
        element_flags: list[list[int]] | None = None,
        break_pressure_closure: bool = False,
        condensate_weights: tuple[float, ...] = (0.0, 0.0),
    ):
        self.return_flag = return_flag
        self.layer_flags = layer_flags
        self.element_flags = element_flags
        self.break_pressure_closure = break_pressure_closure
        self.condensate_weights = condensate_weights
        self.last_input: _FakeInput | None = None

    def getGasSpeciesNumber(self) -> int:
        return len(self.gas_species_symbols)

    def getGasSpeciesSymbol(self, index: int) -> str:
        return self.gas_species_symbols[index]

    def getGasSpeciesWeight(self, index: int) -> float:
        return self.gas_species_weights[index]

    def getCondSpeciesNumber(self) -> int:
        return len(self.condensate_symbols)

    def getElementNumber(self) -> int:
        return len(self.element_symbols)

    def getCondSpeciesSymbol(self, index: int) -> str:
        return self.condensate_symbols[index]

    def getCondSpeciesName(self, index: int) -> str:
        return self.condensate_names[index]

    def getCondSpeciesIndex(self, symbol: str) -> int:
        try:
            return self.condensate_symbols.index(symbol)
        except ValueError:
            return 9999999

    def getElementSymbol(self, index: int) -> str:
        return self.element_symbols[index]

    def getElementWeight(self, index: int) -> float:
        return self.element_weights[index]

    def getCondSpeciesWeight(self, index: int) -> float:
        # The default exercises the zero-weight fallback used by FastChem 3.1.2.
        return self.condensate_weights[index]

    def calcDensities(self, input_data: _FakeInput, output_data: _FakeOutput) -> int:
        self.last_input = input_data
        pressure = np.asarray(input_data.pressure, dtype=float)
        temperature = np.asarray(input_data.temperature, dtype=float)
        n_layers = pressure.size
        gas_number_density = pressure * 1.0e6 / (1.380649e-16 * temperature)
        if self.break_pressure_closure:
            gas_number_density *= 2.0
        output_data.number_densities = np.column_stack(
            (0.7 * gas_number_density, 0.3 * gas_number_density)
        ).tolist()
        output_data.number_densities_cond = np.column_stack(
            (pressure, pressure + 100.0)
        ).tolist()
        output_data.element_cond_degree = np.column_stack(
            (
                pressure / (pressure + 1.0),
                1.0 / (pressure + 1.0),
                np.zeros((n_layers, len(self.element_symbols) - 2)),
            )
        ).tolist()
        output_data.element_conserved = (
            [[1] * len(self.element_symbols) for _ in range(n_layers)]
            if self.element_flags is None
            else self.element_flags
        )
        output_data.fastchem_flag = (
            [0] * n_layers if self.layer_flags is None else self.layer_flags
        )
        output_data.nb_chemistry_iterations = list(range(1, n_layers + 1))
        output_data.nb_cond_iterations = list(range(2, n_layers + 2))
        output_data.nb_iterations = list(range(3, n_layers + 3))
        return self.return_flag


def _fake_api(
    *,
    return_flag: int = 0,
    layer_flags: list[int] | None = None,
    element_flags: list[list[int]] | None = None,
    break_pressure_closure: bool = False,
    condensate_weights: tuple[float, ...] = (0.0, 0.0),
) -> tuple[SimpleNamespace, _FakeFastChem]:
    solver = _FakeFastChem(
        return_flag=return_flag,
        layer_flags=layer_flags,
        element_flags=element_flags,
        break_pressure_closure=break_pressure_closure,
        condensate_weights=condensate_weights,
    )
    api = SimpleNamespace(
        FastChem=lambda *_args: solver,
        FastChemInput=_FakeInput,
        FastChemOutput=_FakeOutput,
        FASTCHEM_SUCCESS=0,
        FASTCHEM_UNKNOWN_SPECIES=9999999,
        __version__="fake-pyfastchem-1",
    )
    return api, solver


def _fastchem_data_tree(root: Path) -> Path:
    files = (
        root / "input" / "element_abundances" / "asplund_2009.dat",
        root / "input" / "logK" / "logK.dat",
        root / "input" / "logK" / "logK_condensates.dat",
    )
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "logK_condensates.dat":
            content = (
                "# FastChem condensate fixture header 1\n"
                "# FastChem condensate fixture header 2\n"
                "# FastChem condensate fixture header 3\n"
                "Fe(s,l) Iron : Fe 1 # fixture\n"
                "sl\n1.0\n0.0\n\n"
                "Mg2SiO4(s,l) Forsterite : Mg 2 Si 1 O 4 # fixture\n"
                "sl\n1.0\n0.0\n"
            )
        else:
            content = f"fixture: {path.name}\n"
        path.write_text(content, encoding="utf-8")
    return root


def _increasing_grid() -> PressureGrid:
    return PressureGrid.logspace(0.1, 100.0, n_layers=3, unit="bar")


def _use_fake_api(monkeypatch, api: SimpleNamespace) -> None:
    monkeypatch.setattr(
        fastchem_condensates,
        "_load_pyfastchem",
        lambda: (api, "fake-pyfastchem-1"),
    )


def test_rainout_runs_deep_to_upper_and_restores_grid_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, solver = _fake_api()
    _use_fake_api(monkeypatch, api)
    grid = _increasing_grid()
    temperature = np.array([900.0, 1000.0, 1100.0])
    model = FastChemCondensateDiagnostics(
        _fastchem_data_tree(tmp_path / "fastchem"), mode="rainout"
    )

    result = model.evaluate(grid, temperature)

    assert solver.last_input is not None
    np.testing.assert_allclose(solver.last_input.pressure, grid.centers[::-1])
    np.testing.assert_allclose(solver.last_input.temperature, temperature[::-1])
    assert solver.last_input.rainout_condensation is True
    assert solver.last_input.equilibrium_condensation is False
    np.testing.assert_allclose(result.condensate_number_density_cm3[:, 0], grid.centers)
    np.testing.assert_allclose(result.temperature_k, temperature)
    np.testing.assert_allclose(
        result.element_condensation_degree[:, 0],
        grid.centers / (grid.centers + 1.0),
    )
    np.testing.assert_array_equal(result.chemistry_iterations, [3, 2, 1])
    assert result.metadata["fastchem_rainout_input_order"] == (
        "decreasing_pressure_deep_to_upper"
    )
    assert result.metadata["fastchem_output_layer_order"] == "input_pressure_grid_order"
    assert result.metadata["fastchem_condensate_mass_interpretation"] == (
        "local_rainout_mass_equivalent_not_cloud_retention"
    )
    assert result.condensate_number_density_cm3.flags.writeable is False


def test_rainout_results_match_when_input_grid_orientation_is_reversed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _solver = _fake_api()
    _use_fake_api(monkeypatch, api)
    increasing_grid = _increasing_grid()
    decreasing_grid = PressureGrid(
        edges=increasing_grid.edges[::-1],
        centers=increasing_grid.centers[::-1],
        unit=increasing_grid.unit,
    )
    temperature = np.array([900.0, 1000.0, 1100.0])
    model = FastChemCondensateDiagnostics(
        _fastchem_data_tree(tmp_path / "fastchem"), mode="rainout"
    )

    increasing_result = model.evaluate(increasing_grid, temperature)
    decreasing_result = model.evaluate(decreasing_grid, temperature[::-1])

    np.testing.assert_allclose(
        increasing_result.condensate_number_density_cm3,
        decreasing_result.condensate_number_density_cm3[::-1],
    )
    np.testing.assert_allclose(
        increasing_result.element_condensation_degree,
        decreasing_result.element_condensation_degree[::-1],
    )


def test_equilibrium_preserves_grid_order_and_selected_species_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, solver = _fake_api()
    _use_fake_api(monkeypatch, api)
    grid = _increasing_grid()
    model = FastChemCondensateDiagnostics(
        _fastchem_data_tree(tmp_path / "fastchem"),
        mode="equilibrium",
        condensate_symbols=("Mg2SiO4(s,l)", "Fe(s,l)"),
    )

    result = model.evaluate(grid, np.full(grid.n_layers, 1200.0))

    assert solver.last_input is not None
    np.testing.assert_allclose(solver.last_input.pressure, grid.centers)
    assert solver.last_input.equilibrium_condensation is True
    assert solver.last_input.rainout_condensation is False
    assert result.condensate_symbols == ("Mg2SiO4(s,l)", "Fe(s,l)")
    np.testing.assert_allclose(
        result.condensate_number_density_cm3[:, 0], grid.centers + 100.0
    )
    np.testing.assert_allclose(result.condensate_number_density_cm3[:, 1], grid.centers)
    assert result.metadata["fastchem_condensation_mode"] == "equilibrium"
    expected_hash = sha256(
        (tmp_path / "fastchem" / "input" / "logK" / "logK_condensates.dat").read_bytes()
    ).hexdigest()
    assert result.metadata["fastchem_condensate_equilibrium_sha256"] == expected_hash


def test_mass_conversion_uses_matched_gas_and_all_condensates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _solver = _fake_api()
    _use_fake_api(monkeypatch, api)
    grid = _increasing_grid()
    temperature = np.array([900.0, 1000.0, 1100.0])
    result = FastChemCondensateDiagnostics(
        _fastchem_data_tree(tmp_path / "fastchem"),
        mode="equilibrium",
        condensate_symbols=("Mg2SiO4(s,l)",),
    ).evaluate(grid, temperature)

    amu_kg = 1.66053906892e-27
    kg_m3_per_u_cm3 = amu_kg * 1.0e6
    pressure_bar = grid.centers
    gas_number_density = pressure_bar * 1.0e6 / (1.380649e-16 * temperature)
    gas_mass_u_cm3 = gas_number_density * (
        0.7 * 2.01588 + 0.3 * 4.002602
    )
    iron_weight_u = 55.845
    forsterite_weight_u = 2.0 * 24.305 + 28.085 + 4.0 * 15.999
    iron_mass_density = pressure_bar * iron_weight_u * kg_m3_per_u_cm3
    forsterite_mass_density = (
        (pressure_bar + 100.0) * forsterite_weight_u * kg_m3_per_u_cm3
    )
    total_condensate_mass_density = iron_mass_density + forsterite_mass_density
    total_mass_density = gas_mass_u_cm3 * kg_m3_per_u_cm3 + total_condensate_mass_density

    assert result.condensate_symbols == ("Mg2SiO4(s,l)",)
    assert result.all_condensate_symbols == ("Fe(s,l)", "Mg2SiO4(s,l)")
    np.testing.assert_allclose(result.condensate_molecular_weight_u, [forsterite_weight_u])
    np.testing.assert_allclose(result.gas_number_density_cm3, gas_number_density)
    np.testing.assert_allclose(
        result.gas_mass_density_kg_m3,
        gas_mass_u_cm3 * kg_m3_per_u_cm3,
    )
    np.testing.assert_allclose(
        result.total_condensate_mass_density_kg_m3,
        total_condensate_mass_density,
    )
    np.testing.assert_allclose(
        result.condensate_mass_density_kg_m3[:, 0], forsterite_mass_density
    )
    np.testing.assert_allclose(
        result.condensate_mass_fraction_of_total[:, 0],
        forsterite_mass_density / total_mass_density,
    )
    np.testing.assert_allclose(
        result.condensate_mass_mixing_ratio_gas[:, 0],
        forsterite_mass_density / result.gas_mass_density_kg_m3,
    )
    np.testing.assert_allclose(
        result.total_condensate_mass_fraction_of_total,
        total_condensate_mass_density / total_mass_density,
    )
    assert result.metadata["fastchem_condensate_weight_source"] == (
        "loaded_condensate_stoichiometry_times_fastchem_element_weights"
    )
    assert result.metadata["fastchem_condensate_mass_fraction_denominator"] == (
        "gas_mass_plus_all_fastchem_condensate_mass"
    )
    assert result.condensate_mass_fraction_of_total.flags.writeable is False


def test_element_conservation_failure_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _solver = _fake_api(element_flags=[[1] * 5, [1] * 4 + [0], [1] * 5])
    _use_fake_api(monkeypatch, api)
    model = FastChemCondensateDiagnostics(_fastchem_data_tree(tmp_path / "fastchem"))

    with pytest.raises(RobertValidationError, match="element conservation failed"):
        model.evaluate(_increasing_grid(), np.full(3, 1200.0))


def test_positive_fastchem_condensate_weights_match_stoichiometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _solver = _fake_api(condensate_weights=(55.845, 140.691))
    _use_fake_api(monkeypatch, api)
    result = FastChemCondensateDiagnostics(
        _fastchem_data_tree(tmp_path / "fastchem")
    ).evaluate(_increasing_grid(), np.full(3, 1200.0))

    np.testing.assert_allclose(result.all_condensate_molecular_weight_u, [55.845, 140.691])
    assert result.metadata["fastchem_condensate_weight_source"] == (
        "pyfastchem_getCondSpeciesWeight"
    )


def test_positive_fastchem_condensate_weight_mismatch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _solver = _fake_api(condensate_weights=(55.0, 140.691))
    _use_fake_api(monkeypatch, api)
    model = FastChemCondensateDiagnostics(_fastchem_data_tree(tmp_path / "fastchem"))

    with pytest.raises(RobertConfigError, match="do not match table stoichiometry"):
        model.evaluate(_increasing_grid(), np.full(3, 1200.0))


def test_gas_pressure_closure_failure_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _solver = _fake_api(break_pressure_closure=True)
    _use_fake_api(monkeypatch, api)
    model = FastChemCondensateDiagnostics(_fastchem_data_tree(tmp_path / "fastchem"))

    with pytest.raises(RobertValidationError, match="do not reproduce the input pressure"):
        model.evaluate(_increasing_grid(), np.full(3, 1200.0))


@pytest.mark.parametrize(
    ("return_flag", "layer_flags"),
    [(1, None), (0, [0, 1, 0])],
)
def test_fastchem_non_success_status_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_flag: int,
    layer_flags: list[int] | None,
) -> None:
    api, _solver = _fake_api(return_flag=return_flag, layer_flags=layer_flags)
    _use_fake_api(monkeypatch, api)
    model = FastChemCondensateDiagnostics(
        _fastchem_data_tree(tmp_path / "fastchem"), mode="rainout"
    )

    with pytest.raises(RobertValidationError, match="FastChem condensate calculation"):
        model.evaluate(_increasing_grid(), np.full(3, 1200.0))


def test_invalid_inputs_and_unknown_condensates_fail_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _solver = _fake_api()
    _use_fake_api(monkeypatch, api)
    data_root = _fastchem_data_tree(tmp_path / "fastchem")

    with pytest.raises(RobertValidationError, match="mode"):
        FastChemCondensateDiagnostics(data_root, mode="none")  # type: ignore[arg-type]
    with pytest.raises(RobertValidationError, match="temperature_k"):
        FastChemCondensateDiagnostics(data_root).evaluate(
            _increasing_grid(), [1200.0, np.nan, 1200.0]
        )
    with pytest.raises(RobertValidationError, match="pressure unit"):
        FastChemCondensateDiagnostics(data_root).evaluate(
            PressureGrid.logspace(1.0, 10.0, 3, unit="kPa"),
            np.full(3, 1200.0),
        )
    with pytest.raises(RobertConfigError, match="does not know condensate"):
        FastChemCondensateDiagnostics(
            data_root, condensate_symbols=("not-a-condensate",)
        ).evaluate(_increasing_grid(), np.full(3, 1200.0))


def test_missing_condensate_table_fails_before_loading_optional_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fastchem"
    (data_root / "input" / "element_abundances").mkdir(parents=True)
    (data_root / "input" / "logK").mkdir(parents=True)
    (data_root / "input" / "element_abundances" / "asplund_2009.dat").touch()
    (data_root / "input" / "logK" / "logK.dat").touch()
    monkeypatch.setattr(
        fastchem_condensates,
        "_load_pyfastchem",
        lambda: pytest.fail(
            "PyFastChem should not load when required tables are missing"
        ),
    )

    with pytest.raises(RobertConfigError, match="logK_condensates.dat"):
        FastChemCondensateDiagnostics(data_root).evaluate(
            _increasing_grid(), np.full(3, 1200.0)
        )


def _optional_fastchem_root() -> Path | None:
    candidates = [
        os.environ.get("ROBERT_FASTCHEM_PATH"),
        os.environ.get("FASTCHEM_PATH"),
        str(Path(__file__).parents[1] / "data" / "chemistry" / "fastchem"),
        str(Path.home() / "Dropbox" / "fastchem"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if all(
            (root / relative).is_file()
            for relative in (
                "input/element_abundances/asplund_2009.dat",
                "input/logK/logK.dat",
                "input/logK/logK_condensates.dat",
            )
        ):
            return root
    return None


def test_real_fastchem_condensate_modes_when_optional_data_are_available() -> None:
    pytest.importorskip("pyfastchem")
    fastchem_path = _optional_fastchem_root()
    if fastchem_path is None:
        pytest.skip("FastChem condensate input tables are not available")

    grid = PressureGrid.logspace(0.1, 10.0, n_layers=3, unit="bar")
    temperature = np.array([1000.0, 1200.0, 1400.0])
    for mode in ("equilibrium", "rainout"):
        result = FastChemCondensateDiagnostics(fastchem_path, mode=mode).evaluate(
            grid, temperature
        )
        assert result.mode == mode
        assert result.condensate_number_density_cm3.shape == (
            grid.n_layers,
            len(result.condensate_symbols),
        )
        assert len(result.condensate_symbols) > 0
        assert result.element_condensation_degree.shape == (
            grid.n_layers,
            len(result.element_symbols),
        )
        assert np.all(result.fastchem_flag == 0)
        assert np.all(result.element_conserved == 1)
        assert np.all(np.isfinite(result.condensate_number_density_cm3))
        assert np.all(np.isfinite(result.gas_species_number_density_cm3))
        assert np.all(np.isfinite(result.gas_mass_density_kg_m3))
        assert np.all(np.isfinite(result.condensate_mass_fraction_of_total))
        assert np.all(result.total_condensate_mass_fraction_of_total <= 1.0)
        assert result.metadata["fastchem_version"]
        assert result.metadata["fastchem_condensate_number_density_unit"] == "cm^-3"
        assert result.metadata["fastchem_mass_density_unit"] == "kg m^-3"
