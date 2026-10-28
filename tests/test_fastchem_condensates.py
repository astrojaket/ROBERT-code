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
        self.number_densities_cond: list[list[float]] = []
        self.element_cond_degree: list[list[float]] = []
        self.fastchem_flag: list[int] = []
        self.nb_chemistry_iterations: list[int] = []
        self.nb_cond_iterations: list[int] = []
        self.nb_iterations: list[int] = []


class _FakeFastChem:
    condensate_symbols = ("Fe(s,l)", "Mg2SiO4(s,l)")
    condensate_names = ("Iron", "Forsterite")
    element_symbols = ("H", "Fe")

    def __init__(self, *, return_flag: int = 0, layer_flags: list[int] | None = None):
        self.return_flag = return_flag
        self.layer_flags = layer_flags
        self.last_input: _FakeInput | None = None

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

    def calcDensities(self, input_data: _FakeInput, output_data: _FakeOutput) -> int:
        self.last_input = input_data
        pressure = np.asarray(input_data.pressure, dtype=float)
        n_layers = pressure.size
        output_data.number_densities_cond = np.column_stack(
            (pressure, pressure + 100.0)
        ).tolist()
        output_data.element_cond_degree = np.column_stack(
            (pressure / (pressure + 1.0), 1.0 / (pressure + 1.0))
        ).tolist()
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
) -> tuple[SimpleNamespace, _FakeFastChem]:
    solver = _FakeFastChem(return_flag=return_flag, layer_flags=layer_flags)
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
        path.write_text(f"fixture: {path.name}\n", encoding="utf-8")
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
        assert np.all(np.isfinite(result.condensate_number_density_cm3))
        assert result.metadata["fastchem_version"]
        assert result.metadata["fastchem_condensate_number_density_unit"] == "cm^-3"
