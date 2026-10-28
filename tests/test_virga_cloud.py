"""Contract tests for the optional Virga eddy-sed cloud adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robert_exoplanets.atmosphere import AtmosphereState
from robert_exoplanets.core import (
    PressureGrid,
    RobertConfigError,
    RobertDataError,
    RobertValidationError,
    SpectralGrid,
)
from robert_exoplanets.forward import virga_cloud
from robert_exoplanets.forward.virga_cloud import VirgaEddySedCloudModel
from robert_exoplanets.opacity import (
    EvaluatedCorrelatedKOpacity,
    PreparedCorrelatedKOpacity,
)
from robert_exoplanets.rt import assemble_gas_optical_depth

_DEFAULT_TEMPERATURE_EDGES = object()


def _pressure_grid() -> PressureGrid:
    edges = np.array([1.0, 0.1, 0.01])
    return PressureGrid(
        edges=edges,
        centers=np.sqrt(edges[:-1] * edges[1:]),
        unit="bar",
    )


def _gas_optical_depth(
    *,
    temperature_edges: np.ndarray | None | object = _DEFAULT_TEMPERATURE_EDGES,
    gravity_m_s2: np.ndarray | float = 15.0,
    mean_molecular_weight: np.ndarray | float = 2.3,
    mean_molecular_weight_unit: str = "amu",
    spectral_grid: SpectralGrid | None = None,
):
    pressure_grid = _pressure_grid()
    grid = spectral_grid or SpectralGrid(
        values=np.array([2.0, 4.0]),
        unit="micron",
        role="native",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=np.array([1050.0, 850.0]),
        temperature_edges=(
            np.array([1100.0, 1000.0, 700.0])
            if temperature_edges is _DEFAULT_TEMPERATURE_EDGES
            else temperature_edges
        ),
        composition={"H2O": np.full(pressure_grid.n_layers, 1.0e-3)},
        mean_molecular_weight=mean_molecular_weight,
        mean_molecular_weight_unit=mean_molecular_weight_unit,
    )
    prepared = PreparedCorrelatedKOpacity(
        provider_name="test-opacity",
        spectral_grid=grid,
        pressure_grid=pressure_grid,
        species=("H2O",),
        g_samples=np.array([0.5]),
        g_weights=np.array([1.0]),
        cache_key="virga-cloud-test",
    )
    opacity = EvaluatedCorrelatedKOpacity(
        prepared=prepared,
        kcoeff=np.zeros((1, pressure_grid.n_layers, grid.size, 1)),
    )
    return assemble_gas_optical_depth(
        atmosphere,
        opacity,
        gravity_m_s2=gravity_m_s2,
    )


def _model(directory: Path, **kwargs) -> VirgaEddySedCloudModel:
    values = {
        "condensibles": ("MgSiO3",),
        "gas_mass_mixing_ratios": {"MgSiO3": 1.0e-3},
        "mie_directory": directory,
        "fsed": 0.5,
        "kz_cm2_s": 1.0e6,
    }
    values.update(kwargs)
    return VirgaEddySedCloudModel(**values)


def _mie_directory(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "mie"
    directory.mkdir()
    for name in names or ("MgSiO3",):
        (directory / f"{name}.mieff").write_text("prepared table\n", encoding="utf-8")
    return directory


class _FakeVirgaAtmosphere:
    def __init__(self, condensibles, **kwargs):
        self.condensibles = tuple(condensibles)
        self.constructor_kwargs = kwargs
        self.gravity_call = None
        self.ptk_call = None

    def gravity(self, **kwargs):
        self.gravity_call = kwargs

    def ptk(self, **kwargs):
        self.ptk_call = kwargs


def _result(
    *,
    wave: np.ndarray | None = None,
    opd_per_layer: np.ndarray | None = None,
    single_scattering: np.ndarray | None = None,
    asymmetry: np.ndarray | None = None,
    condensate_mmr: np.ndarray | None = None,
    pressure: np.ndarray | None = None,
    condensibles=("MgSiO3",),
):
    # Virga returns ascending pressure layers. Wavelengths here are descending.
    return {
        "pressure": np.array([0.055, 0.55]) if pressure is None else pressure,
        "pressure_unit": "bar",
        "wave": np.array([4.0, 2.0, 1.0]) if wave is None else wave,
        "wave_unit": "micron",
        "opd_per_layer": (
            np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
            if opd_per_layer is None
            else opd_per_layer
        ),
        "single_scattering": (
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            if single_scattering is None
            else single_scattering
        ),
        "asymmetry": (
            np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
            if asymmetry is None
            else asymmetry
        ),
        "condensate_mmr": (
            np.array([[2.0e-5], [1.0e-5]])
            if condensate_mmr is None
            else condensate_mmr
        ),
        "condensibles": condensibles,
    }


def _install_fake_virga(monkeypatch, result=None):
    atmosphere_objects = []

    def atmosphere_factory(*args, **kwargs):
        value = _FakeVirgaAtmosphere(*args, **kwargs)
        atmosphere_objects.append(value)
        return value

    def compute(atmosphere, **kwargs):
        assert atmosphere in atmosphere_objects
        return _result() if result is None else result

    fake = SimpleNamespace(Atmosphere=atmosphere_factory, compute=compute)
    monkeypatch.setattr(virga_cloud, "_load_virga", lambda: (fake, "2.0.2"))
    return atmosphere_objects


def test_virga_cloud_uses_pressure_edges_and_maps_layers_and_wavelengths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory)
    fake_atmospheres = _install_fake_virga(monkeypatch)

    (cloud,) = model.evaluate(_gas_optical_depth(), parameters={})

    assert fake_atmospheres[0].condensibles == ("MgSiO3",)
    assert fake_atmospheres[0].constructor_kwargs["gas_mmr"] == {"MgSiO3": 1.0e-3}
    assert fake_atmospheres[0].constructor_kwargs["sig"] == pytest.approx(2.0)
    assert fake_atmospheres[0].gravity_call["gravity"] == pytest.approx(1500.0)
    ptk = fake_atmospheres[0].ptk_call
    np.testing.assert_allclose(ptk["df"]["pressure"], [0.01, 0.1, 1.0])
    np.testing.assert_allclose(ptk["df"]["temperature"], [700.0, 1000.0, 1100.0])
    np.testing.assert_allclose(ptk["df"]["kz"], [1.0e6, 1.0e6, 1.0e6])
    assert ptk["kz_min"] == pytest.approx(1.0e6)

    # ROBERT's first pressure layer is the deepest Virga layer. Virga's
    # descending wavelength rows are also reordered to the requested grid.
    np.testing.assert_allclose(cloud.extinction_tau, [[50.0, 40.0], [20.0, 10.0]])
    np.testing.assert_allclose(
        cloud.single_scattering_albedo,
        [[0.5, 0.4], [0.2, 0.1]],
    )
    np.testing.assert_allclose(cloud.asymmetry_factor, [[0.05, 0.04], [0.02, 0.01]])
    np.testing.assert_array_equal(cloud.pressure_grid.edges, _pressure_grid().edges)
    assert cloud.metadata["virga_version"] == "2.0.2"
    assert cloud.metadata["cloud_condensibles"] == "MgSiO3"
    assert cloud.metadata["cloud_phase_function_closure"] == (
        "henyey_greenstein_from_virga_asymmetry"
    )
    assert cloud.metadata["cloud_virga_do_virtual"] == "false"


def test_virga_cloud_interpolates_wavenumber_targets_without_extrapolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory)
    _install_fake_virga(monkeypatch)
    target_grid = SpectralGrid(
        values=np.array([2500.0, 5000.0]),
        unit="cm^-1",
        role="native",
    )

    (cloud,) = model.evaluate(
        _gas_optical_depth(spectral_grid=target_grid),
        parameters={},
    )

    # 2500 cm^-1 is 4 micron; 5000 cm^-1 is 2 micron.
    np.testing.assert_allclose(cloud.extinction_tau, [[40.0, 50.0], [10.0, 20.0]])


def test_virga_cloud_rejects_spectral_extrapolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory)
    _install_fake_virga(monkeypatch)
    target_grid = SpectralGrid(
        values=np.array([2.0, 5.0]),
        unit="micron",
        role="native",
    )

    with pytest.raises(RobertDataError, match="outside Virga opd_per_layer coverage"):
        model.evaluate(
            _gas_optical_depth(spectral_grid=target_grid),
            parameters={},
        )


def test_virga_cloud_rejects_non_matching_pressure_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory)
    _install_fake_virga(monkeypatch, _result(pressure=np.array([0.05, 0.5])))

    with pytest.raises(RobertDataError, match="pressure-edge intervals"):
        model.evaluate(_gas_optical_depth(), parameters={})


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("opd_per_layer", np.array([[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), "non-negative"),
        ("single_scattering", np.array([[0.1, 0.2, 1.1], [0.4, 0.5, 0.6]]), r"in \[0, 1\]"),
        ("asymmetry", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 1.1]]), r"in \[-1, 1\]"),
        ("condensate_mmr", np.array([[1.0e-5, 2.0e-5], [2.0e-5, 3.0e-5]]), "condensate_mmr must have shape"),
    ],
)
def test_virga_output_is_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    values: np.ndarray,
    message: str,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory)
    _install_fake_virga(monkeypatch, _result(**{field: values}))

    with pytest.raises(RobertDataError, match=message):
        model.evaluate(_gas_optical_depth(), parameters={})


def test_virga_cloud_requires_edge_temperature_and_uniform_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory)
    monkeypatch.setattr(
        virga_cloud,
        "_load_virga",
        lambda: pytest.fail("Virga must not run for an unsupported state"),
    )

    with pytest.raises(RobertValidationError, match="temperature_edges"):
        model.evaluate(
            _gas_optical_depth(temperature_edges=None),
            parameters={},
        )
    with pytest.raises(RobertValidationError, match="constant gravity"):
        model.evaluate(
            _gas_optical_depth(gravity_m_s2=np.array([15.0, 16.0])),
            parameters={},
        )
    with pytest.raises(RobertValidationError, match="constant mean molecular weight"):
        model.evaluate(
            _gas_optical_depth(mean_molecular_weight=np.array([2.3, 2.4])),
            parameters={},
        )


def test_virga_cloud_requires_explicit_gas_mass_mixing_ratios_and_mie_files(
    tmp_path: Path,
) -> None:
    directory = _mie_directory(tmp_path)

    with pytest.raises(RobertValidationError, match="exactly one value"):
        VirgaEddySedCloudModel(
            condensibles=("MgSiO3",),
            gas_mass_mixing_ratios={},
            mie_directory=directory,
            fsed=0.5,
            kz_cm2_s=1.0e6,
        )
    with pytest.raises(RobertConfigError, match="precomputed Mie table"):
        VirgaEddySedCloudModel(
            condensibles=("Fe",),
            gas_mass_mixing_ratios={"Fe": 1.0e-4},
            mie_directory=directory,
            fsed=0.5,
            kz_cm2_s=1.0e6,
        )


def test_virga_kzz_floor_matches_explicit_constant_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory, kz_cm2_s=1.0e4)
    fake_atmospheres = _install_fake_virga(monkeypatch)

    model.evaluate(_gas_optical_depth(), parameters={})

    np.testing.assert_allclose(fake_atmospheres[0].ptk_call["df"]["kz"], 1.0e4)
    assert fake_atmospheres[0].ptk_call["kz_min"] == pytest.approx(1.0e4)


def test_missing_virga_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(name: str):
        raise ModuleNotFoundError("No module named 'virga'", name="virga")

    monkeypatch.setattr(virga_cloud, "import_module", missing_module)

    with pytest.raises(RobertConfigError, match="optional 'virga-exo' package"):
        virga_cloud._load_virga()

