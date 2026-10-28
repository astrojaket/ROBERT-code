"""Contract tests for the optional Virga eddy-sed cloud adapter."""

from __future__ import annotations

import os
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
from robert_exoplanets.rt import OpticalConstantsCatalog, assemble_gas_optical_depth

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
    pressure_grid: PressureGrid | None = None,
    temperature_edges: np.ndarray | None | object = _DEFAULT_TEMPERATURE_EDGES,
    gravity_m_s2: np.ndarray | float = 15.0,
    mean_molecular_weight: np.ndarray | float = 2.3,
    mean_molecular_weight_unit: str = "amu",
    spectral_grid: SpectralGrid | None = None,
):
    pressure_grid = _pressure_grid() if pressure_grid is None else pressure_grid
    if pressure_grid.n_layers == 2:
        layer_temperature = np.array([1050.0, 850.0])
        default_temperature_edges = np.array([1100.0, 1000.0, 700.0])
        if pressure_grid.orientation == "increasing":
            layer_temperature = layer_temperature[::-1]
            default_temperature_edges = default_temperature_edges[::-1]
    else:
        layer_temperature = np.linspace(1050.0, 850.0, pressure_grid.n_layers)
        if pressure_grid.orientation == "increasing":
            layer_temperature = layer_temperature[::-1]
        default_temperature_edges = np.linspace(
            1100.0, 700.0, pressure_grid.n_layers + 1
        )
        if pressure_grid.orientation == "increasing":
            default_temperature_edges = default_temperature_edges[::-1]
    grid = spectral_grid or SpectralGrid(
        values=np.array([2.0, 4.0]),
        unit="micron",
        role="native",
    )
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=layer_temperature,
        temperature_edges=(
            default_temperature_edges
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
    droplet_eff_r: np.ndarray | None = None,
    r_units: str = "micron",
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
            np.array([[2.0e-5], [1.0e-5]]) if condensate_mmr is None else condensate_mmr
        ),
        "droplet_eff_r": (
            np.tile(np.array([[2.0], [1.0]]), (1, len(condensibles)))
            if droplet_eff_r is None
            else droplet_eff_r
        ),
        "r_units": r_units,
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

    molecular_weights = {"MgSiO3": 100.4, "KCl": 74.5}

    def gas_property(species):
        gas_molecular_weight = molecular_weights[species]

        def evaluate(mean_molecular_weight, *, mh=1.0, gas_mmr=None):
            if gas_mmr is None:
                raise AssertionError("ROBERT must pass an explicit Virga gas_mmr")
            return (
                gas_molecular_weight,
                float(gas_mmr) * gas_molecular_weight / mean_molecular_weight,
                1.0,
            )

        return evaluate

    gas_properties = SimpleNamespace(
        **{name: gas_property(name) for name in molecular_weights}
    )
    fake = SimpleNamespace(
        Atmosphere=atmosphere_factory,
        compute=compute,
        gas_properties=gas_properties,
    )
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
    expected_mole_fraction = 1.0e-3 * 2.3 / 100.4
    assert fake_atmospheres[0].constructor_kwargs["gas_mmr"] == {
        "MgSiO3": pytest.approx(expected_mole_fraction)
    }
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
    assert cloud.metadata["cloud_gas_mass_mixing_ratios_g_g"] == "MgSiO3:0.001"
    assert cloud.metadata["cloud_virga_gas_mole_fractions"] == (
        f"MgSiO3:{expected_mole_fraction:.17g}"
    )
    assert (
        cloud.metadata["cloud_virga_gas_molecular_weights_amu"]
        == "MgSiO3:100.40000000000001"
    )
    assert (
        cloud.metadata["cloud_virga_gas_mean_molecular_weight_amu_used"]
        == "2.2999999999999998"
    )
    assert cloud.metadata["cloud_virga_gas_mass_fraction_roundtrip_validated"] == "true"


@pytest.mark.parametrize(
    ("orientation", "expected_condensate", "expected_radius"),
    [
        ("decreasing", [1.0e-5, 2.0e-5], [1.0, 2.0]),
        ("increasing", [2.0e-5, 1.0e-5], [2.0, 1.0]),
    ],
)
def test_virga_cloud_profiles_use_validated_outputs_and_pressure_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    orientation: str,
    expected_condensate: list[float],
    expected_radius: list[float],
) -> None:
    directory = _mie_directory(tmp_path)
    pressure_grid = (
        _pressure_grid()
        if orientation == "decreasing"
        else PressureGrid(
            edges=np.array([0.01, 0.1, 1.0]),
            centers=np.sqrt(np.array([0.01, 0.1]) * np.array([0.1, 1.0])),
        )
    )
    atmosphere = _gas_optical_depth(pressure_grid=pressure_grid).atmosphere
    fake_atmospheres = _install_fake_virga(monkeypatch)
    model = _model(directory)

    profiles = model.compute_cloud_profiles(
        atmosphere,
        gravity_m_s2=np.full(pressure_grid.n_layers, 15.0),
        parameters={},
    )

    assert isinstance(profiles, virga_cloud.VirgaCloudProfileResult)
    assert profiles.condensibles == ("MgSiO3",)
    assert profiles.pressure_grid is pressure_grid
    assert profiles.pressure_orientation == orientation
    assert profiles.condensate_mass_mixing_ratio_unit == "kg/kg"
    assert profiles.effective_particle_radius_unit == "micron"
    np.testing.assert_allclose(
        profiles.condensate_mass_mixing_ratio_kg_kg[:, 0], expected_condensate
    )
    np.testing.assert_allclose(
        profiles.effective_particle_radius_micron[:, 0], expected_radius
    )
    assert not profiles.condensate_mass_mixing_ratio_kg_kg.flags.writeable
    assert not profiles.effective_particle_radius_micron.flags.writeable
    assert profiles.metadata["virga_profile_condensate_source_unit"] == "g/g"
    assert profiles.metadata["virga_profile_radius_field"] == "droplet_eff_r"
    assert profiles.metadata["virga_profile_pressure_orientation"] == orientation
    assert (
        profiles.metadata["virga_profile_layer_reversal_applied"]
        == str(orientation == "decreasing").lower()
    )
    assert fake_atmospheres[0].gravity_call["gravity"] == pytest.approx(1500.0)


@pytest.mark.parametrize(
    ("result_edits", "message"),
    [
        ({"remove": "droplet_eff_r"}, "droplet_eff_r"),
        ({"r_units": "cm"}, "must use microns"),
        ({"droplet_eff_r": np.array([[1.0], [1.0], [1.0]])}, "shape"),
        ({"droplet_eff_r": np.array([[0.0], [1.0]])}, "must be positive"),
        ({"droplet_eff_r": np.array([[np.nan], [1.0]])}, "finite non-negative"),
    ],
)
def test_virga_cloud_profile_rejects_invalid_radius_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_edits: dict[str, object],
    message: str,
) -> None:
    directory = _mie_directory(tmp_path)
    result = _result()
    if "remove" in result_edits:
        result.pop(str(result_edits["remove"]))
    else:
        result.update(result_edits)
    _install_fake_virga(monkeypatch, result=result)
    model = _model(directory)

    with pytest.raises(RobertDataError, match=message):
        model.compute_cloud_profiles(
            _gas_optical_depth().atmosphere,
            gravity_m_s2=15.0,
            parameters={},
        )


def test_virga_cloud_profile_runs_fresh_solve_for_each_parameter_draw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(
        directory,
        fsed=None,
        fsed_parameter="cloud_fsed",
        kz_cm2_s=None,
        log10_kz_cm2_s_parameter="log10_kzz",
    )
    fake_atmospheres = _install_fake_virga(monkeypatch)
    virga, _ = virga_cloud._load_virga()
    compute_count = 0

    def compute(atmosphere, **kwargs):
        nonlocal compute_count
        assert atmosphere in fake_atmospheres
        compute_count += 1
        result = _result()
        result["condensate_mmr"] = np.full((2, 1), compute_count * 1.0e-5, dtype=float)
        result["droplet_eff_r"] = np.full((2, 1), compute_count, dtype=float)
        return result

    virga.compute = compute
    atmosphere = _gas_optical_depth().atmosphere

    first = model.compute_cloud_profiles(
        atmosphere,
        15.0,
        {"cloud_fsed": 0.5, "log10_kzz": 8.0},
    )
    second = model.compute_cloud_profiles(
        atmosphere,
        15.0,
        {"cloud_fsed": 2.0, "log10_kzz": 10.0},
    )

    assert len(fake_atmospheres) == 2
    assert fake_atmospheres[0] is not fake_atmospheres[1]
    assert fake_atmospheres[0].constructor_kwargs["fsed"] == pytest.approx(0.5)
    assert fake_atmospheres[1].constructor_kwargs["fsed"] == pytest.approx(2.0)
    assert fake_atmospheres[0].ptk_call["kz_min"] == pytest.approx(1.0e8)
    assert fake_atmospheres[1].ptk_call["kz_min"] == pytest.approx(1.0e10)
    np.testing.assert_allclose(first.condensate_mass_mixing_ratio_kg_kg, 1.0e-5)
    np.testing.assert_allclose(second.condensate_mass_mixing_ratio_kg_kg, 2.0e-5)
    np.testing.assert_allclose(first.effective_particle_radius_micron, 1.0)
    np.testing.assert_allclose(second.effective_particle_radius_micron, 2.0)


def test_virga_cloud_converts_each_condensible_mass_fraction_with_its_molecular_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path, "MgSiO3", "KCl")
    result = _result(
        condensibles=("MgSiO3", "KCl"),
        condensate_mmr=np.array([[2.0e-5, 3.0e-5], [1.0e-5, 2.0e-5]]),
    )
    fake_atmospheres = _install_fake_virga(monkeypatch, result=result)
    model = _model(
        directory,
        condensibles=("MgSiO3", "KCl"),
        gas_mass_mixing_ratios={"MgSiO3": 1.0e-3, "KCl": 2.0e-4},
    )

    (cloud,) = model.evaluate(_gas_optical_depth(), parameters={})

    np.testing.assert_allclose(
        [
            fake_atmospheres[0].constructor_kwargs["gas_mmr"]["MgSiO3"],
            fake_atmospheres[0].constructor_kwargs["gas_mmr"]["KCl"],
        ],
        [1.0e-3 * 2.3 / 100.4, 2.0e-4 * 2.3 / 74.5],
        rtol=1.0e-14,
    )
    assert cloud.metadata["cloud_virga_gas_mass_fraction_roundtrip_validated"] == "true"


def test_virga_cloud_rejects_gas_property_conversion_that_does_not_recover_mass_fraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    _install_fake_virga(monkeypatch)
    model = _model(directory)

    virga, _ = virga_cloud._load_virga()
    virga.gas_properties.MgSiO3 = lambda mmw, *, mh, gas_mmr: (100.4, gas_mmr, 3.192)

    with pytest.raises(
        RobertDataError, match="did not recover the configured mass mixing ratio"
    ):
        model.evaluate(_gas_optical_depth(), parameters={})


def test_virga_cloud_resolves_retrieved_fsed_and_log10_kzz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(
        directory,
        fsed=None,
        fsed_parameter="cloud_fsed",
        kz_cm2_s=None,
        log10_kz_cm2_s_parameter="cloud_log10_kzz",
    )
    fake_atmospheres = _install_fake_virga(monkeypatch)

    (cloud,) = model.evaluate(
        _gas_optical_depth(),
        parameters={"cloud_fsed": 0.6, "cloud_log10_kzz": 8.0},
    )

    assert model.required_parameters == ("cloud_fsed", "cloud_log10_kzz")
    assert fake_atmospheres[0].constructor_kwargs["fsed"] == pytest.approx(0.6)
    assert fake_atmospheres[0].ptk_call["kz_min"] == pytest.approx(1.0e8)
    np.testing.assert_allclose(fake_atmospheres[0].ptk_call["df"]["kz"], 1.0e8)
    assert cloud.metadata["cloud_fsed"] == "retrieved_parameter:cloud_fsed"
    assert cloud.metadata["cloud_kz_cm2_s"] == ("retrieved_parameter:cloud_log10_kzz")
    assert cloud.metadata["cloud_fsed_used"] == "0.59999999999999998"
    assert cloud.metadata["cloud_kz_cm2_s_used"] == "100000000"


def test_virga_cloud_requires_retrieved_controls_and_valid_runtime_values(
    tmp_path: Path,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(
        directory,
        fsed=None,
        fsed_parameter="cloud_fsed",
        kz_cm2_s=None,
        log10_kz_cm2_s_parameter="cloud_log10_kzz",
    )

    with pytest.raises(RobertValidationError, match="fsed parameter is missing"):
        model.evaluate(_gas_optical_depth(), parameters={})
    with pytest.raises(RobertValidationError, match="finite and positive"):
        model.evaluate(
            _gas_optical_depth(),
            parameters={"cloud_fsed": 0.0, "cloud_log10_kzz": 8.0},
        )
    with pytest.raises(RobertValidationError, match=r"lie in \[-300, 300\]"):
        model.evaluate(
            _gas_optical_depth(),
            parameters={"cloud_fsed": 0.6, "cloud_log10_kzz": 301.0},
        )


def test_virga_model_rejects_control_source_and_name_collisions(
    tmp_path: Path,
) -> None:
    directory = _mie_directory(tmp_path)

    with pytest.raises(RobertValidationError, match="exactly one of fixed fsed"):
        _model(directory, fsed=None)
    with pytest.raises(RobertValidationError, match="exactly one of fixed fsed"):
        _model(directory, fsed_parameter="cloud_fsed")
    with pytest.raises(
        RobertValidationError, match="parameter names must be different"
    ):
        _model(
            directory,
            fsed=None,
            fsed_parameter="same_parameter",
            kz_cm2_s=None,
            log10_kz_cm2_s_parameter="same_parameter",
        )


def test_virga_model_preserves_legacy_positional_control_order(
    tmp_path: Path,
) -> None:
    directory = _mie_directory(tmp_path)

    model = VirgaEddySedCloudModel(
        ("MgSiO3",),
        {"MgSiO3": 1.0e-3},
        directory,
        0.5,
        1.0e6,
        1.5,
        2.0,
        0.0,
        "sh4",
        {"source": "legacy-positional"},
    )

    assert model.fsed == pytest.approx(0.5)
    assert model.kz_cm2_s == pytest.approx(1.0e6)
    assert model.metallicity == pytest.approx(1.5)
    assert model.geometric_stddev == pytest.approx(2.0)
    assert model.metadata["source"] == "legacy-positional"


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
        (
            "opd_per_layer",
            np.array([[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            "non-negative",
        ),
        (
            "single_scattering",
            np.array([[0.1, 0.2, 1.1], [0.4, 0.5, 0.6]]),
            r"in \[0, 1\]",
        ),
        ("asymmetry", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 1.1]]), r"in \[-1, 1\]"),
        (
            "condensate_mmr",
            np.array([[1.0e-5, 2.0e-5], [2.0e-5, 3.0e-5]]),
            "condensate_mmr must have shape",
        ),
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


def test_virga_cloud_scalar_mmw_override_is_limited_to_cloud_calculation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _mie_directory(tmp_path)
    model = _model(directory, mean_molecular_weight_amu=2.35)
    fake_atmospheres = _install_fake_virga(monkeypatch)
    gas_optical_depth = _gas_optical_depth(mean_molecular_weight=np.array([2.3, 2.5]))

    (cloud,) = model.evaluate(gas_optical_depth, parameters={})

    assert fake_atmospheres[0].constructor_kwargs["mmw"] == pytest.approx(2.35)
    np.testing.assert_allclose(
        gas_optical_depth.atmosphere.mean_molecular_weight,
        [2.3, 2.5],
    )
    assert cloud.metadata["cloud_mean_molecular_weight_policy_used"] == (
        "cloud_only_scalar_override"
    )
    assert float(
        cloud.metadata["cloud_mean_molecular_weight_profile_min_amu"]
    ) == pytest.approx(2.3)
    assert float(
        cloud.metadata["cloud_mean_molecular_weight_profile_max_amu"]
    ) == pytest.approx(2.5)
    assert float(
        cloud.metadata["cloud_mean_molecular_weight_amu_used"]
    ) == pytest.approx(2.35)
    assert float(
        cloud.metadata["cloud_mean_molecular_weight_override_amu"]
    ) == pytest.approx(2.35)
    assert cloud.metadata["cloud_mean_molecular_weight_is_approximation"] == "true"
    assert cloud.metadata["cloud_mean_molecular_weight_approximation_scope"] == (
        "virga_cloud_microphysics_only; "
        "ROBERT_gas_and_rt_use_original_atmosphere_profile"
    )


@pytest.mark.parametrize("value", [0.0, -2.3, float("nan"), float("inf")])
def test_virga_cloud_scalar_mmw_override_must_be_finite_and_positive(
    tmp_path: Path,
    value: float,
) -> None:
    with pytest.raises(RobertValidationError, match="mean_molecular_weight_amu"):
        _model(_mie_directory(tmp_path), mean_molecular_weight_amu=value)


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


@pytest.mark.skipif(
    os.environ.get("ROBERT_RUN_VIRGA_TESTS") != "1",
    reason="set ROBERT_RUN_VIRGA_TESTS=1 to run the optional Virga profile test",
)
def test_real_virga_cloud_profiles_are_finite_and_pressure_aligned(
    tmp_path: Path,
) -> None:
    virga = pytest.importorskip(
        "virga.justdoit",
        reason="install virga-exo 2.x to run the Virga profile integration test",
    )
    repository = Path(__file__).resolve().parents[1]
    refractive_index = OpticalConstantsCatalog(
        repository / "data" / "optical_constants" / "exo_skryer"
    ).load("MgSiO3")
    refrind_directory = tmp_path / "refrind"
    mie_directory = tmp_path / "mie"
    refrind_directory.mkdir()
    mie_directory.mkdir()
    np.savetxt(
        refrind_directory / "MgSiO3.refrind",
        np.column_stack(
            (
                np.arange(1, refractive_index.wavelength_micron.size + 1),
                refractive_index.wavelength_micron,
                refractive_index.real_index,
                refractive_index.imaginary_index,
            )
        ),
        fmt=("%d", "%.10g", "%.10g", "%.10g"),
    )
    virga.calc_mie_db(
        "MgSiO3",
        virga_dir=str(refrind_directory),
        dir_out=str(mie_directory),
        rmin=1.0e-8,
        rmax=1.0e-2,
        nradii=20,
        logspace=True,
        aggregates=False,
        fort_calc_mie=False,
    )

    n_layers = 16
    edges = np.logspace(1.0, -5.0, n_layers + 1)
    pressure_grid = PressureGrid(
        edges=edges,
        centers=np.sqrt(edges[:-1] * edges[1:]),
        unit="bar",
    )
    temperature_edges = np.linspace(1800.0, 850.0, n_layers + 1)
    atmosphere = AtmosphereState(
        pressure_grid=pressure_grid,
        temperature=0.5 * (temperature_edges[:-1] + temperature_edges[1:]),
        temperature_edges=temperature_edges,
        composition={"H2O": np.full(n_layers, 1.0e-3)},
        mean_molecular_weight=2.3,
    )
    model = _model(
        mie_directory,
        gas_mass_mixing_ratios={"MgSiO3": 2.75e-3},
        kz_cm2_s=1.0e8,
    )

    profiles = model.compute_cloud_profiles(atmosphere, 15.0, {})

    condensate = profiles.condensate_mass_mixing_ratio_kg_kg
    radius = profiles.effective_particle_radius_micron
    assert profiles.pressure_orientation == "decreasing"
    assert condensate.shape == (n_layers, 1)
    assert radius.shape == (n_layers, 1)
    assert np.all(np.isfinite(condensate)) and np.all(condensate >= 0.0)
    assert np.all(np.isfinite(radius)) and np.all(radius >= 0.0)
    assert np.max(condensate) > 0.0
    assert np.all(radius[condensate > 0.0] > 0.0)
    assert profiles.metadata["virga_profile_condensate_source_unit"] == "g/g"
    assert profiles.metadata["virga_profile_radius_unit"] == "micron"
