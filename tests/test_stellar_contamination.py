"""Algebra, validation, and response-order tests for stellar contamination."""

from __future__ import annotations

import numpy as np
import pytest

from robert_exoplanets import (
    GaussianLikelihood,
    Observation,
    RetrievalParameter,
    RetrievalParameterSet,
    RetrievalProblem,
    SpectralGrid,
    Spectrum,
    TopHatObservationResponse,
    UniformPrior,
)
from robert_exoplanets.core import RobertValidationError
from robert_exoplanets.stellar import (
    StellarContaminationModel,
    StellarHeterogeneity,
    StellarHeterogeneityDefinition,
    prepare_stellar_contamination_model,
)
from robert_exoplanets.bodies import Star
from robert_exoplanets.retrieval.manifest import (
    build_run_manifest,
    read_run_manifest,
    write_run_manifest,
)


def _stellar(values, grid: SpectralGrid | None = None) -> Spectrum:
    return Spectrum(
        spectral_grid=grid or SpectralGrid.from_array([1.0, 2.0, 3.0], unit="micron"),
        values=np.asarray(values, dtype=float),
        unit="W m^-3 sr^-1",
        observable="stellar_spectral_radiance",
    )


def _region(
    name: str,
    values,
    *,
    kind: str,
    fraction: float | None = None,
    parameter: str | None = None,
) -> StellarHeterogeneity:
    return StellarHeterogeneity(
        name=name,
        kind=kind,
        spectrum=_stellar(values),
        covering_fraction=fraction,
        covering_fraction_parameter=parameter,
    )


def test_homogeneous_disk_is_exact_identity() -> None:
    photosphere = _stellar([3.0, 4.0, 5.0])
    model = StellarContaminationModel(photosphere)

    result = model.evaluate()

    np.testing.assert_array_equal(
        result.disk_integrated_spectrum.values, photosphere.values
    )
    np.testing.assert_array_equal(
        result.transit_chord_spectrum.values, photosphere.values
    )
    np.testing.assert_array_equal(
        result.contamination_factor.values, np.ones(3)
    )


@pytest.mark.parametrize(
    ("region", "fraction"),
    [
        (_region("spot", [1.0, 2.0, 4.0], kind="spot", parameter="f"), 0.2),
        (_region("facula", [6.0, 5.0, 8.0], kind="facula", parameter="f"), 0.1),
    ],
)
def test_one_region_matches_poseidon_rackham_equation(
    region: StellarHeterogeneity,
    fraction: float,
) -> None:
    photosphere = _stellar([4.0, 4.0, 4.0])
    model = StellarContaminationModel(photosphere, (region,))

    result = model.evaluate({"f": fraction})
    analytic = 1.0 / (
        1.0 - fraction * (1.0 - region.spectrum.values / photosphere.values)
    )

    np.testing.assert_allclose(result.contamination_factor.values, analytic, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.disk_integrated_spectrum.values,
        (1.0 - fraction) * photosphere.values
        + fraction * region.spectrum.values,
        rtol=0.0,
        atol=0.0,
    )


def test_spot_facula_mixture_and_explicit_chord_use_general_flux_ratio() -> None:
    photosphere = _stellar([10.0, 10.0, 10.0])
    spot = _region("spot", [4.0, 6.0, 8.0], kind="spot", fraction=0.2)
    facula = _region("facula", [14.0, 13.0, 12.0], kind="facula", fraction=0.1)
    chord = _stellar([9.0, 10.0, 11.0])
    model = StellarContaminationModel(
        photosphere,
        (spot, facula),
        transit_chord_spectrum=chord,
    )

    result = model.evaluate()
    disk = 0.7 * photosphere.values + 0.2 * spot.spectrum.values + 0.1 * facula.spectrum.values

    np.testing.assert_allclose(result.disk_integrated_spectrum.values, disk)
    np.testing.assert_allclose(result.contamination_factor.values, chord.values / disk)
    assert result.covering_fractions == {"spot": 0.2, "facula": 0.1}
    assert result.metadata["stellar_contamination_chord"] == "explicit_spectrum"


def test_zero_fraction_regions_preserve_exact_identity() -> None:
    photosphere = _stellar([3.0, 4.0, 5.0])
    model = StellarContaminationModel(
        photosphere,
        (_region("spot", [1.0, 2.0, 3.0], kind="spot", fraction=0.0),),
    )

    np.testing.assert_array_equal(
        model.evaluate().contamination_factor.values, np.ones(3)
    )


def test_fraction_and_mixture_validation() -> None:
    photosphere = _stellar([3.0, 4.0, 5.0])
    with pytest.raises(RobertValidationError, match="exactly one"):
        StellarHeterogeneity(name="bad", kind="spot", spectrum=photosphere)
    with pytest.raises(RobertValidationError, match=r"\[0, 1\]"):
        _region("bad", [1.0, 2.0, 3.0], kind="spot", fraction=-0.1)
    with pytest.raises(RobertValidationError, match="sum to at most one"):
        StellarContaminationModel(
            photosphere,
            (
                _region("spot", [1.0, 2.0, 3.0], kind="spot", fraction=0.7),
                _region("facula", [5.0, 6.0, 7.0], kind="facula", fraction=0.4),
            ),
        )

    dynamic = StellarContaminationModel(
        photosphere,
        (
            _region("spot", [1.0, 2.0, 3.0], kind="spot", parameter="f_spot"),
            _region("facula", [5.0, 6.0, 7.0], kind="facula", parameter="f_fac"),
        ),
    )
    with pytest.raises(RobertValidationError, match="sum to at most one"):
        dynamic.evaluate({"f_spot": 0.7, "f_fac": 0.4})
    with pytest.raises(RobertValidationError, match="parameter is missing"):
        dynamic.evaluate({"f_spot": 0.1})


def test_spectrum_grid_and_positive_flux_validation() -> None:
    photosphere = _stellar([3.0, 4.0, 5.0])
    bad_grid = SpectralGrid.from_array([1.0, 2.1, 3.0], unit="micron")
    with pytest.raises(RobertValidationError, match="share the photosphere"):
        StellarContaminationModel(
            photosphere,
            (
                StellarHeterogeneity(
                    name="spot",
                    kind="spot",
                    spectrum=_stellar([1.0, 2.0, 3.0], bad_grid),
                    covering_fraction=0.1,
                ),
            ),
        )
    with pytest.raises(RobertValidationError, match="finite and positive"):
        StellarContaminationModel(_stellar([3.0, 0.0, 5.0]))


def test_setup_helper_reuses_stellar_spectrum_model_and_validates_temperature() -> None:
    star = Star(name="G star", effective_temperature_k=5200.0)
    grid = SpectralGrid.from_array([1.0, 2.0, 3.0], unit="micron")
    model = prepare_stellar_contamination_model(
        star,
        grid,
        spectrum_model="blackbody",
        heterogeneities=(
            StellarHeterogeneityDefinition(
                name="spot",
                kind="spot",
                temperature_k=4200.0,
                covering_fraction=0.1,
            ),
            StellarHeterogeneityDefinition(
                name="facula",
                kind="facula",
                temperature_k=6000.0,
                covering_fraction=0.1,
            ),
        ),
    )

    assert model.photosphere_spectrum.metadata["stellar_model"] == "blackbody"
    assert model.heterogeneities[0].spectrum.metadata["effective_temperature_k"] == "4200"
    assert model.manifest_metadata["blackbody_scope"] == (
        "controlled_approximation_not_validation_standard"
    )
    with pytest.raises(RobertValidationError, match="spot temperature"):
        prepare_stellar_contamination_model(
            star,
            grid,
            spectrum_model="blackbody",
            heterogeneities=(
                StellarHeterogeneityDefinition(
                    name="bad spot",
                    kind="spot",
                    temperature_k=5300.0,
                    covering_fraction=0.1,
                ),
            ),
        )


def test_contamination_is_multiplied_before_observational_binning() -> None:
    grid = SpectralGrid.from_array([1.0, 2.0, 3.0], unit="micron", role="rt_native")
    planet_depth = Spectrum(
        spectral_grid=grid,
        values=np.array([0.01, 0.02, 0.04]),
        unit="transit_depth",
        observable="transit_depth",
    )
    # With I_chord / I_disk = [2, 1, 0.5], the native contaminated depth is flat.
    component = StellarContaminationModel(
        photosphere_spectrum=_stellar([2.0, 1.0, 0.5], grid),
        transit_chord_spectrum=_stellar([4.0, 1.0, 0.25], grid),
    )
    observation = Observation.from_arrays(
        [2.0],
        [0.02],
        [1.0e-4],
        wavelength_bin_edges=[1.0, 3.0],
        wavelength_unit="micron",
        flux_unit="transit_depth",
        observable="transit_depth",
    )
    response = TopHatObservationResponse().prepare(observation)

    correctly_binned = response.observe(component.apply(planet_depth))
    binned_planet = response.observe(planet_depth).values[0]
    binned_factor = np.trapezoid([2.0, 1.0, 0.5], [1.0, 2.0, 3.0]) / 2.0

    np.testing.assert_allclose(correctly_binned.values, [0.02], rtol=0.0, atol=1.0e-15)
    assert not np.isclose(correctly_binned.values[0], binned_planet * binned_factor)
    assert correctly_binned.metadata["response"] == "top-hat-observation-bins"


def test_spot_fraction_closes_synthetic_likelihood() -> None:
    grid = SpectralGrid.from_array([1.0, 2.0, 3.0], unit="micron")
    planet_depth = Spectrum(
        spectral_grid=grid,
        values=np.array([0.01, 0.0102, 0.0101]),
        unit="transit_depth",
        observable="transit_depth",
    )
    component = StellarContaminationModel(
        _stellar([4.0, 4.0, 4.0], grid),
        (
            StellarHeterogeneity(
                name="spot",
                kind="spot",
                spectrum=_stellar([1.0, 2.0, 3.0], grid),
                covering_fraction_parameter="f_spot",
            ),
        ),
    )
    injected = component.apply(planet_depth, {"f_spot": 0.2})
    observation = Observation.from_arrays(
        grid.values,
        injected.values,
        np.full(3, 1.0e-5),
        flux_unit="transit_depth",
        observable="transit_depth",
    )
    problem = RetrievalProblem(
        name="TSLE likelihood closure",
        observation=observation,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("f_spot", UniformPrior(0.0, 0.5)),)
        ),
        forward_model=lambda parameters: component.apply(planet_depth, parameters),
        likelihood=GaussianLikelihood(include_normalization=False),
    )

    injected_loglike = problem.log_likelihood_from_vector([0.2])
    homogeneous_loglike = problem.log_likelihood_from_vector([0.0])

    assert injected_loglike == pytest.approx(0.0, abs=1.0e-20)
    assert injected_loglike > homogeneous_loglike


def test_stellar_contamination_provenance_round_trips_through_manifest(
    tmp_path,
) -> None:
    grid = SpectralGrid.from_array([1.0, 2.0], unit="micron")
    component = StellarContaminationModel(
        _stellar([4.0, 4.0], grid),
        (
            StellarHeterogeneity(
                name="spot",
                kind="spot",
                spectrum=_stellar([1.0, 2.0], grid),
                covering_fraction_parameter="f_spot",
            ),
        ),
    )
    planet_depth = Spectrum(
        spectral_grid=grid,
        values=np.array([0.01, 0.01]),
        unit="transit_depth",
        observable="transit_depth",
    )
    observation = Observation.from_arrays(
        grid.values,
        planet_depth.values,
        [1.0e-4, 1.0e-4],
        flux_unit="transit_depth",
        observable="transit_depth",
    )
    problem = RetrievalProblem(
        name="TSLE manifest",
        observation=observation,
        parameters=RetrievalParameterSet(
            (RetrievalParameter("f_spot", UniformPrior(0.0, 0.5)),)
        ),
        forward_model=lambda parameters: component.apply(planet_depth, parameters),
        metadata=dict(component.manifest_metadata),
    )
    manifest = build_run_manifest(
        problem,
        method="test",
        settings={"calls": 1},
        random_seed=7,
    )

    path = write_run_manifest(manifest, tmp_path)
    restored = read_run_manifest(path)

    assert restored.problem_metadata["stellar_contamination"] == "enabled"
    assert restored.problem_metadata["stellar_contamination_region_0_kind"] == "spot"
    assert restored.problem_metadata["stellar_contamination_required_parameters"] == "f_spot"
    assert restored.to_mapping()["problem_metadata"] == dict(manifest.problem_metadata)
