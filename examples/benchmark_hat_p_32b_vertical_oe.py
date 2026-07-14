"""Run a 100-bin, pressure-resolved HAT-P-32b emission OE benchmark.

This is an injection/recovery experiment: the bundled HAT-P-32b median T-P
and VMR profiles generate a cloud-free emission spectrum, each synthetic bin
is assigned a 30 ppm uncertainty, and temperature is retrieved at every
atmospheric level with a NEMESIS-style pressure-correlated prior.

The script also writes plain-text atmosphere and spectrum tables that can be
translated into a legacy NEMESIS/archNEMESIS case without making the two codes
share in-memory objects or opacity implementations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/robert-numba-cache")

from robert_exoplanets import (
    CompositionMeanMolecularWeight,
    ExoKOpacitySource,
    ExoKTableBinning,
    LayerByLayerStateVector,
    Observation,
    ParameterizedClearSkyEmissionFactoryConfig,
    ParameterizedClearSkyEmissionModelConfig,
    Planet,
    PressureGrid,
    RetrievalProblem,
    SpectralGrid,
    SplineFreeChemistry,
    SplineTemperatureProfile,
    Star,
    VerticalProfileParameterization,
    build_parameterized_clear_sky_emission_model,
    load_nemesispy_cia_table,
    run_optimal_estimation,
)


R_SUN_M = 6.957e8
R_JUP_M = 7.1492e7
M_JUP_KG = 1.898e27
PLANET_RADIUS_M = 1.98 * R_JUP_M
PLANET_MASS_KG = 0.68 * M_JUP_KG
STAR_RADIUS_M = 1.32 * R_SUN_M
STAR_TEMPERATURE_K = 6001.0
SPECIES = ("H2O", "CO2", "CO", "CH4", "NH3", "HCN")
BUNDLE_ROOT = Path(__file__).resolve().parent / "Depreciated_Benchmarks" / "HAT_P_32b" / "data" / "hat_p_32b"
DEFAULT_KTA_DIR = Path.home() / "Dropbox" / "PostDoc4" / "Emission_Example" / "HAT-P-32b" / "kta_temp"


@dataclass(frozen=True)
class PreparedBenchmark:
    problem: RetrievalProblem
    state: LayerByLayerStateVector
    truth_temperature_K: np.ndarray
    prior_temperature_K: np.ndarray
    truth_spectrum: np.ndarray
    pressure_bar: np.ndarray
    vmr: np.ndarray
    gas_names: tuple[str, ...]
    wavelength_micron: np.ndarray
    bin_edges_micron: np.ndarray
    setup_seconds: float
    forward_seconds: float


def prepare_benchmark(
    *,
    n_bins: int = 100,
    n_layers: int = 100,
    uncertainty_ppm: float = 30.0,
    opacity_g_points: int = 30,
    correlation_length: float = 1.5,
    prior_sigma_K: float = 300.0,
    kta_dir: str | Path = DEFAULT_KTA_DIR,
) -> PreparedBenchmark:
    """Prepare the HAT-P-32b injection and layer-temperature OE problem."""

    started = time.perf_counter()
    reference_path = BUNDLE_ROOT / "reference" / "quench_study_emission_TP_VMR_band.npz"
    with np.load(reference_path, allow_pickle=False) as archive:
        source_pressure = np.asarray(archive["pressure_bar"], dtype=float)
        source_temperature = np.asarray(archive["T_med"], dtype=float)
        source_vmr = np.asarray(archive["VMR_med"], dtype=float)
        source_species = tuple(str(item) for item in archive["gas_names"])
    if set(source_species) != set(SPECIES):
        raise RuntimeError("bundled HAT-P-32b gas names no longer match the benchmark")

    pressure_grid = PressureGrid.from_log_centers(
        float(source_pressure[0]),
        float(source_pressure[-1]),
        n_layers=n_layers,
        unit="bar",
        name="HAT-P-32b vertical OE",
    )
    log_source = np.log(source_pressure[::-1])
    log_target = np.log(pressure_grid.centers[::-1])
    truth_temperature = np.interp(log_target, log_source, source_temperature[::-1])[::-1]
    vmr = np.column_stack(
        [
            np.exp(
                np.interp(
                    log_target,
                    log_source,
                    np.log(np.maximum(source_vmr[::-1, source_species.index(species)], 1.0e-300)),
                )
            )[::-1]
            for species in SPECIES
        ]
    )

    temperature_profile = VerticalProfileParameterization.temperature(
        pressure=pressure_grid.centers,
        prior_temperature=_temperature_prior(truth_temperature),
        prior_sigma_K=prior_sigma_K,
        correlation_length=correlation_length,
        pressure_unit="bar",
        bound_sigma=5.0,
    )
    state = LayerByLayerStateVector((temperature_profile,))
    temperature_model = SplineTemperatureProfile(
        knot_pressure=pressure_grid.centers,
        parameter_names=temperature_profile.parameter_names,
        pressure_unit="bar",
        extrapolation="clip",
        name="HAT-P-32b layer temperature",
    )

    vmr_blocks = tuple(
        VerticalProfileParameterization.positive_profile(
            name=f"fixed_{species}",
            pressure=pressure_grid.centers,
            prior_profile=vmr[:, index],
            prior_fractional_uncertainty=1.0,
            correlation_length=correlation_length,
            kind="vmr",
            pressure_unit="bar",
        )
        for index, species in enumerate(SPECIES)
    )
    chemistry = SplineFreeChemistry(
        active_species=SPECIES,
        knot_pressure=pressure_grid.centers,
        parameter_names={species: block.parameter_names for species, block in zip(SPECIES, vmr_blocks, strict=True)},
        pressure_unit="bar",
        parameter_mode="ln",
        metadata={"source": "bundled HAT-P-32b median VMR profile"},
    )
    fixed_chemistry = {
        name: float(value)
        for block in vmr_blocks
        for name, value in zip(block.parameter_names, block.prior_state, strict=True)
    }

    bin_edges = np.geomspace(2.9, 5.15, n_bins + 1)
    wavelength = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    spectral_grid = SpectralGrid(
        values=wavelength,
        bin_edges=bin_edges,
        unit="micron",
        role="observed",
        name="HAT-P-32b synthetic 100-bin spectrum",
    )
    kta_path = Path(kta_dir).expanduser()
    if not kta_path.is_dir():
        raise FileNotFoundError(
            "The exactly-100-bin benchmark must be recompressed from the original HAT-P-32b "
            f".kta files. Directory not found: {kta_path}. Pass --kta-dir."
        )
    opacity_source = ExoKOpacitySource(
        species=SPECIES,
        directory=kta_path,
        filename_pattern="*_emission_R1000.kta",
        name="HAT-P-32b ExoMol R1000 correlated-k",
        interpolation="log_pressure_temperature_log_k_clip",
        nonfinite_policy="floor",
    )
    config = ParameterizedClearSkyEmissionFactoryConfig(
        planet=Planet("HAT-P-32b", radius_m=PLANET_RADIUS_M, mass_kg=PLANET_MASS_KG),
        star=Star(
            "HAT-P-32",
            radius_m=STAR_RADIUS_M,
            effective_temperature_k=STAR_TEMPERATURE_K,
        ),
        temperature_profile=temperature_model,
        chemistry_model=chemistry,
        mean_molecular_weight_model=CompositionMeanMolecularWeight(),
        pressure_grid=pressure_grid,
        cia_table=load_nemesispy_cia_table(),
        opacity_source=opacity_source,
        opacity_binning=ExoKTableBinning(num=opacity_g_points),
        model=ParameterizedClearSkyEmissionModelConfig(
            opacity_species=SPECIES,
            include_rayleigh=False,
            gas_combination="random_overlap",
            thermal_integration_backend="auto",
            metadata={
                "benchmark": "HAT-P-32b 100-bin layer-temperature OE",
                "clouds": "none",
            },
        ),
    )
    model = build_parameterized_clear_sky_emission_model(config, spectral_grid=spectral_grid)
    setup_seconds = time.perf_counter() - started

    def evaluate_temperature(parameters):
        return model({**fixed_chemistry, **parameters})

    truth_parameters = dict(zip(temperature_profile.parameter_names, truth_temperature, strict=True))
    forward_started = time.perf_counter()
    truth_spectrum = np.asarray(evaluate_temperature(truth_parameters).values, dtype=float)
    forward_seconds = time.perf_counter() - forward_started
    observation = Observation(
        wavelength=wavelength,
        wavelength_bin_edges=bin_edges,
        flux=truth_spectrum,
        uncertainty=np.full(n_bins, uncertainty_ppm * 1.0e-6),
        wavelength_unit="micron",
        flux_unit="eclipse_depth",
        observable="eclipse_depth",
        instrument="synthetic JWST emission",
        metadata={"noise_model": f"independent {uncertainty_ppm:g} ppm; zero-noise injection"},
    )
    problem = RetrievalProblem(
        name="hat-p-32b-layer-temperature-oe",
        observation=observation,
        parameters=state.retrieval_parameters,
        forward_model=evaluate_temperature,
        metadata={
            **dict(model.manifest_metadata),
            "truth": "bundled median HAT-P-32b T-P and VMR profiles",
            "state_parameterization": "temperature at every pressure level",
            "prior_correlation": "exp(-abs(ln(pi/pj))/Lc)",
        },
        opacity_identifiers=model.opacity_identifiers,
    )
    return PreparedBenchmark(
        problem=problem,
        state=state,
        truth_temperature_K=truth_temperature,
        prior_temperature_K=np.asarray(temperature_profile.prior_state),
        truth_spectrum=truth_spectrum,
        pressure_bar=np.asarray(pressure_grid.centers),
        vmr=vmr,
        gas_names=SPECIES,
        wavelength_micron=wavelength,
        bin_edges_micron=bin_edges,
        setup_seconds=setup_seconds,
        forward_seconds=forward_seconds,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.speed_repeats < 1:
        raise ValueError("--speed-repeats must be at least one")
    prepared = prepare_benchmark(
        n_bins=args.bins,
        n_layers=args.layers,
        uncertainty_ppm=args.uncertainty_ppm,
        opacity_g_points=args.opacity_g_points,
        correlation_length=args.correlation_length,
        prior_sigma_K=args.prior_sigma_K,
        kta_dir=args.kta_dir,
    )
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    _write_interchange_tables(output, prepared, args.uncertainty_ppm)

    oe_started = time.perf_counter()
    result = None
    if not args.prepare_only:
        forward_model_error = None if args.forward_model_error_ppm == 0.0 else args.forward_model_error_ppm * 1.0e-6
        result = run_optimal_estimation(
            prepared.problem,
            initial_state=prepared.state.prior_state,
            prior_state=prepared.state.prior_state,
            prior_covariance=prepared.state.prior_covariance,
            max_iterations=args.max_iterations,
            finite_difference_fraction=args.finite_difference_fraction,
            finite_difference_scheme=args.finite_difference_scheme,
            marquardt_lambda=args.marquardt_lambda,
            forward_model_error=forward_model_error,
        )
    oe_seconds = time.perf_counter() - oe_started if result is not None else 0.0
    retrieved = prepared.prior_temperature_K if result is None else result.state_vector
    prior_mapping = prepared.state.retrieval_parameters.vector_to_mapping(prepared.state.prior_state)
    speed_samples: list[float] = []
    prior_spectrum = None
    for _ in range(args.speed_repeats):
        speed_started = time.perf_counter()
        prior_spectrum = np.asarray(
            prepared.problem.forward_model(prior_mapping).values,
            dtype=float,
        )
        speed_samples.append(time.perf_counter() - speed_started)
    retrieved_spectrum = (
        prior_spectrum
        if result is None
        else np.asarray(
            prepared.problem.forward_model(result.best_fit_parameters).values,
            dtype=float,
        )
    )
    uncertainty = args.uncertainty_ppm * 1.0e-6
    normalized_residual = (prepared.truth_spectrum - retrieved_spectrum) / uncertainty
    report = {
        "benchmark": "HAT-P-32b cloud-free 100-bin layer-temperature OE",
        "n_bins": args.bins,
        "n_layers": args.layers,
        "uncertainty_ppm": args.uncertainty_ppm,
        "forward_model_error_ppm": args.forward_model_error_ppm,
        "correlation_length_scale_heights": args.correlation_length,
        "prior_sigma_K": args.prior_sigma_K,
        "setup_seconds": prepared.setup_seconds,
        "single_forward_seconds": prepared.forward_seconds,
        "steady_state_forward_median_seconds": float(np.median(speed_samples)),
        "steady_state_forward_calls_per_second": float(1.0 / np.median(speed_samples)),
        "oe_seconds": oe_seconds,
        "oe_converged": None if result is None else result.converged,
        "oe_message": "prepare only" if result is None else result.message,
        "oe_iterations": 0 if result is None else result.n_iterations,
        "degrees_of_freedom_for_signal": None if result is None else result.degrees_of_freedom_for_signal,
        "prior_temperature_rmse_K": _rmse(prepared.prior_temperature_K, prepared.truth_temperature_K),
        "retrieved_temperature_rmse_K": _rmse(retrieved, prepared.truth_temperature_K),
        "retrieved_reduced_chi_square": float(np.mean(np.square(normalized_residual))),
        "retrieved_max_absolute_residual_ppm": float(
            np.max(np.abs(prepared.truth_spectrum - retrieved_spectrum)) * 1.0e6
        ),
        "nemesis_interchange": {
            "atmosphere": "hat_p_32b_atmosphere.txt",
            "spectrum": "hat_p_32b_spectrum_100_bins.txt",
            "contract": "nemesis_comparison_contract.json",
        },
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    arrays = {
        "pressure_bar": prepared.pressure_bar,
        "truth_temperature_K": prepared.truth_temperature_K,
        "prior_temperature_K": prepared.prior_temperature_K,
        "retrieved_temperature_K": retrieved,
        "wavelength_micron": prepared.wavelength_micron,
        "wavelength_bin_edges_micron": prepared.bin_edges_micron,
        "truth_eclipse_depth": prepared.truth_spectrum,
        "prior_eclipse_depth": prior_spectrum,
        "retrieved_eclipse_depth": retrieved_spectrum,
        "vmr": prepared.vmr,
        "gas_names": np.asarray(prepared.gas_names),
    }
    if result is not None:
        arrays.update(
            covariance=result.covariance,
            averaging_kernel=result.averaging_kernel,
            jacobian=result.jacobian,
            measurement_error_covariance=result.measurement_error_covariance,
            smoothing_error_covariance=result.smoothing_error_covariance,
        )
    np.savez(output / "benchmark_arrays.npz", **arrays)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def _temperature_prior(truth: np.ndarray) -> np.ndarray:
    phase = np.linspace(0.0, np.pi, truth.size)
    return np.maximum(250.0, truth - 180.0 + 80.0 * np.cos(phase))


def _write_interchange_tables(
    output: Path,
    prepared: PreparedBenchmark,
    uncertainty_ppm: float,
) -> None:
    atmosphere = np.column_stack(
        (
            prepared.pressure_bar,
            prepared.truth_temperature_K,
            prepared.prior_temperature_K,
            prepared.vmr,
        )
    )
    np.savetxt(
        output / "hat_p_32b_atmosphere.txt",
        atmosphere,
        header="pressure_bar truth_temperature_K prior_temperature_K " + " ".join(prepared.gas_names),
    )
    np.savetxt(
        output / "hat_p_32b_spectrum_100_bins.txt",
        np.column_stack(
            (
                prepared.bin_edges_micron[:-1],
                prepared.bin_edges_micron[1:],
                prepared.wavelength_micron,
                prepared.truth_spectrum,
                np.full(prepared.wavelength_micron.size, uncertainty_ppm * 1.0e-6),
            )
        ),
        header="lower_micron upper_micron center_micron eclipse_depth uncertainty",
    )
    contract = {
        "target": "HAT-P-32b",
        "geometry": "secondary-eclipse thermal emission",
        "clouds": "none",
        "pressure_unit": "bar",
        "temperature_unit": "K",
        "composition": "volume mixing ratio with H2/He background fill",
        "observable": "planet/star eclipse depth",
        "stellar_model": "6001 K blackbody",
        "planet_radius_m": PLANET_RADIUS_M,
        "planet_mass_kg": PLANET_MASS_KG,
        "star_radius_m": STAR_RADIUS_M,
        "opacity_note": "ROBERT uses bundled ExoMol-derived correlated-k archives; NEMESIS must record its own k-table checksums, so spectral differences are not assumed to be algorithm-only.",
    }
    (output / "nemesis_comparison_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/robert-hat-p-32b-vertical-oe")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--layers", type=int, default=100)
    parser.add_argument("--uncertainty-ppm", type=float, default=30.0)
    parser.add_argument("--forward-model-error-ppm", type=float, default=0.0)
    parser.add_argument("--opacity-g-points", type=int, default=30)
    parser.add_argument("--kta-dir", default=str(DEFAULT_KTA_DIR))
    parser.add_argument("--correlation-length", type=float, default=1.5)
    parser.add_argument("--prior-sigma-K", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--finite-difference-fraction", type=float, default=1.0e-4)
    parser.add_argument(
        "--finite-difference-scheme",
        choices=("forward", "central"),
        default="forward",
        help="Forward differencing halves layer-profile Jacobian cost; central is more accurate.",
    )
    parser.add_argument("--marquardt-lambda", type=float, default=1.0)
    parser.add_argument("--speed-repeats", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    run_benchmark(_parser().parse_args())
