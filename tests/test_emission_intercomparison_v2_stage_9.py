from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (
    FRAMEWORKS,
    GAUSSIAN_NOISE_SEEDS,
    MULTINEST_SETTINGS,
    MULTINEST_SEED_MAX,
    NOISE_TIERS_PPM,
    SCENARIOS,
    build_run_matrix,
    frozen_contract_payload,
    parameter_definitions,
)


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "docs/data/emission_intercomparison/version_2/common_contract.json"
CONTRACT = (
    ROOT
    / "docs/data/emission_intercomparison/version_2/stage_9_retrieval_contract.json"
)
PREPARE = ROOT / "scripts/prepare_emission_intercomparison_v2_stage_9.py"
NATIVE = ROOT / "examples/emission_intercomparison_v2_stage_9_native.py"
RETRIEVAL_RUNNER = (
    ROOT / "examples/run_emission_intercomparison_v2_stage_9_retrieval.py"
)
SINGLE_RUN_PLOT = (
    ROOT / "examples/plot_emission_intercomparison_v2_stage_9_run.py"
)
BIG_COMPARISON_PLOT = (
    ROOT / "examples/plot_emission_intercomparison_v2_stage_9_big_comparison.py"
)
PAPER_ATLAS_PLOT = (
    ROOT / "examples/plot_emission_intercomparison_v2_stage_9_paper_atlas.py"
)
MPI_LAUNCHER = (
    ROOT / "scripts/launch_emission_intercomparison_v2_stage_9_mpi.sh"
)
TASK_LAUNCHER = ROOT / "scripts/submit_emission_intercomparison_v2_stage_9_task.sh"
PRODUCTION_LAUNCHER = (
    ROOT / "scripts/submit_emission_intercomparison_v2_stage_9.sh"
)
SHARD_SUBMITTER = (
    ROOT / "scripts/queue_emission_intercomparison_v2_stage_9_shard.sh"
)
ENVELOPE_SUBMITTER = (
    ROOT / "scripts/queue_emission_intercomparison_v2_stage_9_envelopes.sh"
)


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("stage9_prepare_for_tests", PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_native_module():
    spec = importlib.util.spec_from_file_location("stage9_native_for_tests", NATIVE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_retrieval_runner_module():
    spec = importlib.util.spec_from_file_location(
        "stage9_retrieval_runner_for_tests", RETRIEVAL_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_single_run_plot_module():
    spec = importlib.util.spec_from_file_location(
        "stage9_single_run_plot_for_tests", SINGLE_RUN_PLOT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_big_comparison_plot_module():
    spec = importlib.util.spec_from_file_location(
        "stage9_big_comparison_plot_for_tests", BIG_COMPARISON_PLOT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_paper_atlas_plot_module():
    spec = importlib.util.spec_from_file_location(
        "stage9_paper_atlas_plot_for_tests", PAPER_ATLAS_PLOT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_stage_9_matrix_counts_and_resources() -> None:
    runs = build_run_matrix()
    assert len(runs) == 72
    assert len({run.run_id for run in runs}) == 72
    assert len({run.shard_id for run in runs}) == 12
    assert {
        sum(item.shard_id == shard for item in runs)
        for shard in {item.shard_id for item in runs}
    } == {6}
    assert sum(run.control == "directed_cross_framework" for run in runs) == 72
    assert sum(run.control == "self_retrieval_control" for run in runs) == 0
    assert {run.mpi_ranks for run in runs} == {12}
    assert {run.threads_per_rank for run in runs} == {1}
    assert MULTINEST_SETTINGS["n_live_points"] == 400
    assert MULTINEST_SETTINGS["mpi_nprocs"] == 12
    assert MULTINEST_SETTINGS["verbose"] is True
    assert len(GAUSSIAN_NOISE_SEEDS) == 0
    assert {run.noise_id for run in runs} == {"mean"}
    assert all(0 <= run.sampler_seed <= 0x7FFF_FFFF for run in runs)
    assert any(run.sampler_seed > MULTINEST_SEED_MAX for run in runs)


def test_stage_9_parameters_exclude_area_and_retrieve_clouds() -> None:
    dimensions = {
        scenario.name: len(parameter_definitions(scenario)) for scenario in SCENARIOS
    }
    assert dimensions == {
        "clear_non_inverted": 9,
        "clear_inverted": 9,
        "grey_absorbing_non_inverted": 11,
        "grey_scattering_non_inverted": 12,
    }
    cloudy_truths = {
        scenario.name: (
            scenario.cloud_tau_truth,
            scenario.cloud_top_pressure_bar_truth,
        )
        for scenario in SCENARIOS
        if scenario.cloudy
    }
    assert cloudy_truths == {
        "grey_absorbing_non_inverted": (3.0, 3.0e-3),
        "grey_scattering_non_inverted": (3.0, 3.0e-3),
    }
    for scenario in SCENARIOS:
        definitions = parameter_definitions(scenario)
        names = {item.name for item in definitions}
        assert "area_scale" not in names
        assert "log10_area_scale" not in names
        if scenario.cloudy:
            assert {"log10_cloud_tau_5um", "log10_cloud_top_pressure_bar"} <= names
            truth_by_name = {item.name: item.truth for item in definitions}
            np.testing.assert_allclose(
                truth_by_name["log10_cloud_tau_5um"],
                np.log10(scenario.cloud_tau_truth),
            )
            np.testing.assert_allclose(
                truth_by_name["log10_cloud_top_pressure_bar"],
                np.log10(scenario.cloud_top_pressure_bar_truth),
            )
        if scenario.cloud == "grey_isotropic_scattering":
            assert "cloud_single_scattering_albedo" in names
            assert {item.name: item.truth for item in definitions}[
                "cloud_single_scattering_albedo"
            ] == scenario.cloud_single_scattering_albedo_truth


def test_stage_9_robert_cloud_path_selects_compiled_cpu_backends() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    assert "retain_species_tau=False" in source
    assert 'backend="numba"' in source
    assert 'boundary_backend="numba"' in source


def test_picaso_projection_uses_native_bin_support_without_interpolation() -> None:
    module = _load_native_module()
    projected = module._native_bin_overlap_mean(
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
        np.array([2.0, 4.0]),
        np.array([0.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(projected, [3.0, 4.0])

    center, lower, upper, order = module._picaso_native_wavelength_support(
        np.array([1000.0, 2000.0]),
        np.array([1000.0, 1000.0]),
    )
    np.testing.assert_allclose(center, [5.0, 10.0])
    np.testing.assert_allclose(lower, [4.0, 1.0e4 / 1500.0])
    np.testing.assert_allclose(upper, [1.0e4 / 1500.0, 20.0])
    np.testing.assert_array_equal(order, [1, 0])


def test_cloud_optical_depth_combination_copies_read_only_gas_array() -> None:
    module = _load_native_module()
    gas = np.ones((2, 3, 1))
    gas.setflags(write=False)
    cia = np.full((2, 3), 0.25)

    combined = module._combined_gas_optical_depth(gas, [cia])

    assert combined.flags.writeable
    np.testing.assert_allclose(combined, 1.25)
    np.testing.assert_allclose(gas, 1.0)


def test_weighted_spectral_quantiles_use_posterior_samples() -> None:
    module = _load_retrieval_runner_module()
    q16, q50, q84 = module._weighted_spectral_quantiles(
        np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        np.asarray([0.2, 0.3, 0.5]),
    )
    np.testing.assert_allclose(q16, [1.0, 10.0])
    np.testing.assert_allclose(q50, [2.0, 20.0])
    np.testing.assert_allclose(q84, [2.68, 26.8])


def test_default_envelopes_include_one_and_two_sigma_and_chemistry() -> None:
    module = _load_retrieval_runner_module()
    assert module.POSTERIOR_ENVELOPE_QUANTILES == (
        0.025,
        0.16,
        0.50,
        0.84,
        0.975,
    )
    definitions = parameter_definitions("clear_non_inverted")
    names = tuple(item.name for item in definitions)
    truth = np.asarray([[item.truth for item in definitions]], dtype=float)
    species, chemistry = module._posterior_chemistry(names, truth)

    assert species == ("H2", "He", "H2O", "CO", "CO2", "CH4")
    assert chemistry.shape == (1, 6)
    np.testing.assert_allclose(np.sum(chemistry, axis=1), 1.0)


def test_retrieval_runner_loads_serialized_posterior_samples(
    tmp_path: Path,
) -> None:
    module = _load_retrieval_runner_module()
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.savez(
        tmp_path / "result_arrays.npz",
        samples=expected,
        log_likelihood=np.asarray([-2.0, -1.0]),
    )

    actual = module._load_saved_posterior_samples(tmp_path, parameter_count=2)

    np.testing.assert_array_equal(actual, expected)


def test_posterior_plot_metadata_follows_result_names_and_run_config() -> None:
    module = _load_single_run_plot_module()
    run = {
        "parameters": [
            {"name": "cloud_fraction", "label": "Cloud fraction"},
            {
                "name": "temperature",
                "label": "Temperature",
                "unit": "K",
                "reference_value": 1500.0,
            },
        ]
    }

    metadata = module._parameter_metadata(
        run,
        ("temperature", "cloud_fraction"),
    )

    assert [item["name"] for item in metadata] == [
        "temperature",
        "cloud_fraction",
    ]
    assert module._parameter_label(metadata[0]) == "Temperature [K]"
    assert module._reference_value(metadata[0]) == 1500.0
    assert module._reference_value(metadata[1]) is None


def test_committed_stage_9_contract_matches_source_of_truth() -> None:
    sha = hashlib.sha256(COMMON.read_bytes()).hexdigest()
    expected = frozen_contract_payload(common_contract_sha256=sha)
    assert json.loads(CONTRACT.read_text(encoding="utf-8")) == expected
    assert expected["scope_exclusions"] == [
        "Track A shared-tensor retrievals",
        "MgSiO3 or other condensate microphysics",
        "Mie optical properties",
        "anisotropic scattering",
        "wavelength-dependent cloud extinction",
        "high-order scattering methods",
        "fabricated shared opacity or cloud tensors",
    ]
    assert expected["execution"]["scheduler_queue"] == "redwood"
    assert expected["execution"]["addqueue_allocation"] == "single_wrapper_n12"
    assert expected["execution"]["mpi_launcher"] == "conda_mpich_hydra_fork"
    assert expected["execution"]["single_node_required"] is True
    assert expected["noise"]["spectral_points_randomized"] is False
    assert expected["sampler"]["seed_policy"] == {
        "requested_derivation": (
            "first_32_bits_sha256_masked_to_nonnegative_31_bit"
        ),
        "effective_mapping": "requested_seed_modulo_30081_at_multinest_boundary",
        "effective_minimum": 0,
        "effective_maximum": 30080,
    }
    assert (
        expected["fixed"]["picaso_r100_projection"]
        == "native_wavenumber_bin_support_overlap_no_center_interpolation"
    )


def test_directory_generator_creates_and_verifies_complete_tree(tmp_path: Path) -> None:
    module = _load_prepare_module()
    project = tmp_path / "stage9"
    summary = module.prepare(project)
    assert summary["run_count"] == 72
    assert summary["shard_count"] == 12
    assert summary["noise_vector_count"] == 0
    assert len(list((project / "runs").glob("*/*/*/run.json"))) == 72
    assert len(list((project / "shards").glob("*.json"))) == 12
    assert not (project / "noise").exists()
    assert len(list((project / "injections").glob("*/*"))) >= 12
    module.prepare(project, verify_only=True)


def test_execution_contract_refresh_preserves_deployment(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    project = tmp_path / "stage9"
    expected_manifest = module.prepare(project)

    contract_path = project / "contracts" / "stage_9_retrieval_contract.json"
    old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    old_contract["execution"] = {
        "cluster": "glamdring",
        "scheduler_queue": "redwood",
        "mpi_ranks_per_retrieval": 12,
        "threads_per_rank": 1,
        "nested_mpirun_forbidden_under_addqueue": True,
    }
    contract_path.write_text(
        json.dumps(old_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = project / "integrity" / "setup_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_manifest["stage_9_contract_repository_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(old_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    refreshed = module.prepare(project, refresh_execution_contract=True)
    assert refreshed == expected_manifest
    assert json.loads(contract_path.read_text(encoding="utf-8")) == json.loads(
        CONTRACT.read_text(encoding="utf-8")
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest

    injection = (
        project
        / "injections"
        / "robert"
        / "clear_non_inverted"
        / "native_mean.npz"
    )
    injection.touch()
    contract_path.write_text(
        json.dumps(old_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_manifest["stage_9_contract_repository_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(old_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    module.prepare(project, refresh_execution_contract=True)


def test_multinest_seed_refresh_preserves_injections_and_rejects_run_outputs(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    project = tmp_path / "stage9"
    module.prepare(project)

    contract_path = project / "contracts" / "stage_9_retrieval_contract.json"
    legacy_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    legacy_contract["sampler"].pop("seed_policy")
    contract_path.write_text(
        json.dumps(legacy_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = project / "integrity" / "setup_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage_9_contract_repository_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_files = tuple(sorted(project.glob("runs/*/*/*/run.json")))
    run_bytes_before = {path: path.read_bytes() for path in run_files}

    injection = (
        project
        / "injections"
        / "picaso"
        / "clear_non_inverted"
        / "native_mean.npz"
    )
    injection.write_bytes(b"preserve-me")

    module.prepare(project, refresh_multinest_seeds=True)

    assert injection.read_bytes() == b"preserve-me"
    assert {path: path.read_bytes() for path in run_files} == run_bytes_before
    module.prepare(project, verify_only=True)

    first_run = next(project.glob("runs/*/*/*"))
    (first_run / "sampler_status.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="after production output"):
        module.prepare(project, refresh_multinest_seeds=True)


def test_glamdring_launchers_use_one_wrapper_and_conda_mpich() -> None:
    assert MPI_LAUNCHER.stat().st_mode & 0o111
    mpi_text = MPI_LAUNCHER.read_text(encoding="utf-8")
    assert 'PMI*|PMIX*) unset "$name"' in mpi_text
    assert "export HYDRA_LAUNCHER=fork" in mpi_text
    assert "export HYDRA_RMK=user" in mpi_text
    assert "-rmk user" in mpi_text
    assert "-launcher fork" in mpi_text

    task_text = TASK_LAUNCHER.read_text(encoding="utf-8")
    production_text = PRODUCTION_LAUNCHER.read_text(encoding="utf-8")
    assert "launch_emission_intercomparison_v2_stage_9_mpi.sh" in task_text
    assert '"$environment_prefix" 1 "$python_executable"' in task_text
    assert "posterior-envelopes|spectral-envelope)" in task_text
    assert "generate_emission_intercomparison_v2_stage_9_posterior_envelopes.py" in task_text
    assert "launch_emission_intercomparison_v2_stage_9_mpi.sh" in production_text

    shard_text = SHARD_SUBMITTER.read_text(encoding="utf-8")
    assert "addqueue -q redwood -s" in shard_text
    assert "-n 1x12" in shard_text
    assert "-n 12" not in shard_text
    assert '"preliminary_memory_gb"' in shard_text
    assert "memory_per_cpu_gb=$(( (total_memory_gb + 11) / 12 ))" in shard_text
    assert '-m "$memory_per_cpu_gb"' in shard_text
    assert "Batch job submission failed" in shard_text
    assert "Stopping shard after an unconfirmed addqueue submission" in shard_text

    envelope_text = ENVELOPE_SUBMITTER.read_text(encoding="utf-8")
    assert "STAGE9_TASK=posterior-envelopes" in envelope_text
    assert "-n 1x12" in envelope_text
    assert 'addqueue -q redwood -s -c "s9-envelope-' in envelope_text


def test_single_run_plotter_writes_spectrum_tp_and_corner_products(
    tmp_path: Path,
) -> None:
    module = _load_single_run_plot_module()
    assert module.MODEL_COLORS == {
        "robert": "#9370DB",
        "petitradtrans": "#DDA0DD",
        "picaso": "#36454F",
    }
    definitions = parameter_definitions("clear_non_inverted")
    names = [item.name for item in definitions]
    truths = np.asarray([item.truth for item in definitions], dtype=float)
    rng = np.random.default_rng(42)
    samples = np.tile(truths, (80, 1))
    for index, item in enumerate(definitions):
        samples[:, index] += rng.normal(
            0.0, 0.01 * (item.upper - item.lower), samples.shape[0]
        )
        samples[:, index] = np.clip(samples[:, index], item.lower, item.upper)
    weights = np.linspace(1.0, 2.0, samples.shape[0])
    weights /= np.sum(weights)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wavelength = np.linspace(1.0, 12.0, 40)
    injection = np.linspace(5.0e-4, 2.5e-3, wavelength.size)
    common = json.loads(COMMON.read_text(encoding="utf-8"))
    pressure = np.asarray(
        next(
            item["centers_bar"]
            for item in common["pressure_grids"]
            if item["n_cells"] == 80
        ),
        dtype=float,
    )
    best_tp = np.linspace(2500.0, 1000.0, pressure.size)
    np.savez(
        run_dir / "result_arrays.npz",
        samples=samples,
        log_likelihood=np.linspace(-10.0, 0.0, samples.shape[0]),
        weights=weights,
    )
    np.savez(
        run_dir / "diagnostic_spectra.npz",
        wavelength_micron=wavelength,
        observed_eclipse_depth=injection,
        observational_uncertainty_eclipse_depth=np.full(
            wavelength.size, 60.0e-6
        ),
        best_fit_eclipse_depth=injection + 2.0e-6,
        posterior_spectrum_q025_eclipse_depth=injection - 9.0e-6,
        posterior_spectrum_q16_eclipse_depth=injection - 5.0e-6,
        posterior_spectrum_q50_eclipse_depth=injection - 1.0e-6,
        posterior_spectrum_q84_eclipse_depth=injection + 6.0e-6,
        posterior_spectrum_q975_eclipse_depth=injection + 10.0e-6,
    )
    np.savez(
        run_dir / "diagnostic_tp.npz",
        pressure_bar=pressure,
        best_fit_temperature_k=best_tp,
        posterior_temperature_q025_k=best_tp - 180.0,
        posterior_temperature_q16_k=best_tp - 90.0,
        posterior_temperature_q50_k=best_tp,
        posterior_temperature_q84_k=best_tp + 90.0,
        posterior_temperature_q975_k=best_tp + 180.0,
    )
    np.savez(
        run_dir / "diagnostic_chemistry.npz",
        species=np.asarray(("H2", "He", "H2O", "CO", "CO2", "CH4")),
        best_fit_vmr=np.asarray((0.84, 0.14, 1.0e-3, 1.0e-3, 1.0e-4, 1.0e-5)),
        posterior_vmr_q025=np.full(6, 1.0e-6),
        posterior_vmr_q16=np.full(6, 2.0e-6),
        posterior_vmr_q50=np.full(6, 3.0e-6),
        posterior_vmr_q84=np.full(6, 4.0e-6),
        posterior_vmr_q975=np.full(6, 5.0e-6),
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "parameter_names": names,
                "best_fit_parameters": dict(zip(names, truths, strict=True)),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "posterior_summary.json").write_text("{}", encoding="utf-8")
    run_config = tmp_path / "run.json"
    run_config.write_text(
        json.dumps(
            {
                "run_id": "synthetic-single-run",
                "scenario": "clear_non_inverted",
                "injector": "robert",
                "retriever": "picaso",
                "noise_ppm": 60,
                "run_directory": str(run_dir),
                "common_contract": str(COMMON),
                "parameters": json.loads(CONTRACT.read_text(encoding="utf-8"))[
                    "parameters_by_scenario"
                ]["clear_non_inverted"],
            }
        ),
        encoding="utf-8",
    )

    output = module.plot_individual_run(run_config, output=tmp_path / "plots")

    assert (output / "spectrum_fit.png").stat().st_size > 0
    assert (output / "temperature_pressure.png").stat().st_size > 0
    assert (output / "posterior_corner.png").stat().st_size > 0


def test_big_comparison_uses_input_tp_and_four_molecular_posteriors(
    tmp_path: Path,
) -> None:
    module = _load_big_comparison_plot_module()
    assert module.MODEL_COLORS == {
        "robert": "#9370DB",
        "petitradtrans": "#DDA0DD",
        "picaso": "#36454F",
    }
    definitions = parameter_definitions("clear_non_inverted")
    names = [item.name for item in definitions]
    truths = np.asarray([item.truth for item in definitions], dtype=float)
    project = tmp_path / "stage9"
    contracts = project / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "common_contract.json").write_bytes(COMMON.read_bytes())
    rows = []
    rng = np.random.default_rng(91)
    wavelength = np.linspace(1.0, 12.0, 30)
    injection = np.linspace(5.0e-4, 2.5e-3, wavelength.size)
    common = json.loads(COMMON.read_text(encoding="utf-8"))
    pressure = np.asarray(
        next(
            item["centers_bar"]
            for item in common["pressure_grids"]
            if item["n_cells"] == 80
        ),
        dtype=float,
    )
    best_tp = np.linspace(2400.0, 1100.0, pressure.size)
    for retriever in ("picaso", "petitradtrans"):
        run_id = f"synthetic-robert-to-{retriever}"
        run_dir = project / "runs" / retriever / "clear_non_inverted" / run_id
        run_dir.mkdir(parents=True)
        samples = np.tile(truths, (60, 1))
        for index, item in enumerate(definitions):
            samples[:, index] += rng.normal(
                0.0, 0.01 * (item.upper - item.lower), samples.shape[0]
            )
            samples[:, index] = np.clip(samples[:, index], item.lower, item.upper)
        weights = np.ones(samples.shape[0]) / samples.shape[0]
        np.savez(
            run_dir / "result_arrays.npz",
            samples=samples,
            log_likelihood=np.linspace(-4.0, 0.0, samples.shape[0]),
            weights=weights,
        )
        np.savez(
            run_dir / "diagnostic_spectra.npz",
            wavelength_micron=wavelength,
            observed_eclipse_depth=injection,
            observational_uncertainty_eclipse_depth=np.full(
                wavelength.size, 100.0e-6
            ),
            best_fit_eclipse_depth=injection + 1.0e-6,
            posterior_spectrum_q025_eclipse_depth=injection - 8.0e-6,
            posterior_spectrum_q16_eclipse_depth=injection - 4.0e-6,
            posterior_spectrum_q50_eclipse_depth=injection,
            posterior_spectrum_q84_eclipse_depth=injection + 5.0e-6,
            posterior_spectrum_q975_eclipse_depth=injection + 9.0e-6,
        )
        np.savez(
            run_dir / "diagnostic_tp.npz",
            pressure_bar=pressure,
            best_fit_temperature_k=best_tp,
            posterior_temperature_q025_k=best_tp - 160.0,
            posterior_temperature_q16_k=best_tp - 80.0,
            posterior_temperature_q50_k=best_tp,
            posterior_temperature_q84_k=best_tp + 80.0,
            posterior_temperature_q975_k=best_tp + 160.0,
        )
        np.savez(
            run_dir / "diagnostic_chemistry.npz",
            species=np.asarray(("H2", "He", "H2O", "CO", "CO2", "CH4")),
            best_fit_vmr=np.asarray(
                (0.84, 0.14, 1.0e-3, 1.0e-3, 1.0e-4, 1.0e-5)
            ),
            posterior_vmr_q025=np.full(6, 1.0e-6),
            posterior_vmr_q16=np.full(6, 2.0e-6),
            posterior_vmr_q50=np.full(6, 3.0e-6),
            posterior_vmr_q84=np.full(6, 4.0e-6),
            posterior_vmr_q975=np.full(6, 5.0e-6),
        )
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "parameter_names": names,
                    "best_fit_parameters": dict(zip(names, truths, strict=True)),
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "posterior_summary.json").write_text("{}", encoding="utf-8")
        relative_config = Path("runs") / retriever / "clear_non_inverted" / run_id / "run.json"
        run_config = project / relative_config
        run_config.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "scenario": "clear_non_inverted",
                    "injector": "robert",
                    "retriever": retriever,
                    "noise_ppm": 100,
                    "run_directory": str(run_dir),
                    "parameters": json.loads(
                        CONTRACT.read_text(encoding="utf-8")
                    )["parameters_by_scenario"]["clear_non_inverted"],
                }
            ),
            encoding="utf-8",
        )
        rows.append({"run_config": str(relative_config)})
    (project / "run_index.json").write_text(json.dumps(rows), encoding="utf-8")

    pdf, pages = module.plot_big_comparison(
        project,
        scenario_filter="clear_non_inverted",
        tier_filter=100,
        injector_filter="robert",
    )

    assert pages == 1
    assert pdf.stat().st_size > 0
    assert (
        project
        / "diagnostics"
        / "big_comparison"
        / "clear_non_inverted"
        / "100ppm"
        / "data_generated_with_robert.png"
    ).stat().st_size > 0


def test_paper_atlas_compresses_complete_cloudy_scenario(
    tmp_path: Path,
) -> None:
    module = _load_paper_atlas_plot_module()
    scenario = "grey_scattering_non_inverted"
    project = tmp_path / "stage9-paper-atlas"
    contracts = project / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "common_contract.json").write_bytes(COMMON.read_bytes())
    common = json.loads(COMMON.read_text(encoding="utf-8"))
    parameters = json.loads(CONTRACT.read_text(encoding="utf-8"))[
        "parameters_by_scenario"
    ][scenario]
    definitions = parameter_definitions(scenario)
    names = tuple(item.name for item in definitions)
    truths = np.asarray([item.truth for item in definitions], dtype=float)
    truth_mapping = {item.name: item.truth for item in definitions}
    reference_tp = module.atmospheric_state(
        common,
        scenario,
        truth_mapping,
    ).temperature_cells_k
    pressure = np.asarray(
        next(
            item["centers_bar"]
            for item in common["pressure_grids"]
            if item["n_cells"] == 80
        ),
        dtype=float,
    )
    wavelength = np.linspace(0.8, 12.0, 110)
    base_spectrum = (
        1.45e-3
        + 6.0e-4 * np.tanh((wavelength - 3.0) / 2.7)
        + 1.2e-4 * np.sin(1.9 * wavelength) * np.exp(-wavelength / 10.0)
    )
    rows = []
    for tier in NOISE_TIERS_PPM:
        for injector_index, injector in enumerate(FRAMEWORKS):
            observed = base_spectrum + injector_index * 7.0e-6
            for retriever_index, retriever in enumerate(FRAMEWORKS):
                if retriever == injector:
                    continue
                run_id = (
                    f"synthetic-{scenario}-{injector}-to-{retriever}-{tier:03d}ppm"
                )
                run_directory = project / "runs" / retriever / scenario / run_id
                run_directory.mkdir(parents=True)
                direction = float(retriever_index - injector_index)
                residual_sigma = (
                    0.24 * np.sin(0.9 * wavelength + retriever_index)
                    + 0.10 * direction
                )
                best_spectrum = observed + tier * 1.0e-6 * residual_sigma
                width68 = tier * 1.0e-6 * (
                    0.32 + 0.04 * np.cos(wavelength)
                )
                width95 = 1.9 * width68
                np.savez(
                    run_directory / "diagnostic_spectra.npz",
                    wavelength_micron=wavelength,
                    observed_eclipse_depth=observed,
                    observational_uncertainty_eclipse_depth=np.full(
                        wavelength.size, tier * 1.0e-6
                    ),
                    best_fit_eclipse_depth=best_spectrum,
                    posterior_spectrum_q025_eclipse_depth=(
                        best_spectrum - width95
                    ),
                    posterior_spectrum_q16_eclipse_depth=(
                        best_spectrum - width68
                    ),
                    posterior_spectrum_q50_eclipse_depth=best_spectrum,
                    posterior_spectrum_q84_eclipse_depth=(
                        best_spectrum + width68
                    ),
                    posterior_spectrum_q975_eclipse_depth=(
                        best_spectrum + width95
                    ),
                )
                tp_offset = direction * (
                    18.0 + 10.0 * np.sin(np.linspace(0.0, np.pi, pressure.size))
                )
                best_tp = reference_tp + tp_offset
                tp_width68 = 35.0 + 7.0 * np.cos(
                    np.linspace(0.0, np.pi, pressure.size)
                )
                np.savez(
                    run_directory / "diagnostic_tp.npz",
                    pressure_bar=pressure,
                    best_fit_temperature_k=best_tp,
                    posterior_temperature_q025_k=best_tp - 1.9 * tp_width68,
                    posterior_temperature_q16_k=best_tp - tp_width68,
                    posterior_temperature_q50_k=best_tp,
                    posterior_temperature_q84_k=best_tp + tp_width68,
                    posterior_temperature_q975_k=best_tp + 1.9 * tp_width68,
                )
                seed = 1000 + 100 * tier + 10 * injector_index + retriever_index
                rng = np.random.default_rng(seed)
                samples = np.tile(truths, (450, 1))
                for parameter_index, definition in enumerate(definitions):
                    span = definition.upper - definition.lower
                    sample_sigma = 0.025 * span * np.sqrt(tier / 30.0)
                    target_pull = (0.25, 1.35, 2.60)[parameter_index % 3]
                    samples[:, parameter_index] += rng.normal(
                        np.sign(direction) * target_pull * sample_sigma,
                        sample_sigma,
                        samples.shape[0],
                    )
                    samples[:, parameter_index] = np.clip(
                        samples[:, parameter_index],
                        definition.lower,
                        definition.upper,
                    )
                weights = np.linspace(1.0, 2.0, samples.shape[0])
                weights /= np.sum(weights)
                log_likelihood = -0.5 * np.sum(
                    ((samples - truths) / np.maximum(0.05, np.abs(truths))) ** 2,
                    axis=1,
                )
                np.savez(
                    run_directory / "result_arrays.npz",
                    samples=samples,
                    weights=weights,
                    log_likelihood=log_likelihood,
                )
                best_parameters = dict(zip(names, truths, strict=True))
                (run_directory / "result.json").write_text(
                    json.dumps(
                        {
                            "parameter_names": names,
                            "best_fit_parameters": best_parameters,
                        }
                    ),
                    encoding="utf-8",
                )
                reduced = float(np.mean(residual_sigma**2))
                rms = float(
                    np.sqrt(np.mean((best_spectrum - observed) ** 2)) * 1.0e6
                )
                (run_directory / "posterior_summary.json").write_text(
                    json.dumps(
                        {
                            "fit_metrics": {
                                "best_fit_reduced_chi_square": reduced,
                                "best_fit_residual_rms_ppm": rms,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                np.savez(
                    run_directory / "diagnostic_chemistry.npz",
                    species=np.asarray(("H2", "He", "H2O", "CO", "CO2", "CH4")),
                    best_fit_vmr=np.asarray(
                        (0.84, 0.14, 1.0e-3, 1.0e-3, 1.0e-4, 1.0e-5)
                    ),
                )
                relative_config = (
                    Path("runs") / retriever / scenario / run_id / "run.json"
                )
                (project / relative_config).write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "scenario": scenario,
                            "injector": injector,
                            "retriever": retriever,
                            "noise_ppm": tier,
                            "noise_id": "mean",
                            "run_directory": str(run_directory),
                            "parameters": parameters,
                        }
                    ),
                    encoding="utf-8",
                )
                rows.append({"run_config": str(relative_config)})
    (project / "run_index.json").write_text(json.dumps(rows), encoding="utf-8")

    # Older completed Stage-9 runs can have all compact retrieval products but
    # no optional fit-summary JSON. The atlas must calculate its fit metrics
    # from the saved spectrum in that case.
    next(project.glob("runs/*/*/*/posterior_summary.json")).unlink()

    pdf, manifest, created = module.generate_paper_atlas(
        project,
        scenario_filter=scenario,
        dpi=140,
    )

    assert pdf.stat().st_size > 0
    assert manifest.stat().st_size > 0
    assert len(created) == 10
    assert all(path.stat().st_size > 0 for path in created)
    assert {
        path.stem.removeprefix(f"{scenario}_")
        for path in created
        if path.suffix == ".png"
    } == {
        "spectral_residual_atlas",
        "tp_atlas",
        "chemistry_posterior_atlas",
        "cloud_posterior_atlas",
        "parameter_bias_coverage_matrix",
    }
    report = json.loads(
        (project / "diagnostics/paper_atlas/parameter_bias_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    records = (
        project / "diagnostics/paper_atlas/parameter_bias_coverage.csv"
    ).read_text(encoding="utf-8").splitlines()
    reported_definitions = [
        definition
        for definition in definitions
        if definition.name.startswith("log10_vmr_")
        or "cloud" in definition.name
        or "scattering_albedo" in definition.name
    ]
    assert report["record_count"] == 18 * len(reported_definitions)
    assert len(records) == 1 + report["record_count"]
    assert len(report["aggregates"]["by_scenario_and_noise_tier"]) == 3
    assert len(report["aggregates"]["by_directed_pair"]) == 18
    assert {
        item["parameter_family"]
        for item in report["aggregates"]["by_parameter_family"]
    } == {"chemistry", "cloud"}
    assert module._parameter_label({"name": "log10_vmr_H2O"}) == (
        r"$\log_{10}\,\mathrm{VMR}(\mathrm{H_2O})$"
    )
    assert module._parameter_label(
        {"name": "cloud_single_scattering_albedo"}
    ) == r"$\omega_0$"
