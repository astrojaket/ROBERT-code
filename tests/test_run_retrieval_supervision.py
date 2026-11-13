"""Tests for the configured retrieval supervision option."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import run_retrieval


def test_supervision_flag_launches_fresh_multinest_command(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "configuration.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    config = SimpleNamespace(
        sampler=SimpleNamespace(engine="multinest"),
        runtime=SimpleNamespace(mpi_processes="auto"),
        outputs=SimpleNamespace(directory=tmp_path / "outputs"),
    )
    captured = {}

    def supervise(
        command, *, fallback_output_dir, mpi_nprocs, mpi_launcher
    ):
        captured["command"] = list(command)
        captured["fallback_output_dir"] = fallback_output_dir
        captured["mpi_nprocs"] = mpi_nprocs
        captured["mpi_launcher"] = mpi_launcher
        return 23

    monkeypatch.setattr(run_retrieval, "load_task_config", lambda path: config)
    monkeypatch.setattr(run_retrieval, "supervise_retrieval_command", supervise)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval.py",
            "--config",
            str(config_path),
            "--supervise-callback-failures",
        ],
    )

    assert run_retrieval.main() == 23
    assert captured["command"] == [
        sys.executable,
        "-u",
        str(Path(run_retrieval.__file__).resolve()),
        "--config",
        str(config_path.resolve()),
    ]
    assert captured["fallback_output_dir"] == tmp_path / "outputs" / "multinest"
    assert captured["mpi_nprocs"] == 1
    assert captured["mpi_launcher"] is None


def test_supervision_uses_nested_sampling_output_for_hybrid_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "configuration.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    config = SimpleNamespace(
        sampler=SimpleNamespace(engine="optimal_estimation_to_multinest"),
        runtime=SimpleNamespace(mpi_processes=1),
        outputs=SimpleNamespace(directory=tmp_path / "outputs"),
    )
    captured = {}

    def supervise(
        command, *, fallback_output_dir, mpi_nprocs, mpi_launcher
    ):
        del command
        captured["fallback_output_dir"] = fallback_output_dir
        captured["mpi_nprocs"] = mpi_nprocs
        captured["mpi_launcher"] = mpi_launcher
        return 0

    monkeypatch.setattr(run_retrieval, "load_task_config", lambda path: config)
    monkeypatch.setattr(run_retrieval, "supervise_retrieval_command", supervise)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval.py",
            "--config",
            str(config_path),
            "--supervise-callback-failures",
        ],
    )

    assert run_retrieval.main() == 0
    assert captured["fallback_output_dir"] == tmp_path / "outputs" / "nested_sampling"
    assert captured["mpi_nprocs"] == 1
    assert captured["mpi_launcher"] is None


def test_supervision_supports_explicit_multi_rank_config(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "configuration.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    config = SimpleNamespace(
        sampler=SimpleNamespace(engine="multinest"),
        runtime=SimpleNamespace(mpi_processes=8),
        outputs=SimpleNamespace(directory=tmp_path / "outputs"),
    )
    monkeypatch.setattr(run_retrieval, "load_task_config", lambda path: config)
    monkeypatch.setenv("ROBERT_MPIEXEC", "/usr/bin/mpirun")
    monkeypatch.setenv("ROBERT_MPI_RANKS", "8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval.py",
            "--config",
            str(config_path),
            "--supervise-callback-failures",
        ],
    )

    captured = {}

    def supervise(command, *, fallback_output_dir, mpi_nprocs, mpi_launcher):
        captured["mpi_nprocs"] = mpi_nprocs
        captured["mpi_launcher"] = mpi_launcher
        return 0

    monkeypatch.setattr(run_retrieval, "supervise_retrieval_command", supervise)

    assert run_retrieval.main() == 0
    assert captured == {"mpi_nprocs": 8, "mpi_launcher": "/usr/bin/mpirun"}


def test_supervision_resolves_auto_ranks_from_run_environment(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "configuration.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    config = SimpleNamespace(
        sampler=SimpleNamespace(engine="multinest"),
        runtime=SimpleNamespace(mpi_processes="auto"),
        outputs=SimpleNamespace(directory=tmp_path / "outputs"),
    )
    captured = {}

    def supervise(command, *, fallback_output_dir, mpi_nprocs, mpi_launcher):
        captured["mpi_nprocs"] = mpi_nprocs
        captured["mpi_launcher"] = mpi_launcher
        return 0

    monkeypatch.setenv("ROBERT_MPI_RANKS", "12")
    monkeypatch.setenv("ROBERT_MPIEXEC", "/conda/bin/mpirun")
    monkeypatch.setattr(run_retrieval, "load_task_config", lambda path: config)
    monkeypatch.setattr(run_retrieval, "supervise_retrieval_command", supervise)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval.py",
            "--config",
            str(config_path),
            "--supervise-callback-failures",
        ],
    )

    assert run_retrieval.main() == 0
    assert captured == {"mpi_nprocs": 12, "mpi_launcher": "/conda/bin/mpirun"}
