"""Tests for serial callback-failure supervision."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from robert_exoplanets.core import RobertConfigError
from robert_exoplanets.retrieval.samplers.callback_supervisor import (
    CALLBACK_FAILURE_MARKER_ENV,
    supervise_retrieval_command,
    _rank_process_matches,
)


def test_supervisor_stops_child_marks_failure_and_keeps_partial_chains(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run" / "multinest"
    chains = output_dir / "chains"
    chains.mkdir(parents=True)
    partial_chain = chains / "1-resume.dat"
    partial_chain.write_text("partial chain data", encoding="utf-8")
    started = tmp_path / "started.txt"
    completed = tmp_path / "completed.txt"
    child = tmp_path / "fake_retrieval.py"
    child.write_text(
        "import json, os, pathlib, time\n"
        "marker = pathlib.Path(os.environ['ROBERT_MULTINEST_CALLBACK_FAILURE_MARKER'])\n"
        f"output = pathlib.Path({str(output_dir)!r})\n"
        "payload = {\n"
        "  'schema_version': 1,\n"
        "  'output_dir': str(output),\n"
        "  'status_base': {'attempt_id': 'attempt-1', 'state': 'running'},\n"
        "  'elapsed_seconds': 3.0,\n"
        "  'exception_type': 'RuntimeError',\n"
        "  'message': 'callback failed',\n"
        "  'traceback': 'callback traceback',\n"
        "  'rank': 0,\n"
        "}\n"
        "temporary = marker.with_suffix('.tmp')\n"
        "temporary.write_text(json.dumps(payload), encoding='utf-8')\n"
        "temporary.replace(marker)\n"
        f"pathlib.Path({str(started)!r}).write_text('started', encoding='utf-8')\n"
        "time.sleep(1.0)\n"
        f"pathlib.Path({str(completed)!r}).write_text('completed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    return_code = supervise_retrieval_command(
        [sys.executable, str(child)],
        fallback_output_dir=output_dir,
        poll_interval_seconds=0.01,
        terminate_timeout_seconds=1.0,
    )

    assert return_code == 1
    assert started.read_text(encoding="utf-8") == "started"
    assert not completed.exists()
    assert partial_chain.read_text(encoding="utf-8") == "partial chain data"
    status = json.loads((output_dir / "sampler_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["callback_failure"] is True
    assert status["multinest_resume_safe"] is False
    assert status["partial_chains_preserved"] is True
    assert not (output_dir / "result.json").exists()
    retained_marker = output_dir / "multinest_callback_failure_attempt-1.json"
    assert json.loads(retained_marker.read_text(encoding="utf-8"))["message"] == (
        "callback failed"
    )


def test_supervisor_uses_a_fresh_marker_and_ignores_stale_marker(
    tmp_path: Path,
) -> None:
    stale_marker = tmp_path / "stale-marker.json"
    stale_marker.write_text('{"message":"old failure"}', encoding="utf-8")
    observed_marker = tmp_path / "observed-marker.txt"
    child = tmp_path / "successful_retrieval.py"
    child.write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(observed_marker)!r}).write_text("
        "os.environ['ROBERT_MULTINEST_CALLBACK_FAILURE_MARKER'], encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment[CALLBACK_FAILURE_MARKER_ENV] = str(stale_marker)

    return_code = supervise_retrieval_command(
        [sys.executable, str(child)],
        fallback_output_dir=tmp_path / "run" / "multinest",
        environment=environment,
        poll_interval_seconds=0.01,
    )

    assert return_code == 0
    assert observed_marker.read_text(encoding="utf-8") != str(stale_marker)
    assert stale_marker.read_text(encoding="utf-8") == '{"message":"old failure"}'
    assert not (tmp_path / "run" / "multinest" / "sampler_status.json").exists()


def test_supervisor_marks_marker_write_failure_after_child_exit(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run" / "multinest"
    output_dir.mkdir(parents=True)
    (output_dir / "sampler_status.json").write_text(
        json.dumps({"attempt_id": "attempt-2", "state": "running"}),
        encoding="utf-8",
    )
    child = tmp_path / "marker_failure.py"
    child.write_text("import os\nos._exit(70)\n", encoding="utf-8")

    return_code = supervise_retrieval_command(
        [sys.executable, str(child)],
        fallback_output_dir=output_dir,
        poll_interval_seconds=0.01,
    )

    assert return_code == 1
    status = json.loads((output_dir / "sampler_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["exception_type"] == "CallbackMarkerWriteError"
    assert status["multinest_resume_safe"] is False


def test_supervisor_rejects_mpi_launcher_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PMI_SIZE", "4")

    with pytest.raises(RobertConfigError, match="outside an MPI world"):
        supervise_retrieval_command(
            [sys.executable, "-c", "pass"],
            fallback_output_dir=tmp_path,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="MPI cleanup uses Linux /proc identities")
def test_mpi_supervisor_stops_detached_nonroot_rank_and_preserves_diagnostics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run" / "multinest"
    chains = output_dir / "chains"
    chains.mkdir(parents=True)
    partial_chain = chains / "1-resume.dat"
    partial_chain.write_text("partial chain data", encoding="utf-8")
    worker = tmp_path / "fake_rank.py"
    worker.write_text(
        "import os, pathlib, time\n"
        "from robert_exoplanets.retrieval.samplers.callback_supervisor import "
        "write_callback_rank_record\n"
        "directory = pathlib.Path(os.environ['ROBERT_MULTINEST_CALLBACK_FAILURE_MARKER_DIR'])\n"
        "rank = int(os.environ['FAKE_MPI_RANK'])\n"
        "size = int(os.environ['ROBERT_MPI_RANKS'])\n"
        "if rank == 0: pathlib.Path(os.environ['FAKE_RANK_ZERO_READY']).write_text('ready')\n"
        "write_callback_rank_record(directory, 'ready', rank)\n"
        "deadline = time.monotonic() + 10\n"
        "while len(list(directory.glob('ready-rank-*-pid-*.json'))) < size:\n"
        "    if time.monotonic() >= deadline: raise RuntimeError('ready barrier timed out')\n"
        "    time.sleep(0.01)\n"
        "if rank == 1:\n"
        "    write_callback_rank_record(directory, 'failure', rank, {\n"
        "        'exception_type': 'RuntimeError', 'message': 'non-root failed',\n"
        "        'traceback': 'fake traceback', 'elapsed_seconds': 2.0})\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    launcher = tmp_path / "fake_mpirun.py"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('mpirun (Open MPI) 5.0.10'); raise SystemExit(0)\n"
        "if sys.argv[1] != '-n': raise SystemExit(2)\n"
        "size = int(sys.argv[2]); command = sys.argv[3:]\n"
        "children = []\n"
        "for rank in range(size):\n"
        "    env = dict(os.environ, FAKE_MPI_RANK=str(rank))\n"
        "    children.append(subprocess.Popen(command, env=env, start_new_session=True))\n"
        "raise SystemExit(max(child.wait() for child in children))\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    environment = {
        **os.environ,
        "SLURM_NNODES": "1",
        "FAKE_RANK_ZERO_READY": str(tmp_path / "rank-zero-ready.txt"),
    }
    for name in ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "PMI_RANK", "PMI_SIZE", "PMIX_RANK", "PMIX_SIZE"):
        environment.pop(name, None)

    return_code = supervise_retrieval_command(
        [sys.executable, str(worker)],
        fallback_output_dir=output_dir,
        mpi_nprocs=2,
        mpi_launcher=launcher,
        environment=environment,
        poll_interval_seconds=0.02,
        terminate_timeout_seconds=0.5,
        startup_timeout_seconds=10.0,
    )

    assert return_code == 1
    assert (tmp_path / "rank-zero-ready.txt").is_file()
    assert partial_chain.read_text(encoding="utf-8") == "partial chain data"
    status = json.loads((output_dir / "sampler_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["callback_failure"] is True
    assert status["callback_rank"] == 1
    assert status["callback_failure_ranks"] == [1]
    assert status["callback_supervisor_cleanup_verified"] is True
    assert status["multinest_resume_safe"] is False
    assert status["partial_chains_preserved"] is True
    assert not (output_dir / "result.json").exists()
    retained = json.loads(
        (output_dir / f"multinest_callback_failure_{status['attempt_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert retained["failures"][0]["message"] == "non-root failed"
    for marker_path in Path(retained["marker_directory"]).glob("ready-rank-*.json"):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert not _rank_process_matches(marker)


def test_mpi_supervisor_rejects_multi_node_allocations(tmp_path: Path) -> None:
    with pytest.raises(RobertConfigError, match="one-node"):
        supervise_retrieval_command(
            [sys.executable, "-c", "pass"],
            fallback_output_dir=tmp_path,
            mpi_nprocs=2,
            mpi_launcher="mpirun",
            environment={"SLURM_NNODES": "2"},
        )
