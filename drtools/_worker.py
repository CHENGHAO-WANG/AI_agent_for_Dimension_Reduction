"""Runs one candidate pipeline in its own process.

Invoked as `python -m drtools._worker <job.json>`, never called directly. Isolation
exists because the failure modes here are not all catchable: MDS on a large matrix can
take the interpreter down with an allocation the OS refuses, a BLAS thread can abort the
process, and a method that has not converged can simply run for an hour. A separate
process can be killed; an in-process call cannot.

Success and expected failure both leave a record and exit cleanly. Anything that kills
the process outright leaves no record, and the parent synthesises one from the exit
status — which is the case this design exists for.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

from drtools import jsonio
from drtools.cache import read_cache
from drtools.executors import ExecutionError
from drtools.pipeline import PipelineError, failure_record, run_pipeline, save_embedding
from drtools.runs import RunDir

EXIT_OK = 0
EXIT_EXPECTED_FAILURE = 1
EXIT_UNEXPECTED_FAILURE = 2


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m drtools._worker <job.json>", file=sys.stderr)
        return EXIT_UNEXPECTED_FAILURE

    job = jsonio.read(Path(argv[0]))
    run = RunDir(Path(job["run_dir"]))
    candidate_id = job["id"]
    record_path = run.path / "embeddings" / f"{candidate_id}.json"

    started = time.perf_counter()
    try:
        X, labels, meta = read_cache(run)
        result, tuning = execute(run, X, labels, job["stages"], job.get("seed", 0),
                                 job.get("tuning"))
    except (ExecutionError, PipelineError) as error:
        jsonio.write(
            record_path,
            {
                "id": candidate_id,
                "status": "failed",
                "duration_s": round(time.perf_counter() - started, 3),
                "stages_requested": job["stages"],
                "failure": (
                    failure_record(error)
                    if isinstance(error, ExecutionError)
                    else {"op": None, "error_type": "PipelineError", "message": str(error)}
                ),
            },
        )
        return EXIT_EXPECTED_FAILURE
    except Exception as error:  # noqa: BLE001 - the point is to record anything
        jsonio.write(
            record_path,
            {
                "id": candidate_id,
                "status": "failed",
                "duration_s": round(time.perf_counter() - started, 3),
                "stages_requested": job["stages"],
                "failure": {
                    "op": None,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback_tail": traceback.format_exc().strip().splitlines()[-3:],
                },
            },
        )
        return EXIT_UNEXPECTED_FAILURE

    embeddings = run.path / "embeddings"
    # The fitted objects stay in this process: a candidate that subsampled has already
    # projected its other rows here, so nothing fitted crosses a process boundary.
    save_embedding(embeddings, candidate_id, result)

    jsonio.write(
        record_path,
        {
            "id": candidate_id,
            "status": "ok",
            "run_id": run.id,
            "dataset": meta.get("name"),
            "stages_requested": job["stages"],
            "seed": job.get("seed", 0),
            "embedding_path": str(embeddings / f"{candidate_id}.npy"),
            **result.as_dict(),
            **({"tuning": tuning} if tuning is not None else {}),
        },
    )
    return EXIT_OK


def execute(
    run: RunDir,
    X: Any,
    labels: Any,
    stages: list[dict[str, Any]],
    seed: int,
    tuning: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any] | None]:
    """Run one Attempt: tuned when the job carries a tuning payload, as given if not.

    Shared with `embed --in-process`, so the two routes cannot tune differently.
    """
    if tuning is None:
        return run_pipeline(X, labels, stages, seed=seed), None
    from drtools.tuning import TuningSettings, tune_and_refit

    profile = run.read_artifact("profile.json") if run.profile_path.exists() else {}
    recon = run.read_artifact("recon.json") if run.recon_path.exists() else None
    return tune_and_refit(
        X, labels, stages,
        n_base=int(tuning["n_base"]),
        seed=seed,
        weights=dict(tuning["weights"]),
        settings=TuningSettings.from_plan(tuning.get("settings")),
        profile=profile,
        recon=recon,
    )


if __name__ == "__main__":
    raise SystemExit(main())
