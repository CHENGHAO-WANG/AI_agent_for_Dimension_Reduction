"""Launch the dr-agent plugin headless, exactly as the graded Runs are launched.

    python -m preflight.launch representation | resume | visualization
    python -m preflight.launch graded --dataset pbmc3k --run-id graded-pbmc3k

Arguments go to `claude` as a list, so no shell rewrites the prompt or the plugin path
(day 21's Git Bash traps). Each scenario runs in a new folder outside the repository, so
neither the build instructions in `.claude/` nor an earlier Run can reach the agent.
Set DRTOOLS_DATA_DIR to reuse a download cache for the graded datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from preflight.transcript import Transcript, read

REPO = Path(__file__).resolve().parents[1]
ALLOWED = ["Bash(drtools:*)", "Bash(ls:*)", "Bash(cat:*)",
           "Read", "Write", "Edit", "Glob", "Grep", "Skill"]
SCENARIOS = {
    "representation": ("wide_counts --auto", "pf-rep"),
    "resume": ("wide_counts --auto", "pf-resume"),
    "visualization": ("wide_counts --auto --purpose visualization", "pf-vis"),
}
#: Registration finished inside 25 turns on days 20 and 21, so this stops the first
#: `resume` launch partway through execution.
RESUME_FIRST_TURNS = 25


def build_command(claude: str, prompt: str, max_turns: int | None) -> list[str]:
    command = [claude, "-p", prompt,
               "--plugin-dir", REPO.as_posix(),
               "--settings", json.dumps({"disableAllHooks": True}),
               "--strict-mcp-config",
               "--disallowedTools", "PowerShell",
               "--allowedTools", *ALLOWED,
               "--output-format", "stream-json", "--verbose"]
    if max_turns:
        command += ["--max-turns", str(max_turns)]
    return command


def plugin_loaded(transcript: Transcript) -> bool:
    return "dr-agent:analyze" in transcript.init.get("slash_commands", [])


def _next_stage(run_dir: Path) -> str | None:
    from drtools.runs import RunDir
    from drtools.status import run_status

    return run_status(RunDir(run_dir, create=False))["next"] if run_dir.exists() else None


def _launch(claude: str, prompt: str, workdir: Path, name: str, run_id: str,
            env: dict, max_turns: int | None) -> dict:
    with open(workdir / name, "w", encoding="utf-8") as out, \
            open(workdir / f"{name}.err", "w", encoding="utf-8") as err:
        subprocess.run(build_command(claude, prompt, max_turns), cwd=workdir,
                       stdout=out, stderr=err, env=env, check=False)
    transcript = read(workdir / name)
    if not plugin_loaded(transcript):
        raise SystemExit(f"the dr-agent plugin did not load from {REPO.as_posix()}; "
                         f"see {workdir / name}")
    result = transcript.result or {}
    return {"transcript": name, "turns": result.get("num_turns"),
            "minutes": round((result.get("duration_ms") or 0) / 60000, 1),
            "cost_usd": result.get("total_cost_usd"),
            "next": _next_stage(workdir / "runs" / run_id)}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=[*SCENARIOS, "graded"])
    parser.add_argument("--dataset", help="the dataset spec, for a graded Run")
    parser.add_argument("--run-id", help="the Run id, for a graded Run")
    parser.add_argument("--out", type=Path, default=REPO.parent / "dr-agent-preflight")
    args = parser.parse_args(argv)

    if args.scenario == "graded":
        if not (args.dataset and args.run_id):
            parser.error("a graded Run needs --dataset and --run-id")
        arguments, run_id = f"{args.dataset} --auto", args.run_id
    else:
        arguments, run_id = SCENARIOS[args.scenario]
    prompt = f"/dr-agent:analyze {arguments} --run-id {run_id}"

    claude = shutil.which("claude")
    if claude is None:
        raise SystemExit("claude is not on PATH")
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if shutil.which("drtools", path=env["PATH"]) is None:
        raise SystemExit("drtools is not on PATH; install the toolbox into this Python")

    workdir = args.out / f"{args.scenario}-{datetime.now():%Y%m%d-%H%M%S}"
    workdir.mkdir(parents=True)
    launches = []
    if args.scenario == "resume":
        launches.append(_launch(claude, prompt, workdir, "transcript-1.jsonl", run_id, env,
                                RESUME_FIRST_TURNS))
        launches.append(_launch(claude, prompt, workdir, "transcript-2.jsonl", run_id, env, None))
    else:
        launches.append(_launch(claude, prompt, workdir, "transcript.jsonl", run_id, env, None))
    summary = {"scenario": args.scenario, "run_id": run_id, "launches": launches}
    (workdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\ncheck it with: DRAGENT_PREFLIGHT={args.out.as_posix()} pytest tests/test_preflight.py")
    return workdir


if __name__ == "__main__":
    main()
