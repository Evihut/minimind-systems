"""One-command GPU experiment suite: resumable, budget-capped, JSON per cell.

Every experiment is a *cell* that runs as its own subprocess and writes one JSON
report. A manifest records what finished, so re-running the suite after a crash,
a Ctrl-C, or an expired rental picks up where it stopped instead of repeating
paid GPU time.

Typical use on a rented box::

    python -m benchmarks.gpu_suite --device cuda --time-budget-minutes 360

and to rehearse the whole flow locally for free::

    python -m benchmarks.gpu_suite --device cpu --quick
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.common import PROJECT_ROOT

MANIFEST_SCHEMA_VERSION = 1
GROUPS = ("compile", "kvcache", "pipeline", "ddp")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def suite_compile_backend(device: str, requested: str | None) -> str:
    if requested:
        return requested
    return "inductor" if device.startswith("cuda") else "aot_eager"


@dataclass
class Cell:
    """A single experiment: one subprocess, one JSON report."""

    cell_id: str
    group: str
    command: list[str]
    output: Path
    description: str = ""
    skip_reason: str | None = None
    estimated_seconds: float = 60.0


@dataclass
class SuiteState:
    interrupted: bool = False
    cells: dict = field(default_factory=dict)


def module_command(module: str, options: dict) -> list[str]:
    command = [sys.executable, "-m", module]
    for key, value in options.items():
        if value is None or value is False:
            continue
        if value is True:
            command.append(key)
        else:
            command.extend([key, str(value)])
    return command


def distributed_command(nproc: int, port: int, module: str, options: dict) -> list[str]:
    launcher = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        "-m",
        module,
    ]
    return launcher + module_command(module, options)[3:]


# --------------------------------------------------------------------------- #
# Experiment plan
# --------------------------------------------------------------------------- #

def build_plan(args: argparse.Namespace) -> list[Cell]:
    root = args.output_dir
    preset = args.preset
    device = args.device
    attn = args.attn
    cells: list[Cell] = []

    if args.quick:
        contexts = [32, 64]
        batches = [1, 2]
        new_tokens, repeats, warmup = 8, 2, 1
        train_steps, train_warmup, seq_len = 4, 1, 32
    else:
        contexts = [128, 512, 1024]
        batches = [1, 8]
        new_tokens, repeats, warmup = args.max_new_tokens, args.repeats, args.warmup
        train_steps, train_warmup, seq_len = args.steps, args.warmup_steps, args.sequence_length

    # --- Group 1: eager vs torch.compile/inductor -------------------------- #
    for label, compiled in (("eager", False), (f"compile_{args.compile_backend}", True)):
        output = root / "compile" / f"{label}.json"
        cells.append(
            Cell(
                cell_id=f"compile__{label}",
                group="compile",
                description=f"{preset} {args.precision}, {label}",
                command=module_command(
                    "benchmarks.training_benchmark",
                    {
                        "--preset": preset,
                        "--device": device,
                        "--precision": args.precision,
                        "--attn": attn,
                        "--steps": train_steps,
                        "--warmup-steps": train_warmup,
                        "--batch-size": args.batch_size,
                        "--sequence-length": seq_len,
                        "--compile": compiled,
                        "--compile-backend": args.compile_backend if compiled else None,
                        "--output": output,
                    },
                ),
                output=output,
                estimated_seconds=240.0 if compiled else 90.0,
            )
        )

    # --- Group 2: KV cache across context x batch -------------------------- #
    # One anchor cell keeps no_cache in the matrix as the correctness ground
    # truth; the rest skip it because it is quadratic and would dominate cost.
    anchor_context, anchor_batch = contexts[0], batches[0]
    for context in contexts:
        for batch in batches:
            is_anchor = context == anchor_context and batch == anchor_batch
            variants = "no_cache,dynamic_cache,static_cache" if is_anchor else "dynamic_cache,static_cache"
            label = f"ctx{context}_bs{batch}"
            output = root / "kvcache" / f"{label}.json"
            cells.append(
                Cell(
                    cell_id=f"kvcache__{label}",
                    group="kvcache",
                    description=f"context {context}, batch {batch}"
                    + (" (correctness anchor, includes no_cache)" if is_anchor else ""),
                    command=module_command(
                        "benchmarks.inference_benchmark",
                        {
                            "--preset": preset,
                            "--device": device,
                            "--attn": attn,
                            "--precision": args.precision,
                            "--prompt-tokens": context,
                            "--batch-size": batch,
                            "--max-new-tokens": new_tokens,
                            "--warmup": warmup,
                            "--repeats": repeats,
                            "--variants": variants,
                            "--output": output,
                        },
                    ),
                    output=output,
                    estimated_seconds=60.0 + context * batch * 0.05,
                )
            )

    # --- Group 3: short pretrain on real data ------------------------------ #
    # Refused rather than silently downgraded: the built-in corpus is 32
    # repeated sentences, so its perplexity says nothing about model quality.
    train_path = Path(args.train_data).resolve() if args.train_data is not None else None
    validation_path = (
        Path(args.validation_data).resolve() if args.validation_data is not None else None
    )
    if train_path is None or not train_path.exists():
        pipeline_skip_reason = (
            "--train-data not provided or missing; refusing to report perplexity "
            "from the 32-sentence built-in corpus"
        )
    elif validation_path is None or not validation_path.exists():
        pipeline_skip_reason = (
            "--validation-data not provided or missing; held-out validation is required"
        )
    elif validation_path == train_path:
        pipeline_skip_reason = (
            "--validation-data must be a different file from --train-data to prevent leakage"
        )
    else:
        pipeline_skip_reason = None
    pipeline_output = root / "pipeline" / "pretrain.json"
    cells.append(
        Cell(
            cell_id="pipeline__pretrain",
            group="pipeline",
            description="short pretrain on a real corpus with held-out eval",
            command=module_command(
                "benchmarks.training_benchmark",
                {
                    "--preset": preset,
                    "--device": device,
                    "--precision": args.precision,
                    "--attn": attn,
                    "--steps": train_steps * 10,
                    "--warmup-steps": train_warmup,
                    "--batch-size": args.batch_size,
                    "--sequence-length": seq_len,
                    "--train-data": args.train_data,
                    "--validation-data": args.validation_data,
                    "--text-field": args.text_field,
                    "--save-model": root / "pipeline" / "checkpoint",
                    "--output": pipeline_output,
                },
            ),
            output=pipeline_output,
            skip_reason=pipeline_skip_reason,
            estimated_seconds=600.0,
        )
    )

    # --- Group 4: DDP strong scaling (optional, lowest priority) ----------- #
    if args.ddp_ranks:
        for ranks in args.ddp_ranks:
            label = f"world{ranks}"
            output = root / "ddp" / f"{label}.json"
            cells.append(
                Cell(
                    cell_id=f"ddp__{label}",
                    group="ddp",
                    description=f"strong scaling, global batch {args.global_batch_size}, {ranks} rank(s)",
                    command=distributed_command(
                        ranks,
                        args.master_port,
                        "benchmarks.training_benchmark",
                        {
                            "--preset": preset,
                            "--device": device,
                            "--precision": args.precision,
                            "--attn": attn,
                            "--steps": train_steps,
                            "--warmup-steps": train_warmup,
                            "--global-batch-size": args.global_batch_size,
                            "--sequence-length": seq_len,
                            "--output": output,
                        },
                    ),
                    output=output,
                    estimated_seconds=120.0 * ranks,
                )
            )

    selected = set(args.groups)
    return [cell for cell in cells if cell.group in selected]


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #

def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "created_at": utc_now(), "cells": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "created_at": utc_now(), "cells": {}}
    manifest.setdefault("cells", {})
    return manifest


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written through a temp file so an interrupt cannot leave a half-file that
    # would lose the record of everything already paid for.
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def is_complete(cell: Cell, record: dict | None) -> bool:
    """A cell counts as done only if the same command produced a readable report."""
    if not record or record.get("status") != "done":
        return False
    if record.get("command") != [str(part) for part in cell.command]:
        return False
    output = Path(record.get("output", ""))
    if not output.exists():
        return False
    try:
        json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def run_cell(cell: Cell, log_dir: Path, state: SuiteState) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cell.cell_id}.log"
    cell.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(part) for part in cell.command]
    started_at = utc_now()
    started = time.perf_counter()
    record = {
        "group": cell.group,
        "description": cell.description,
        "command": command,
        "output": str(cell.output),
        "log": str(log_path),
        "started_at": started_at,
    }
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# {' '.join(command)}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            state.interrupted = True
            process.terminate()
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
            record.update(
                status="interrupted",
                returncode=returncode,
                duration_seconds=round(time.perf_counter() - started, 3),
                finished_at=utc_now(),
            )
            return record

    record.update(
        returncode=returncode,
        duration_seconds=round(time.perf_counter() - started, 3),
        finished_at=utc_now(),
    )
    if returncode == 0 and cell.output.exists():
        record["status"] = "done"
    else:
        record["status"] = "failed"
        record["error"] = f"exit code {returncode}; see {log_path}"
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", default="64m")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--attn", choices=["manual", "sdpa"], default="sdpa")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=["compile", "kvcache", "pipeline"])
    parser.add_argument(
        "--compile-backend",
        default=None,
        help="defaults to inductor on CUDA and aot_eager for CPU rehearsal",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--train-data", type=Path, default=None)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--ddp-ranks", nargs="*", type=int, default=[])
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--master-port", type=int, default=29533)
    parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=None,
        help="stop launching new cells once this much wall time has passed",
    )
    parser.add_argument("--force", action="store_true", help="re-run cells already recorded as done")
    parser.add_argument("--only", nargs="*", default=[], help="run only these cell ids")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="tiny configuration for rehearsing the flow")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "gpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.compile_backend = suite_compile_backend(args.device, args.compile_backend)
    if args.quick and args.preset == "64m":
        args.preset = "smoke"
    plan = build_plan(args)
    if args.only:
        plan = [cell for cell in plan if cell.cell_id in set(args.only)]
        if not plan:
            raise SystemExit(f"--only matched no cells; available: {[cell.cell_id for cell in build_plan(args)]}")

    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest["suite"] = {
        "device": args.device,
        "preset": args.preset,
        "precision": args.precision,
        "attn": args.attn,
        "groups": list(args.groups),
        "quick": args.quick,
    }

    if args.dry_run:
        print(f"{len(plan)} cell(s) planned; output under {args.output_dir}")
        total = 0.0
        for cell in plan:
            record = manifest["cells"].get(cell.cell_id)
            if cell.skip_reason:
                status = "SKIP"
            elif not args.force and is_complete(cell, record):
                status = "DONE"
            else:
                status = "RUN"
                total += cell.estimated_seconds
            print(f"  [{status:4s}] {cell.cell_id:34s} {cell.description}")
            if cell.skip_reason:
                print(f"           reason: {cell.skip_reason}")
        print(f"estimated remaining: {total / 60:.1f} min (rough)")
        return 0

    state = SuiteState()

    def handle_sigint(signum, frame):  # noqa: ARG001
        state.interrupted = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    suite_started = time.perf_counter()
    budget_seconds = args.time_budget_minutes * 60 if args.time_budget_minutes else None
    counts = {"done": 0, "failed": 0, "skipped": 0, "reused": 0, "not_run": 0}

    for cell in plan:
        record = manifest["cells"].get(cell.cell_id)

        if cell.skip_reason:
            manifest["cells"][cell.cell_id] = {
                "group": cell.group,
                "description": cell.description,
                "status": "skipped",
                "skip_reason": cell.skip_reason,
                "finished_at": utc_now(),
            }
            counts["skipped"] += 1
            print(f"[skip] {cell.cell_id}: {cell.skip_reason}")
            save_manifest(manifest_path, manifest)
            continue

        if not args.force and is_complete(cell, record):
            counts["reused"] += 1
            print(f"[keep] {cell.cell_id} already complete")
            continue

        elapsed = time.perf_counter() - suite_started
        if budget_seconds is not None and elapsed >= budget_seconds:
            counts["not_run"] += 1
            print(f"[stop] time budget reached before {cell.cell_id}; re-run to continue")
            continue

        if state.interrupted:
            counts["not_run"] += 1
            continue

        print(f"[run ] {cell.cell_id}: {cell.description}")
        result = run_cell(cell, args.output_dir / "logs", state)
        manifest["cells"][cell.cell_id] = result
        save_manifest(manifest_path, manifest)

        if result["status"] == "done":
            counts["done"] += 1
            print(f"       ok in {result['duration_seconds']:.1f}s -> {result['output']}")
        elif result["status"] == "interrupted":
            print(f"       interrupted; re-run to resume ({result['log']})")
            break
        else:
            counts["failed"] += 1
            print(f"       FAILED: {result.get('error')}")

    save_manifest(manifest_path, manifest)
    print(
        f"\nsummary: {counts['done']} run, {counts['reused']} reused, "
        f"{counts['failed']} failed, {counts['skipped']} skipped, {counts['not_run']} not reached"
    )
    print(f"manifest: {manifest_path}")
    if state.interrupted:
        return 130
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
