import argparse
import json
from pathlib import Path

import pytest

from benchmarks.gpu_suite import (
    Cell,
    build_plan,
    is_complete,
    load_manifest,
    save_manifest,
    suite_compile_backend,
)


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        device="cpu",
        preset="smoke",
        precision="bf16",
        attn="sdpa",
        groups=["compile", "kvcache", "pipeline"],
        compile_backend="inductor",
        steps=4,
        warmup_steps=1,
        batch_size=2,
        sequence_length=32,
        max_new_tokens=8,
        repeats=2,
        warmup=1,
        train_data=None,
        validation_data=None,
        text_field="text",
        ddp_ranks=[],
        global_batch_size=8,
        master_port=29533,
        quick=True,
        output_dir=Path("/tmp/does-not-need-to-exist"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_plan_covers_the_context_by_batch_matrix():
    cells = build_plan(make_args(groups=["kvcache"]))
    ids = {cell.cell_id for cell in cells}
    assert ids == {
        "kvcache__ctx32_bs1",
        "kvcache__ctx32_bs2",
        "kvcache__ctx64_bs1",
        "kvcache__ctx64_bs2",
    }


def test_only_the_anchor_cell_pays_for_the_no_cache_baseline():
    cells = {cell.cell_id: cell for cell in build_plan(make_args(groups=["kvcache"]))}
    anchor = " ".join(cells["kvcache__ctx32_bs1"].command)
    other = " ".join(cells["kvcache__ctx64_bs2"].command)
    assert "no_cache,dynamic_cache,static_cache" in anchor
    assert "no_cache" not in other


def test_kvcache_cells_are_measured_at_the_requested_precision():
    cells = build_plan(make_args(groups=["kvcache"], precision="bf16"))
    for cell in cells:
        assert "--precision bf16" in " ".join(str(part) for part in cell.command)


def test_pipeline_is_skipped_rather_than_run_on_the_toy_corpus():
    (cell,) = build_plan(make_args(groups=["pipeline"]))
    assert cell.skip_reason is not None
    assert "built-in corpus" in cell.skip_reason


def test_pipeline_runs_once_real_data_exists(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"text": "hello"}\n', encoding="utf-8")
    validation.write_text('{"text": "held out"}\n', encoding="utf-8")
    (cell,) = build_plan(
        make_args(groups=["pipeline"], train_data=train, validation_data=validation)
    )
    assert cell.skip_reason is None
    rendered = " ".join(str(part) for part in cell.command)
    assert str(train) in rendered
    assert str(validation) in rendered


def test_pipeline_requires_a_separate_holdout(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"text": "hello"}\n', encoding="utf-8")

    (missing,) = build_plan(make_args(groups=["pipeline"], train_data=train))
    assert "held-out validation is required" in missing.skip_reason

    (leaked,) = build_plan(
        make_args(groups=["pipeline"], train_data=train, validation_data=train)
    )
    assert "prevent leakage" in leaked.skip_reason


def test_compile_group_pairs_eager_against_the_chosen_backend():
    cells = build_plan(make_args(groups=["compile"]))
    assert [cell.cell_id for cell in cells] == ["compile__eager", "compile__compile_inductor"]
    eager, compiled = (" ".join(str(part) for part in cell.command) for cell in cells)
    assert "--compile" not in eager
    assert "--compile --compile-backend inductor" in compiled


def test_compile_backend_is_safe_for_rehearsal_and_real_for_cuda():
    assert suite_compile_backend("cpu", None) == "aot_eager"
    assert suite_compile_backend("cuda", None) == "inductor"
    assert suite_compile_backend("cuda:0", None) == "inductor"
    assert suite_compile_backend("cpu", "eager") == "eager"


def test_ddp_cells_pin_the_global_batch_for_strong_scaling():
    cells = build_plan(make_args(groups=["ddp"], ddp_ranks=[1, 2]))
    assert [cell.cell_id for cell in cells] == ["ddp__world1", "ddp__world2"]
    for cell, ranks in zip(cells, (1, 2), strict=True):
        rendered = " ".join(str(part) for part in cell.command)
        assert f"--nproc_per_node={ranks}" in rendered
        assert "--global-batch-size 8" in rendered
        assert "--batch-size" not in rendered


def completed_record(cell: Cell, output: Path) -> dict:
    return {
        "status": "done",
        "command": [str(part) for part in cell.command],
        "output": str(output),
    }


def cell_for(output: Path) -> Cell:
    return Cell(cell_id="c", group="kvcache", command=["python", "-m", "x", "--output", str(output)], output=output)


def test_completed_cell_is_reused(tmp_path):
    output = tmp_path / "report.json"
    output.write_text(json.dumps({"ok": True}), encoding="utf-8")
    cell = cell_for(output)
    assert is_complete(cell, completed_record(cell, output))


def test_changed_command_invalidates_a_completed_cell(tmp_path):
    output = tmp_path / "report.json"
    output.write_text(json.dumps({"ok": True}), encoding="utf-8")
    cell = cell_for(output)
    record = completed_record(cell, output)
    record["command"] = [*record["command"], "--repeats", "99"]
    assert not is_complete(cell, record)


def test_missing_or_corrupt_artifact_invalidates_a_completed_cell(tmp_path):
    output = tmp_path / "report.json"
    cell = cell_for(output)
    record = completed_record(cell, output)
    assert not is_complete(cell, record), "missing artifact must not count as done"
    output.write_text("{not json", encoding="utf-8")
    assert not is_complete(cell, record), "unreadable artifact must not count as done"


@pytest.mark.parametrize("status", ["failed", "interrupted", "skipped", "pending"])
def test_only_done_cells_are_reused(tmp_path, status):
    output = tmp_path / "report.json"
    output.write_text(json.dumps({"ok": True}), encoding="utf-8")
    cell = cell_for(output)
    record = completed_record(cell, output) | {"status": status}
    assert not is_complete(cell, record)


def test_manifest_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{truncated", encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest["cells"] == {}


def test_manifest_round_trips(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    save_manifest(path, {"cells": {"a": {"status": "done"}}})
    assert load_manifest(path)["cells"]["a"]["status"] == "done"
    assert not path.with_suffix(".json.tmp").exists(), "temp file must not be left behind"
