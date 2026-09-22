import json
import shutil
from pathlib import Path

import pytest

from tools.summarize_gpu_results import build_summary, render_markdown, render_svg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPU_RESULTS = PROJECT_ROOT / "artifacts" / "gpu"


def test_checked_in_gpu_reports_produce_expected_summary():
    summary = build_summary(GPU_RESULTS)

    assert summary["environment"]["accelerator"] == "NVIDIA GeForce RTX 4090"
    assert summary["suite"]["cells_completed"] == 9
    assert summary["suite"]["duration_seconds"] == pytest.approx(258.587)
    assert summary["model"]["parameters"] == 63_912_192
    assert summary["compile"]["throughput_improvement_percent"] == pytest.approx(64.16)
    assert summary["compile"]["estimated_break_even_steps"] == pytest.approx(1228.9)
    assert summary["pipeline"]["final_validation_perplexity"] == pytest.approx(430.0795)
    assert summary["pipeline"]["perplexity_is_indicative_only"] is False
    assert summary["kvcache"]["all_required_correctness_passed"] is True
    assert summary["kvcache"]["max_peak_gpu_memory_reduction_percent"] == pytest.approx(16.26)


def test_report_renderers_include_scope_and_tradeoffs():
    summary = build_summary(GPU_RESULTS)
    markdown = render_markdown(summary)
    svg = render_svg(summary)

    assert "estimated break-even point" in markdown
    assert "not claim a fully converged chat model" in markdown
    assert "not a universal speedup" in markdown
    assert "MiniMind 63.9M" in svg
    assert svg.startswith("<svg")


def test_failed_required_cache_correctness_is_rejected(tmp_path):
    copied = tmp_path / "gpu"
    shutil.copytree(GPU_RESULTS, copied)
    report_path = copied / "kvcache" / "ctx512_bs8.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["required_correctness"]["static_matches_dynamic"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="correctness failed"):
        build_summary(copied)


def test_incomplete_suite_is_rejected(tmp_path):
    copied = tmp_path / "gpu"
    shutil.copytree(GPU_RESULTS, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cells"]["pipeline__pretrain"]["status"] = "interrupted"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="only completed cells"):
        build_summary(copied)
