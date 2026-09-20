"""End-to-end checks for the benchmark axes the GPU suite depends on."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarks.inference_benchmark import correctness_requirements

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str, *arguments: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{module} failed:\n{result.stdout}\n{result.stderr}"


@pytest.fixture(scope="module")
def batched_inference_report(tmp_path_factory) -> dict:
    output = tmp_path_factory.mktemp("inference") / "report.json"
    run_module(
        "benchmarks.inference_benchmark",
        "--device", "cpu",
        "--precision", "bf16",
        "--attn", "sdpa",
        "--prompt-tokens", "16",
        "--max-new-tokens", "4",
        "--warmup", "1",
        "--repeats", "2",
        "--batch-size", "3",
        "--variants", "dynamic_cache,static_cache",
        "--output", str(output),
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_batch_size_reaches_the_workload_record(batched_inference_report):
    assert batched_inference_report["workload"]["batch_size"] == 3
    assert all(variant["batch_size"] == 3 for variant in batched_inference_report["variants"])


def test_throughput_counts_every_sequence_in_the_batch(batched_inference_report):
    # repeats(2) * max_new_tokens(4) * batch(3)
    expected = 2 * 4 * 3
    assert all(
        variant["generated_tokens"] == expected for variant in batched_inference_report["variants"]
    )


def test_partial_matrix_falls_back_to_an_available_correctness_reference(batched_inference_report):
    assert batched_inference_report["correctness_reference"] == "dynamic_cache"
    assert batched_inference_report["correctness_matches_reference"]["static_cache"] is True
    assert "static_vs_dynamic" in batched_inference_report["comparisons"]
    assert "dynamic_vs_no_cache" not in batched_inference_report["comparisons"]


def test_bf16_inference_uses_bf16_parameters_and_enforces_cache_equivalence(
    batched_inference_report,
):
    assert batched_inference_report["parameter_dtype"] == "bfloat16"
    assert batched_inference_report["required_correctness"] == {
        "static_matches_dynamic": True
    }
    assert batched_inference_report["correctness_is_enforced"] is True


def test_bf16_cache_divergence_is_a_required_failure():
    outputs = {
        "dynamic_cache": torch.tensor([[1, 2, 3]]),
        "static_cache": torch.tensor([[1, 2, 4]]),
    }
    correctness = {"dynamic_cache": True, "static_cache": False}
    required = correctness_requirements(outputs, correctness, "dynamic_cache", "bf16")
    assert required == {"static_matches_dynamic": False}


def test_unknown_variant_is_rejected(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.inference_benchmark",
            "--device", "cpu", "--variants", "turbo_cache",
            "--output", str(tmp_path / "x.json"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown variants" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "expected_mode", "expected_per_rank"),
    [
        (["--batch-size", "4"], "weak", 4),
        (["--global-batch-size", "6"], "strong", 6),
    ],
)
def test_scaling_mode_is_recorded(tmp_path, arguments, expected_mode, expected_per_rank):
    output = tmp_path / "training.json"
    run_module(
        "benchmarks.training_benchmark",
        "--device", "cpu",
        "--steps", "2",
        "--sequence-length", "32",
        *arguments,
        "--output", str(output),
    )
    training = json.loads(output.read_text(encoding="utf-8"))["training"]
    assert training["scaling_mode"] == expected_mode
    assert training["per_rank_batch_size"] == expected_per_rank


def test_toy_corpus_perplexity_is_flagged(tmp_path):
    output = tmp_path / "training.json"
    run_module(
        "benchmarks.training_benchmark",
        "--device", "cpu", "--steps", "2", "--batch-size", "2",
        "--sequence-length", "32", "--output", str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data"]["corpus_is_toy"] is True
    assert report["validation"]["perplexity_is_indicative_only"] is True


def test_real_corpus_clears_the_flag(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(
        "\n".join(json.dumps({"text": f"sample sentence number {index}"}) for index in range(16)),
        encoding="utf-8",
    )
    validation.write_text(
        "\n".join(json.dumps({"text": f"held out sentence {index}"}) for index in range(8)),
        encoding="utf-8",
    )
    output = tmp_path / "training.json"
    run_module(
        "benchmarks.training_benchmark",
        "--device", "cpu", "--steps", "2", "--batch-size", "2",
        "--sequence-length", "32",
        "--train-data", str(train),
        "--validation-data", str(validation),
        "--output", str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data"]["corpus_is_toy"] is False
    assert report["data"]["validation_is_toy"] is False
    assert report["validation"]["perplexity_is_indicative_only"] is False


def test_training_rejects_validation_leakage(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "same file"}\n', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.training_benchmark",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--train-data",
            str(corpus),
            "--validation-data",
            str(corpus),
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be different files" in result.stderr


def test_step_percentiles_are_reported(tmp_path):
    output = tmp_path / "training.json"
    run_module(
        "benchmarks.training_benchmark",
        "--device", "cpu", "--steps", "4", "--warmup-steps", "1",
        "--batch-size", "2", "--sequence-length", "32", "--output", str(output),
    )
    training = json.loads(output.read_text(encoding="utf-8"))["training"]
    assert training["step_seconds_p50_ms"] > 0
    assert training["step_seconds_p95_ms"] >= training["step_seconds_p50_ms"]
    assert training["warmup_steps"] == 1
