.PHONY: help setup-gpu test lint smoke benchmark-inference benchmark-training gpu-suite gpu-suite-plan gpu-suite-rehearse gpu-report dataset-sample verify

PYTHON ?= python3
DEVICE ?= cpu

help:
	@echo "make test    - run the CPU test suite"
	@echo "make lint    - lint portfolio code and tests"
	@echo "make setup-gpu - install the project and verification dependencies"
	@echo "make smoke   - run the end-to-end tiny training pipeline"
	@echo "make benchmark-inference - compare no/dynamic/static KV cache"
	@echo "make benchmark-training  - run the controlled training benchmark"
	@echo "make gpu-suite          - run the resumable GPU experiment suite"
	@echo "make gpu-suite-plan     - print the suite plan without running it"
	@echo "make gpu-suite-rehearse - rehearse the whole suite on CPU in seconds"
	@echo "make gpu-report         - validate GPU artifacts and regenerate the report/dashboard"
	@echo "make dataset-sample     - fetch a small real corpus prefix from ModelScope"
	@echo "make verify  - run lint, tests, and smoke validation"

setup-gpu:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check benchmarks model/cache.py portfolio serving scripts/api_schema.py tests tools

smoke:
	$(PYTHON) -m portfolio.smoke_pipeline --device $(DEVICE)

benchmark-inference:
	$(PYTHON) -m benchmarks.inference_benchmark --device $(DEVICE) --profile

benchmark-training:
	$(PYTHON) -m benchmarks.training_benchmark --device $(DEVICE) --profile

# TRAIN_DATA is required for the pipeline group; without it that group is
# skipped rather than reporting perplexity from the toy corpus.
GPU_DEVICE ?= cuda
GPU_BUDGET ?= 360
SAMPLES ?= 20000
HOLDOUT ?= 2000
TRAIN_DATA ?=
VALIDATION_DATA ?=

dataset-sample:
	$(PYTHON) -m tools.fetch_dataset_sample --samples $(SAMPLES) --holdout $(HOLDOUT)

gpu-suite:
	$(PYTHON) -m benchmarks.gpu_suite --device $(GPU_DEVICE) --time-budget-minutes $(GPU_BUDGET) \
		$(if $(TRAIN_DATA),--train-data $(TRAIN_DATA)) \
		$(if $(VALIDATION_DATA),--validation-data $(VALIDATION_DATA))

gpu-suite-plan:
	$(PYTHON) -m benchmarks.gpu_suite --device $(GPU_DEVICE) --dry-run \
		$(if $(TRAIN_DATA),--train-data $(TRAIN_DATA)) \
		$(if $(VALIDATION_DATA),--validation-data $(VALIDATION_DATA))

gpu-suite-rehearse:
	$(PYTHON) -m benchmarks.gpu_suite --device cpu --quick

gpu-report:
	$(PYTHON) -m tools.summarize_gpu_results

verify: lint test smoke
