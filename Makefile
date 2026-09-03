.PHONY: help test lint smoke benchmark-inference benchmark-training verify

PYTHON ?= python3
DEVICE ?= cpu

help:
	@echo "make test    - run the CPU test suite"
	@echo "make lint    - lint portfolio code and tests"
	@echo "make smoke   - run the end-to-end tiny training pipeline"
	@echo "make benchmark-inference - compare no/dynamic/static KV cache"
	@echo "make benchmark-training  - run the controlled training benchmark"
	@echo "make verify  - run lint, tests, and smoke validation"

test:
	$(PYTHON) -m pytest

lint:
	uvx ruff check benchmarks model/cache.py portfolio serving scripts/api_schema.py tests tools

smoke:
	$(PYTHON) -m portfolio.smoke_pipeline --device $(DEVICE)

benchmark-inference:
	$(PYTHON) -m benchmarks.inference_benchmark --device $(DEVICE) --profile

benchmark-training:
	$(PYTHON) -m benchmarks.training_benchmark --device $(DEVICE) --profile

verify: lint test smoke
