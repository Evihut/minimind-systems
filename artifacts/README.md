# Experiment evidence

The checked-in JSON files and profiler operator tables record local CPU
experiments described in docs/EXPERIMENTS.md. They are small exploratory runs,
not production service benchmarks or 64M model-quality evaluations.

Generated model weights and raw Chrome traces stay on the local machine and
are excluded from Git. Recreate them with the benchmark commands in README.md.
The training report's local absolute path fields were converted to
repository-relative paths for publication; its numeric results were not changed.

The images/ directory contains upstream illustrations and results, not results
from these personal experiments.
