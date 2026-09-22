# Experiment evidence

The checked-in JSON files and profiler operator tables record the controlled
CPU and RTX 4090 experiments described in docs/EXPERIMENTS.md. GPU evidence is
under `gpu/`, including raw reports, dataset provenance, run metadata, and a
generated summary. The 500-step run is a learning-pipeline check, not a
converged 64M model-quality evaluation.

Generated model weights and raw Chrome traces stay on the local machine and
are excluded from Git. Recreate them with the benchmark commands in README.md.
The training report's local absolute path fields were converted to
repository-relative paths for publication; its numeric results were not changed.

`images/gpu_results.svg` is generated from the checked-in GPU JSON by
`make gpu-report`. Other historical images may come from upstream MiniMind and
are not presented as personal experiment results.
