# Repository Scope

This repository is the clean FANoS-v2 optimizer package.

It includes:

- installable PyTorch package under `src/fanos_v2`;
- optimizer tests under `tests`;
- benchmark and study scripts under `benchmarks` and `tools`;
- mathematical notes under `docs`;
- the v0.2 paper package under `paper/fanos_v2_v02`.

It intentionally excludes:

- virtual environments;
- raw datasets;
- local result folders;
- cache/build artifacts.

For full experiment outputs and historical CSV/log evidence, use the
`FANoS_PIPELINE` repository.

Critical status: FANoS-v2 is an alpha research optimizer. The v0.2 evidence
supports a repeated-seed lightweight-vision accuracy signal, but the optimizer
is still slower than AdamW and is not a universal replacement.

