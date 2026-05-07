# Contributing to FANoS-v2

FANoS-v2 is a research optimizer. Contributions should preserve reproducibility and avoid broad performance claims without evidence.

## Development Setup

```bash
python3 -m pip install virtualenv
python3 -m virtualenv fanos_virtualenv
source fanos_virtualenv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Dataset Policy

Keep downloaded datasets outside the package directory. The default local root is:

```text
/Users/nalin/Downloads/fanos_v2/datasets
```

Use:

```bash
python tools/fetch_datasets.py --dataset mnist
python tools/fetch_datasets.py --dataset eegbci --subject 1 --runs 3 4
```

## Benchmark Rules

- Report all optimizer hyperparameters.
- Report at least three seeds for paper-quality claims.
- Compare against AdamW with clipping when making stability claims.
- Include loss, accuracy, time, optimizer-state memory, and GPU peak memory when available.
- Do not claim outperformance from smoke tests.

## Pull Request Checklist

- `pytest` passes.
- New optimizer behavior has a unit test.
- Benchmark scripts write results outside the package directory.
- Documentation says whether changes affect `update_mode="parameter"`, `update_mode="physical"`, or both.
