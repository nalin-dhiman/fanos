# GitHub and PyPI Release Plan

## Recommended Release Name

`v0.2.0-alpha`

## Release Positioning

FANoS-v2 should be released as an alpha research optimizer for PyTorch. The
public README should present two optimizers:

- `FANoSV2`: exact reference implementation with full diagnostics.
- `FANoSV2Fast`: experimental speed preset validated on lightweight vision.

## Required Before GitHub Release

- Run `./fanos_virtualenv/bin/python -m pytest`.
- Run the final three vision checks or link to the frozen CSV result folders.
- Add the paper package under `paper/fanos_v2_v02`.
- Add a GitHub release note with the exact claim boundary.
- Avoid marketing language like "beats AdamW" without the runtime caveat.

## Required Before PyPI

- Update `pyproject.toml` version from `0.1.0` to `0.2.0a0` or another alpha
  version.
- Build locally:

```bash
./fanos_virtualenv/bin/python -m pip install -U build twine
./fanos_virtualenv/bin/python -m build
./fanos_virtualenv/bin/python -m twine check dist/*
```

- Test install from the wheel in a fresh environment.
- Publish to TestPyPI first.
- Only publish to PyPI after TestPyPI install works.

## Suggested GitHub Release Note

```text
FANoS-v2 v0.2.0-alpha introduces FANoSV2Fast, an experimental fast preset for
the feedback-controlled momentum optimizer. On five-seed reduced-sample MNIST,
Fashion-MNIST, and CIFAR-10 benchmarks, FANoSV2Fast improves mean top-1 accuracy
over AdamW but remains about 50-60% slower. This is a research release, not a
drop-in AdamW replacement.
```

