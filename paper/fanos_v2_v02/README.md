# FANoS-v2 v0.2 Paper Package

This directory contains an arXiv-style draft paper and generated evidence
artifacts for the FANoS-v2 v0.2 research milestone.

## Build

```bash
cd /Users/nalin/Downloads/fanos_v2/fanos_v2_project
MPLCONFIGDIR=/private/tmp/matplotlib ./fanos_virtualenv/bin/python tools/build_arxiv_package.py
cd paper/fanos_v2_v02
pdflatex -interaction=nonstopmode fanos_v2_v02.tex
```

## Critical Status

The package supports a cautious alpha release:

```text
FANoS-v2 has a repeated-seed lightweight-vision accuracy signal.
FANoSV2Fast reduces but does not remove runtime overhead.
Physics/PINN results remain preliminary.
```

It does not support a broad claim that FANoS-v2 is a universal optimizer.

