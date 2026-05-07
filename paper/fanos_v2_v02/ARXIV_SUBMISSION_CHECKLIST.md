# arXiv Submission Checklist

## Include

- `fanos_v2_v02.tex`
- `figures/*.pdf`
- `tables/*.tex`
- `MANIFEST.md`
- `REPRODUCIBILITY.md`

## Before Upload

- Compile from this directory with `pdflatex fanos_v2_v02.tex`.
- Confirm the PDF contains all figures and tables.
- Confirm the abstract does not claim universal AdamW replacement.
- Confirm the paper describes PINN and EEG results as preliminary.
- Confirm the repository has a stable tag, for example `v0.2.0-alpha`.
- Confirm package metadata is updated before PyPI: version, description,
  classifiers, long description, and license.
- Confirm a clean install works in a fresh virtual environment:

```bash
python -m venv /tmp/fanos-release-test
source /tmp/fanos-release-test/bin/activate
pip install -U pip build twine
pip install .
python - <<'PY'
import torch
from fanos_v2 import FANoSV2, FANoSV2Fast
p = torch.nn.Parameter(torch.tensor([1.0]))
opt = FANoSV2Fast([p])
print(type(opt).__name__, opt.param_groups[0]["preset"])
PY
```

## Do Not Claim

- State-of-the-art vision performance.
- Large-model validation.
- Broad physics/PINN superiority.
- Production speed parity with AdamW.

