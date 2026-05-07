# FANoS-v2 v0.2 Paper Package Manifest

Generated artifacts are derived from local CSV result folders under `../results`.

## Final vision sources
- MNIST: `results/fanosv2fast_default_mnist_mps_20260507_222936`
- Fashion-MNIST: `results/fanosv2fast_default_fashionmnist_mps_20260507_223017`
- CIFAR-10: `results/fanosv2fast_default_cifar10_mps_20260507_223058`

## Ablation sources
- MNIST k=1: `results/gradnorm_sweep_k1_mps_20260506_223104`
- MNIST k=2: `results/gradnorm_sweep_k2_mps_20260506_223156`
- MNIST k=4: `results/gradnorm_sweep_k4_mps_20260506_223238`
- MNIST k=8: `results/gradnorm_sweep_k8_mps_20260506_223316`
- Fashion-MNIST k=2: `results/gradnorm_fashionmnist_k2_mps_20260507_213549`
- Fashion-MNIST k=4: `results/gradnorm_fashionmnist_k4_mps_20260507_213633`
- CIFAR-10 k=2: `results/gradnorm_cifar10_k2_mps_20260507_213715`
- CIFAR-10 k=4: `results/gradnorm_cifar10_k4_mps_20260507_213814`

## Preliminary non-vision sources
- Full fixed research root: `results/full_research_mps_fixed_20260505_110606`

## Files
- `fanos_v2_v02.tex`: arXiv-style manuscript.
- `figures/*.pdf`: paper figures.
- `tables/*.tex`: generated LaTeX tables.
- `data/*.csv`: aggregated CSV inputs for audit.
- `REPRODUCIBILITY.md`: commands and claim boundaries.
- `ARXIV_SUBMISSION_CHECKLIST.md`: package/submission checklist.
