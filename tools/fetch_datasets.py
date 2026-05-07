"""Fetch lightweight public datasets outside the project directory.

Defaults to ``../datasets`` relative to the repository root.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def fetch_mnist(data_root: Path) -> None:
    from torchvision import datasets

    datasets.MNIST(root=str(data_root / "vision"), train=True, download=True)
    datasets.MNIST(root=str(data_root / "vision"), train=False, download=True)
    print(f"MNIST ready under {data_root / 'vision'}")


def fetch_fashionmnist(data_root: Path) -> None:
    from torchvision import datasets

    datasets.FashionMNIST(root=str(data_root / "vision"), train=True, download=True)
    datasets.FashionMNIST(root=str(data_root / "vision"), train=False, download=True)
    print(f"FashionMNIST ready under {data_root / 'vision'}")


def fetch_cifar10(data_root: Path) -> None:
    from torchvision import datasets

    datasets.CIFAR10(root=str(data_root / "vision"), train=True, download=True)
    datasets.CIFAR10(root=str(data_root / "vision"), train=False, download=True)
    print(f"CIFAR-10 ready under {data_root / 'vision'}")


def fetch_eegbci(data_root: Path, subject: int, runs: list[int]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(data_root / ".mplconfig"))
    from mne.datasets import eegbci

    paths = eegbci.load_data(subjects=subject, runs=runs, path=str(data_root / "eeg"), update_path=False)
    print("EEGBCI files:")
    for path in paths:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    parser.add_argument("--dataset", choices=["mnist", "fashionmnist", "cifar10", "eegbci", "all"], default="all")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--runs", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"mnist", "all"}:
        fetch_mnist(data_root)
    if args.dataset in {"fashionmnist", "all"}:
        fetch_fashionmnist(data_root)
    if args.dataset in {"cifar10", "all"}:
        fetch_cifar10(data_root)
    if args.dataset in {"eegbci", "all"}:
        fetch_eegbci(data_root, subject=args.subject, runs=args.runs)


if __name__ == "__main__":
    main()
