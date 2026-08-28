"""PyTorch datasets shared by the training and evaluation scripts.

Public LHCO files do not all use the same HDF5 layout. ``LHCDataset`` handles
flat event arrays, particle tensors, and files with separate labels. It opens
the file lazily, which keeps DataLoader workers from sharing an HDF5 handle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:  # Importing the plugin registers Blosc, LZ4, and Zstd filters with h5py.
    import hdf5plugin  # noqa: F401
except ImportError:  # Gzip and uncompressed files still work without it.
    hdf5plugin = None


_FEATURE_KEYS = ("features", "particles", "Particles", "events", "data", "X", "x")
_LABEL_KEYS = ("labels", "label", "targets", "target", "y", "Y")


def _iter_datasets(group: h5py.Group, prefix: str = "") -> Iterator[tuple[str, h5py.Dataset]]:
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            yield path, value
        elif isinstance(value, h5py.Group):
            yield from _iter_datasets(value, path)


def _choose_dataset(entries: list[tuple[str, h5py.Dataset]], candidates: tuple[str, ...]):
    by_leaf = {path.rsplit("/", 1)[-1]: (path, ds) for path, ds in entries}
    for key in candidates:
        if key in by_leaf:
            return by_leaf[key]
    return None


class LHCDataset(Dataset):
    """Read LHCO events from HDF5 without loading the full file into memory.

    If the file has no separate label dataset, a binary final column is treated
    as the truth label. Unlabelled files, such as background-only simulation,
    use ``default_label``.
    """

    def __init__(self, path: str | Path, *, default_label: int = 0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 data file not found: {self.path}")
        self.default_label = int(default_label)
        self._file: h5py.File | None = None

        with h5py.File(self.path, "r") as handle:
            entries = [(p, d) for p, d in _iter_datasets(handle) if d.ndim >= 1]
            if not entries:
                raise ValueError(f"No datasets found in {self.path}")

            label_entry = _choose_dataset(entries, _LABEL_KEYS)
            feature_entry = _choose_dataset(entries, _FEATURE_KEYS)
            if feature_entry is None:
                candidates = [(p, d) for p, d in entries if d.ndim >= 2 and d.dtype.kind in "fiu"]
                if not candidates:
                    raise ValueError(f"No numeric feature array found in {self.path}")
                feature_entry = max(candidates, key=lambda item: int(np.prod(item[1].shape[1:])))

            self.feature_key = feature_entry[0]
            self.label_key = label_entry[0] if label_entry and label_entry[0] != self.feature_key else None
            shape = feature_entry[1].shape
            self._length = int(shape[0])
            raw_dim = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            self._label_in_last_column = False

            if self.label_key is None and len(shape) == 2 and raw_dim > 1:
                sample_size = min(self._length, 1024)
                last = np.asarray(feature_entry[1][:sample_size, -1])
                finite = last[np.isfinite(last)]
                if finite.size and np.all(np.isin(np.unique(finite), [0, 1])):
                    self._label_in_last_column = True
                    raw_dim -= 1

            self.input_dim = raw_dim

            if self.label_key is not None:
                label_len = int(handle[self.label_key].shape[0])
                if label_len != self._length:
                    raise ValueError(
                        f"Feature/label length mismatch in {self.path}: "
                        f"{self._length} != {label_len}"
                    )

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        handle = self._handle()
        raw = np.asarray(handle[self.feature_key][index], dtype=np.float32).reshape(-1)
        if self.label_key is not None:
            label = int(np.asarray(handle[self.label_key][index]).reshape(-1)[0])
        elif self._label_in_last_column:
            label = int(raw[-1])
            raw = raw[:-1]
        else:
            label = self.default_label
        return torch.from_numpy(raw.copy()), torch.tensor(label, dtype=torch.long)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


class SyntheticLHCDataset(Dataset):
    """Create a reproducible background/signal sample for quick validation.

    Signal events receive a small, localized shift in mean and variance. When
    the feature count represents ``(pT, eta, phi)`` triplets, pT is kept positive
    so the same sample can be passed through the ParT preprocessor.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        input_dim: int = 128,
        *,
        signal_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        if not 0.0 < signal_fraction < 1.0:
            raise ValueError("signal_fraction must be strictly between 0 and 1")
        rng = np.random.default_rng(seed)
        labels = (rng.random(n_samples) < signal_fraction).astype(np.int64)
        features = rng.normal(0.0, 1.0, size=(n_samples, input_dim)).astype(np.float32)
        affected = max(1, input_dim // 8)
        signal = labels == 1
        features[signal, :affected] += 2.0
        features[signal, affected : 2 * affected] *= 1.8

        if input_dim % 3 == 0:
            particles = features.reshape(n_samples, input_dim // 3, 3)
            particles[:, :, 0] = np.abs(particles[:, :, 0]) + 0.05
            particles[:, :, 2] = ((particles[:, :, 2] + np.pi) % (2 * np.pi)) - np.pi

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.input_dim = int(input_dim)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def build_dataloaders(
    dataset: Dataset,
    *,
    batch_size: int = 512,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build reproducible training and validation DataLoaders.

    For evaluation, ``val_fraction=1`` returns an empty training loader and a
    sequential loader over the full dataset.
    """
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    n_total = len(dataset)
    n_val = int(round(n_total * val_fraction))
    n_val = min(max(n_val, 0), n_total)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=n_train > 0, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader
