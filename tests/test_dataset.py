"""Check the HDF5 layouts accepted by the LHCO dataset reader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from src.data.dataset import LHCDataset, build_dataloaders


def test_flat_hdf5_with_final_label_column(tmp_path):
    path = tmp_path / "events.h5"
    features = np.arange(30, dtype=np.float32).reshape(5, 6)
    labels = np.array([0, 1, 0, 1, 0], dtype=np.float32)[:, None]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("events", data=np.concatenate([features, labels], axis=1))

    dataset = LHCDataset(path)
    assert len(dataset) == 5
    assert dataset.input_dim == 6
    x, y = dataset[1]
    assert tuple(x.shape) == (6,)
    assert y.item() == 1

    train_loader, evaluation_loader = build_dataloaders(
        dataset, batch_size=2, val_fraction=1.0, seed=4
    )
    assert len(train_loader.dataset) == 0
    assert sum(len(batch_x) for batch_x, _ in evaluation_loader) == 5
