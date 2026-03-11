"""
dataloader.py
-------------
Data loading and preprocessing for Fashion-MNIST SNN project.
Reads raw CSV files, normalizes pixel intensities, reshapes to
[N, 1, 28, 28], and returns PyTorch DataLoaders.

Expected CSV layout (Zalando research format):
  - Column 0  : label  (0-9)
  - Columns 1-784 : pixel values (0-255)
"""


import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


LABEL_MAP = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

"""
    Load Fashion-MNIST from CSV files and return (train_loader, test_loader).

    Parameters
    ----------
    train_csv      : path to fashion-mnist_train.csv
    test_csv       : path to fashion-mnist_test.csv
    batch_size     : mini-batch size
    shuffle_train  : whether to shuffle the training set each epoch
    num_workers    : DataLoader worker processes

    Returns
    -------
    (train_loader, test_loader) – both yield (images, labels) batches
        images : FloatTensor of shape [B, 1, 28, 28], values in [0, 1]
        labels : LongTensor  of shape [B]
    """

def load_fashion_mnist(
        train_csv:str,
        test_csv:str,
        batch_size:int=128,
        num_workers:int=0,
) -> tuple[DataLoader, DataLoader]:
    train_loader = _build_loader(train_csv, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = _build_loader(test_csv, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

"""Internal helper: read a single CSV and wrap it in a DataLoader."""

def _build_loader(
        csv_path:str,
        batch_size:int,
        shuffle:bool,
        num_workers:int,
)-> DataLoader:
    df = pd.read_csv(csv_path)

    #basic integrity check
    assert df.isnull().sum().sum() == 0, f"NaN values found in {csv_path}"
    assert df.shape[1] == 785, (
        f"Expected 785 columns (1 label + 784 pixels), got {df.shape[1]}"
    )

    #label data (colum 0)
    labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)

    #pixels normalized to [0, 1], reshape [N, 1, 28, 28]
    pixels = torch.tensor(df.iloc[:, 1:].values/255.0, dtype=torch.float32)
    images = pixels.view(-1, 1, 28, 28)

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# ── Quick sanity check (run as script) ────────────────────────────────────────
if __name__ == "__main__":
    train = "data/fashion-mnist_test.csv"
    test = "data/fashion-mnist_test.csv"

    train_ld, test_ld = load_fashion_mnist(train, test)

    imgs, lbls = next(iter(train_ld))
    print(f"Train batches : {len(train_ld)}")
    print(f"Test  batches : {len(test_ld)}")
    print(f"Image shape   : {imgs.shape}")   # [128, 1, 28, 28]
    print(f"Label dtype   : {lbls.dtype}")   # torch.int64
    print(f"Pixel range   : [{imgs.min():.3f}, {imgs.max():.3f}]")