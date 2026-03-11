"""
train.py
--------
Training loop for the Fashion-MNIST SNN classifier.

Strategy:
    - Loss     : Cross Entropy on output membrane potentials at each time step
                 (surrogate gradient method — standard for SNNs)
    - Optimizer: Adam (lr=1e-3)
    - Metric   : Accuracy via spike counts (argmax of total spikes per class)

Data splits:
    - Train      : 80% of fashion-mnist_train.csv (model learns from this)
    - Validation : 20% of fashion-mnist_train.csv (used to monitor training
                   and tune hyperparameters — NOT for final reporting)
    - Test       : fashion-mnist_test.csv (touched ONCE at the very end)

Why membrane potentials for loss?
    Spike counts are not differentiable (binary 0/1), so we cannot
    backpropagate through them directly. Instead we use the raw membrane
    voltage at each time step, which is continuous and smooth, allowing
    snntorch's surrogate gradient to compute meaningful gradients.

Usage:
    python train.py --epochs 5
    python train.py --epochs 10 --lr 1e-3
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from dataloader import load_fashion_mnist
from encode import encode_latency
from model import FashionSNN

#-----Helpers---------------------------------------

"""
    Accumulate cross entropy loss over all T time steps.

    Parameters
    ----------
    mem3_rec : list of FloatTensor [B, 10] — membrane potentials per step
    targets  : LongTensor [B] — ground truth class indices
    criterion: loss function (CrossEntropyLoss)

    Returns
    -------
    loss : scalar Tensor
    """

def compute_loss(mem3_rec, targets, criterion):
    return sum(criterion(mem, targets) for mem in mem3_rec) / len(mem3_rec)

"""
    Evaluate accuracy on an entire DataLoader split.

    Returns
    -------
    accuracy : float (0.0 – 1.0)
    """

def evaluate_accuracy(model, data_loader, device, num_steps):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            spike_data = encode_latency(images, num_steps=num_steps)
            spike_counts, _ = model(spike_data)
            predictions = spike_counts.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        return correct / total

#-----Training Loop--------------------------------------------------------

def train(
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 128,
        beta: float = 0.95,
        num_steps: int = 25,
        val_split: float = 0.2,
        train_csv_path: str = "Fashion-MNIST-SNN-train.csv",
        test_csv_path: str = "Fashion-MNIST-SNN-test.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader_full, test_loader = load_fashion_mnist(train_csv=train_csv_path, test_csv=test_csv_path, batch_size=batch_size)

    #split training set into train + validation
    full_dataset = train_loader_full.dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Train samples : {train_size}")
    print(f"Val samples   : {val_size}")
    print(f"Test samples  : {len(test_loader.dataset)}")

    #----Model----------------------------------------
    model = FashionSNN(beta=beta).to(device)
    print(f"\nModel  : FashionSNN | beta={beta} | T={num_steps}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer & Loss ------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    #----Training------------------------------------------------------------
    print(f"\nTraining for {epochs} epoch(s)...\n")
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        correct = total = 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            #Encode batches into spike trains [T, B, 1, 28, 28]
            spike_data = encode_latency(images, num_steps=num_steps)

            #Forward pass
            optimizer.zero_grad()
            spike_counts, mem3_rec = model(spike_data)

            #Loss over all time steps
            loss = compute_loss(mem3_rec, labels, criterion)

            #Backward pass
            loss.backward()
            optimizer.step()

            #Metrics
            epoch_loss += loss.item()
            predictions = spike_counts.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            #progress every 20 batches
            if (batch_idx + 1)%20 == 0:
                print(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )
        #--- Epoch summary---------------------
        train_acc = correct / total
        val_acc = evaluate_accuracy(model, val_loader, device, num_steps)
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"\nEpoch {epoch}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Time: {elapsed:.1f}s\n"
        )

    #final test evaluation
    print("=" * 60)
    print("Training complete! Running final evaluation on test set...")
    test_acc = evaluate_accuracy(model, test_loader, device, num_steps)
    print(f"Final Test Accuracy : {test_acc:.2%}")
    print("=" * 60)

    return model, history, test_acc

#Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FashionSNN")
    parser.add_argument("--epochs",     type=int,   default=15,     help="Number of epochs")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--batch_size", type=int,   default=128,   help="Batch size")
    parser.add_argument("--beta",       type=float, default=0.95,  help="LIF decay rate")
    parser.add_argument("--num_steps",  type=int,   default=25,   help="Time steps (T)")
    parser.add_argument("--val_split",  type=float, default=0.2,   help="Validation split fraction")
    parser.add_argument("--train_csv",  type=str,   default="data/fashion-mnist_train.csv")
    parser.add_argument("--test_csv",   type=str,   default="data/fashion-mnist_test.csv")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        beta=args.beta,
        num_steps=args.num_steps,
        val_split=args.val_split,
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
    )