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
import shutil
from typing import Optional

import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from utils.logger import ExperimentLogger
import os

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
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 128,
        beta: float = 0.95,
        num_steps: int = 25,
        val_split: float = 0.2,
        train_csv_path: str = "Fashion-MNIST-SNN-train.csv",
        test_csv_path: str = "Fashion-MNIST-SNN-test.csv",
        seed: int = 42,
        experiment_num: int = 1,
        ckpt_path: Optional[str] = None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader_full, test_loader = load_fashion_mnist(train_csv=train_csv_path, test_csv=test_csv_path, batch_size=batch_size)

    #split training set into train + validation
    full_dataset = train_loader_full.dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )

    #----Training------------------------------------------------------------
    print(f"\nTraining for {epochs} epoch(s)...\n")
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    #Experimental Logger
    logger = ExperimentLogger("experiments.md", experiment_num=experiment_num)
    logger.log_config({
        "seed": seed,
        "beta": beta,
        "num_steps": num_steps,
        "lr_initial": lr,
        "batch_size": batch_size,
        "epochs": epochs,
    })

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0

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
        scheduler.step(val_acc)
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        logger.log_epoch(epoch, avg_loss, train_acc * 100, val_acc * 100, elapsed,
                         lr=optimizer.param_groups[0]['lr'])

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s\n")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"Best validation accuracy: {best_val_acc:.2%} (epoch {best_epoch})")

    #final test evaluation (weights = best val checkpoint)
    print("=" * 60)
    print("Training complete! Running final evaluation on test set...")
    test_acc = evaluate_accuracy(model, test_loader, device, num_steps)
    print(f"Final Test Accuracy : {test_acc:.2%}")
    print("=" * 60)

    logger.save(test_acc=test_acc * 100)

    os.makedirs("results/models", exist_ok=True)
    out_path = ckpt_path or "results/models/best_model.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Best-val checkpoint saved to {out_path}")

    return model, history, test_acc, best_val_acc, best_epoch

#Entry point
if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Train FashionSNN")
    parser.add_argument("--epochs",     type=int,   default=30,     help="Number of epochs")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--batch_size", type=int,   default=128,   help="Batch size")
    parser.add_argument("--beta",       type=float, default=0.95,  help="LIF decay rate")
    parser.add_argument("--num_steps",  type=int,   default=25,   help="Time steps (T)")
    parser.add_argument("--val_split",  type=float, default=0.2,   help="Validation split fraction")
    parser.add_argument("--train_csv",  type=str,   default="data/fashion-mnist_train.csv")
    parser.add_argument("--test_csv",   type=str,   default="data/fashion-mnist_test.csv")
    parser.add_argument("--seed",       type=int,   default=42,    help="Base random seed (single run)")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of training runs with seeds seed, seed+1, ...; keeps best val across trials",
    )
    args = parser.parse_args()

    os.makedirs("results/models", exist_ok=True)
    archive_dir = os.path.join("results", "other model")
    os.makedirs(archive_dir, exist_ok=True)
    meta_path = "results/models/best_model_meta.json"

    overall_best_val = float("-inf")
    overall_best_trial = None
    overall_best_test = None
    overall_best_epoch = None

    for t in range(args.trials):
        trial_seed = args.seed + t
        trial_ckpt = (
            None
            if args.trials == 1
            else f"results/models/trial_{t + 1}_best.pth"
        )
        _, _, test_acc, best_val_acc, best_epoch = train(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            beta=args.beta,
            num_steps=args.num_steps,
            val_split=args.val_split,
            train_csv_path=args.train_csv,
            test_csv_path=args.test_csv,
            seed=trial_seed,
            experiment_num=t + 1,
            ckpt_path=trial_ckpt,
        )
        saved_msg = trial_ckpt or "results/models/best_model.pth"
        archive_path = os.path.join(archive_dir, f"trial_{t + 1}_best.pth")
        shutil.copyfile(saved_msg, archive_path)
        print(
            f"\nTrial {t + 1}/{args.trials} | best val {best_val_acc:.2%} (epoch {best_epoch}) | "
            f"test {test_acc:.2%} | saved {saved_msg} | archived {archive_path}\n"
        )

        if best_val_acc > overall_best_val:
            overall_best_val = best_val_acc
            overall_best_trial = t + 1
            overall_best_test = test_acc
            overall_best_epoch = best_epoch
            if trial_ckpt is not None:
                shutil.copyfile(trial_ckpt, "results/models/best_model.pth")

    if args.trials > 1:
        print("=" * 60)
        print(
            f"Overall best: trial {overall_best_trial} | val {overall_best_val:.2%} "
            f"(epoch {overall_best_epoch}) | test {overall_best_test:.2%}"
        )
        print("Copied to results/models/best_model.pth")
        print("=" * 60)

    meta = {
        "best_val_acc": overall_best_val,
        "best_test_acc": overall_best_test,
        "best_trial": overall_best_trial,
        "best_epoch": overall_best_epoch,
        "num_trials": args.trials,
        "seed_base": args.seed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")