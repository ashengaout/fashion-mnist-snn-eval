import time
from pathlib import Path

class ExperimentLogger:
    def __init__(self, log_path="experiments.md", experiment_num=8):
        self.log_path = Path(log_path)
        self.exp_num = experiment_num
        self.epoch_rows = []

    def log_config(self, params: dict, notes: str = ""):
        self.config = params
        self.notes = notes

    def log_epoch(self, epoch, loss, train_acc, val_acc, elapsed, lr=None):
        self.epoch_rows.append((epoch, loss, train_acc, val_acc, elapsed, lr))
        # Print live so you still see it in PyCharm console
        print(f"Epoch {epoch:2d} | loss {loss:.4f} | train {train_acc:.2f}% | val {val_acc:.2f}% | {elapsed:.1f}s")

    def save(self, test_acc=None):
        lines = [f"\n## Experiment {self.exp_num}\n"]
        lines.append(self.notes + "\n")

        # Hyperparameter table
        lines.append("### Hyperparameter Notes\n")
        lines.append("| Parameter | Value | Reasoning |\n|-----------|-------|-----------|")
        for k, v in self.config.items():
            lines.append(f"\n|{k}|{v}|  |")

        # Epoch table
        lines.append("\n\n### Epoch Results\n")
        # And update the table in save():
        lines.append("| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |\n"
                     "|-------|--------------|----------------|--------------|-----|------|")
        for epoch, loss, train_acc, val_acc, t, lr in self.epoch_rows:
            lr_str = f"{lr:.2e}" if lr else "-"
            lines.append(f"\n| {epoch} | {loss:.4f} | {train_acc:.2f}% | {val_acc:.2f}% | {lr_str} | {t:.1f}s |")

        if test_acc:
            lines.append(f"\n\nFinal Test Accuracy: {test_acc:.2f}%\n")

        lines.append("\n\n### Observations\n\n_Fill in after reviewing results._\n")

        with open(self.log_path, "a", encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\nLogged to {self.log_path}")