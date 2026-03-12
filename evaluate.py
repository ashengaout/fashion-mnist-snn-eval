import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataloader import load_fashion_mnist
from encode import encode_latency
from model import FashionSNN

# Fashion-MNIST class labels
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

"""
    Evaluation suite for FashionSNN.

    Computes
    --------
    - Overall and per-class accuracy (mean ± std over n_runs)
    - Precision, Recall, F1 (per class + macro)
    - Confusion matrix
    - Sparsity rate per layer      (energy proxy)
    - Synaptic operations / SOPs   (energy proxy)
    - Inference time per sample

    Parameters
    ----------
    model     : FashionSNN — trained model already on `device`
    loader    : DataLoader — test loader from load_fashion_mnist()
    device    : torch.device
    num_steps : int — must match the value used during training (default 25)
    n_runs    : int — number of evaluation runs to average accuracy over
    """
class SNNEvaluator:
    def __init__(self, model, loader, device, num_steps: int = 25, n_runs: int = 5):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_steps = num_steps
        self.n_runs = n_runs

        # populated after run()
        self.all_preds = []
        self.all_labels = []
        self.sparsity_records = []
        self.sop_total = 0
        self.inference_times = []
        self.run_accuracies = []

    #Entry point ----------------------------------------------
    """
           Full evaluation pass.
           Runs inference n_runs times to compute mean/std accuracy.
           Sparsity, SOPs, confusion matrix collected on the final run.
           """
    def run(self):
        print(f"Running evaluation ({self.n_runs} runs for accuracy mean/std)...")

        for run_idx in range(self.n_runs):
            preds, labels, correct, total = [], [], 0, 0

            # only collect sparsity/SOP/timing on the last run

            collect_extras = (run_idx == self.n_runs - 1)
            if collect_extras:
                self.sparsity_records = []
                self.inference_times = []
                self.sop_total = 0

            self.model.eval()
            with torch.no_grad():
                for images, label_batch in self.loader:
                    images = images.to(self.device)
                    label_batch = label_batch.to(self.device)

                    spike_data = encode_latency(images, num_steps=self.num_steps)

                    t0 = time.perf_counter()
                    if collect_extras:
                        spike_counts, _ = self._forward_with_hooks(spike_data)
                    else:
                        spike_counts, _ = self.model(spike_data)
                    elapsed = time.perf_counter() - t0

                    if collect_extras:
                        self.inference_times.append(elapsed / images.size(0))

                    batch_preds = spike_counts.argmax(dim=1)
                    preds.extend(batch_preds.cpu().numpy())
                    labels.extend(label_batch.cpu().numpy())
                    correct += (batch_preds == label_batch).sum().item()
                    total += label_batch.size(0)

            run_acc = correct / total
            self.run_accuracies.append(run_acc)
            print(f"  Run {run_idx + 1}/{self.n_runs} — Accuracy: {run_acc:.2%}")

        # store final run predictions for confusion matrix / F1
        self.all_preds = np.array(preds)
        self.all_labels = np.array(labels)

        #-----Forward Pass with sparsity + SOP tracking
        """Mirrors FashionSNN.forward() and records spike activity."""

        def forward_with_hooks(self, x):
            m = self.model
            T = x.shape[0]
            B = x.shape[1]

            mem1 = m.lif1.init_leaky()
            mem2 = m.lif2.init_leaky()
            mem3 = m.lif3.init_leaky()

            spike_counts = torch.zeros(B, 10, device=self.device)
            mem3_rec = []

            # fan-out sizes for SOP calculation
            w2 = m.fc2.weight.shape[0] * m.fc2.weight.shape[1]  # 256 * 512
            w3 = m.fc3.weight.shape[0] * m.fc3.weight.shape[1]  # 10  * 256

            for t in range(T):
                x_t = x[t].view(B, -1)

                # Layer 1
                cur1 = m.fc1(x_t)
                spk1, mem1 = m.lif1(cur1, mem1)
                if hasattr(m, 'drop1'):
                    spk1 = m.drop1(spk1)

                # Layer 2
                cur2 = m.fc2(spk1)
                spk2, mem2 = m.lif2(cur2, mem2)
                if hasattr(m, 'drop2'):
                    spk2 = m.drop2(spk2)

                # Output layer
                cur3 = m.fc3(spk2)
                spk3, mem3 = m.lif3(cur3, mem3)

                spike_counts += spk3
                mem3_rec.append(mem3)

                # sparsity per layer
                self.sparsity_records.append({"layer": 1, "sparsity": 1.0 - spk1.mean().item()})
                self.sparsity_records.append({"layer": 2, "sparsity": 1.0 - spk2.mean().item()})
                self.sparsity_records.append({"layer": 3, "sparsity": 1.0 - spk3.mean().item()})

                # SOPs
                self.sop_total += int(spk1.sum().item()) * w2
                self.sop_total += int(spk2.sum().item()) * w3

            return spike_counts, mem3_rec

        # ─────────────────────────────────────────────────────────────────────────
        # Metrics
        # ─────────────────────────────────────────────────────────────────────────

        def mean_accuracy(self) -> float:
            return float(np.mean(self.run_accuracies)) * 100

        def std_accuracy(self) -> float:
            return float(np.std(self.run_accuracies)) * 100

        def per_class_accuracy(self) -> dict:
            results = {}
            for i, name in enumerate(CLASS_NAMES):
                mask = self.all_labels == i
                results[name] = float((self.all_preds[mask] == i).mean()) * 100
            return results

        def macro_f1(self) -> float:
            return f1_score(self.all_labels, self.all_preds, average='macro') * 100

        def sparsity_summary(self) -> dict:
            summary = {}
            for layer_id in [1, 2, 3]:
                vals = [r["sparsity"] for r in self.sparsity_records
                        if r["layer"] == layer_id]
                summary[f"layer_{layer_id}"] = float(np.mean(vals)) * 100
            return summary

        def avg_inference_time_ms(self) -> float:
            return float(np.mean(self.inference_times)) * 1000

        # ─────────────────────────────────────────────────────────────────────────
        # Print summary
        # ─────────────────────────────────────────────────────────────────────────

        def print_summary(self):
            print("\n" + "=" * 60)
            print("  SNN EVALUATION SUMMARY")
            print("=" * 60)

            print(f"\nMean Test Accuracy : {self.mean_accuracy():.2f}% "
                  f"(± {self.std_accuracy():.2f}% over {self.n_runs} runs)")
            print(f"Macro F1 Score     : {self.macro_f1():.2f}%")
            print(f"Avg Inference Time : {self.avg_inference_time_ms():.3f} ms/sample")
            print(f"Total Synaptic Ops : {self.sop_total:,}")

            print("\n── Per-Class Accuracy ──────────────────────────────")
            for name, acc in self.per_class_accuracy().items():
                print(f"  {name:<15} {acc:.2f}%")

            print("\n── Sparsity (% silent neurons) ─────────────────────")
            for layer, val in self.sparsity_summary().items():
                print(f"  {layer} : {val:.2f}% silent")

            print("\n── Full Classification Report ──────────────────────")
            print(classification_report(
                self.all_labels, self.all_preds, target_names=CLASS_NAMES
            ))

        # ─────────────────────────────────────────────────────────────────────────
        # Plots
        # ─────────────────────────────────────────────────────────────────────────

        def plot_confusion_matrix(self, save_path: str = None):
            cm = confusion_matrix(self.all_labels, self.all_preds)
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix (%) — FashionSNN\n"
                         f"Mean Accuracy: {self.mean_accuracy():.2f}% ± {self.std_accuracy():.2f}%")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()

        def plot_per_class_accuracy(self, save_path: str = None):
            names = list(CLASS_NAMES)
            accs = [self.per_class_accuracy()[n] for n in names]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(names, accs, color='steelblue', edgecolor='white')
            ax.axhline(self.mean_accuracy(), color='red', linestyle='--',
                       label=f'Mean Overall: {self.mean_accuracy():.2f}%')
            ax.set_ylim(0, 100)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Per-Class Accuracy — FashionSNN")
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()

        def plot_sparsity(self, save_path: str = None):
            summary = self.sparsity_summary()
            labels = [f"Layer {k.split('_')[1]}" for k in summary]
            values = list(summary.values())

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, values, color=['#4C72B0', '#DD8452', '#55A868'],
                   edgecolor='white')
            ax.set_ylabel("% Silent Neurons")
            ax.set_ylim(0, 100)
            ax.set_title("Sparsity per Layer — FashionSNN")
            for i, v in enumerate(values):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()

        def plot_accuracy_runs(self, save_path: str = None):
            """Bar chart of accuracy across all n_runs with mean line."""
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(1, self.n_runs + 1),
                   [a * 100 for a in self.run_accuracies],
                   color='steelblue', edgecolor='white')
            ax.axhline(self.mean_accuracy(), color='red', linestyle='--',
                       label=f'Mean: {self.mean_accuracy():.2f}% ± {self.std_accuracy():.2f}%')
            ax.set_xlabel("Run")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(80, 90)
            ax.set_title("Accuracy Across Evaluation Runs — FashionSNN")
            ax.legend()
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()

        def plot_all(self, save_dir: str = None):
            """Generates all four plots."""
            self.plot_confusion_matrix(
                save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None)
            self.plot_per_class_accuracy(
                save_path=f"{save_dir}/per_class_accuracy.png" if save_dir else None)
            self.plot_sparsity(
                save_path=f"{save_dir}/sparsity.png" if save_dir else None)
            self.plot_accuracy_runs(
                save_path=f"{save_dir}/accuracy_runs.png" if save_dir else None)


if __name__ == "__main__":
    import os
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create result directories
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    # load test data
    _, test_loader = load_fashion_mnist(
        train_csv="data/fashion-mnist_train.csv",
        test_csv="data/fashion-mnist_test.csv",
        batch_size=128,
    )

    # load trained model
    model = FashionSNN(beta=0.95).to(device)
    model.load_state_dict(torch.load("results/models/best_model.pth", map_location=device))

    # run evaluation
    evaluator = SNNEvaluator(model, test_loader, device, num_steps=25, n_runs=5)
    evaluator.run()

    # print to console
    evaluator.print_summary()

    # save summary to file
    with open("results/metrics/snn_eval_summary.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        evaluator.print_summary()
        sys.stdout = sys.__stdout__

    print("\nSummary saved to results/metrics/snn_eval_summary.txt")

    # save all plots
    evaluator.plot_all(save_dir="results/figures")
    print("Figures saved to results/figures/")