"""
eda.py
------
Exploratory Data Analysis and Visualization suite for the Fashion-MNIST SNN project.

Visualizations
--------------
Dataset:
    1. Sample images per class
    2. Class distribution bar chart
    3. Pixel intensity distribution

Encoding:
    4. Spike timing heatmap       — 28x28 grid showing when each pixel fires
    5. Spike raster plot          — which neurons fire at which timestep
    6. Sparsity over timesteps    — fraction of silent neurons per timestep

Model Activity:
    7. Layer activity over timesteps — firing rate per layer per timestep
    8. Membrane potential trace      — single LIF neuron voltage over time

Usage
-----
    python eda.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

matplotlib.use('Agg')  # non-interactive backend

from dataloader import load_fashion_mnist
from encode import encode_latency
from model import FashionSNN

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

"""
    EDA and visualization suite for the Fashion-MNIST SNN project.

    Parameters
    ----------
    train_csv  : str  path to training CSV
    test_csv   : str  path to test CSV
    num_steps  : int  timesteps for encoding (must match training config)
    batch_size : int  batch size for data loading
    device     : torch.device
    save_dir   : str  directory to save all figures
    """

class FashionMNIST_EDA:
    def __init__(
            self,
            train_csv: str = "data/fashion-mnist_train.csv",
            test_csv: str = "data/fashion-mnist_test.csv",
            num_steps: int = 25,
            batch_size: int = 128,
            device: torch.device = None,
            save_dir: str = "results/eda",
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import os
        os.makedirs(save_dir, exist_ok=True)

        print("Loading data...")
        self.train_loader, self.test_loader = load_fashion_mnist(
            train_csv=train_csv,
            test_csv=test_csv,
            batch_size=batch_size,
        )

        # grab one batch for encoding visualizations
        self.sample_images, self.sample_labels = next(iter(self.test_loader))
        print(f"Sample batch loaded: {self.sample_images.shape}")

        # encode the sample batch once
        print(f"Encoding sample batch (T={num_steps})...")
        self.spike_data = encode_latency(self.sample_images, num_steps=num_steps)
        print(f"Spike data shape: {self.spike_data.shape}")
        sparsity = (self.spike_data == 0).float().mean().item()
        print(f"Overall sparsity: {sparsity:.2%} zeros\n")

        # ─────────────────────────────────────────────────────────────────────────
        # 1. Sample images per class
        # ─────────────────────────────────────────────────────────────────────────

    def plot_sample_images(self):
        """Show one sample image per class."""
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle("Fashion-MNIST — One Sample Per Class", fontsize=14)

        shown = {}
        for img, label in zip(self.sample_images, self.sample_labels):
            l = label.item()
            if l not in shown:
                shown[l] = img.squeeze().numpy()
            if len(shown) == 10:
                break

        for idx, (l, img) in enumerate(sorted(shown.items())):
            ax = axes[idx // 5][idx % 5]
            ax.imshow(img, cmap='gray')
            ax.set_title(CLASS_NAMES[l], fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        path = f"{self.save_dir}/sample_images.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

        # ─────────────────────────────────────────────────────────────────────────
        # 2. Class distribution
        # ─────────────────────────────────────────────────────────────────────────

    def plot_class_distribution(self):
        """Bar chart of class counts in the training set."""
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.numpy())

        counts = [all_labels.count(i) for i in range(10)]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(CLASS_NAMES, counts, color='steelblue', edgecolor='white')
        ax.set_ylabel("Sample Count")
        ax.set_title("Class Distribution — Training Set")
        ax.axhline(sum(counts) / 10, color='red', linestyle='--',
                   label=f'Mean: {sum(counts) // 10:,}')
        ax.legend()
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f'{count:,}', ha='center', fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = f"{self.save_dir}/class_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

        # ─────────────────────────────────────────────────────────────────────────
        # 3. Pixel intensity distribution
        # ─────────────────────────────────────────────────────────────────────────

    def plot_pixel_intensity(self):
        """Histogram of pixel intensities — motivates latency encoding."""
        pixels = self.sample_images.numpy().flatten()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pixels, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_xlabel("Pixel Intensity (normalized)")
        ax.set_ylabel("Frequency")
        ax.set_title("Pixel Intensity Distribution\n"
                     "(High zero density motivates latency encoding)")
        ax.axvline(pixels.mean(), color='red', linestyle='--',
                   label=f'Mean: {pixels.mean():.3f}')
        ax.legend()
        plt.tight_layout()
        path = f"{self.save_dir}/pixel_intensity.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Spike timing heatmap
    # ─────────────────────────────────────────────────────────────────────────

    def plot_spike_timing_heatmap(self, sample_idx: int = 0, class_idx: int = None):
        """
        28x28 heatmap showing the timestep at which each pixel fires.
        Silent pixels (no spike) shown in a distinct color.

        Parameters
        ----------
        sample_idx : index into the batch to visualize
        class_idx  : if set, finds the first sample of that class instead
        """
        if class_idx is not None:
            for i, l in enumerate(self.sample_labels):
                if l.item() == class_idx:
                    sample_idx = i
                    break

        # spike_data: [T, B, 1, 28, 28] → find first spike time per pixel
        spikes = self.spike_data[:, sample_idx, 0, :, :]  # [T, 28, 28]
        label = self.sample_labels[sample_idx].item()

        # for each pixel, find the timestep of first spike (-1 if silent)
        timing = torch.full((28, 28), -1.0)
        for t in range(self.num_steps):
            fired = (spikes[t] == 1) & (timing == -1)
            timing[fired] = t

        # mask silent pixels
        timing_np = timing.numpy()
        masked = np.ma.masked_where(timing_np == -1, timing_np)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Latency Encoding — {CLASS_NAMES[label]}", fontsize=13)

        # original image
        axes[0].imshow(self.sample_images[sample_idx].squeeze().numpy(), cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # spike timing heatmap
        cmap = plt.cm.plasma.copy()
        cmap.set_bad(color='black')  # silent pixels = black
        im = axes[1].imshow(masked, cmap=cmap, vmin=0, vmax=self.num_steps - 1)
        axes[1].set_title("Spike Timing Heatmap\n"
                          "(bright = early spike = bright pixel | black = silent)")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label="Timestep of First Spike")

        plt.tight_layout()
        path = f"{self.save_dir}/spike_timing_heatmap_{CLASS_NAMES[label].replace('/', '_')}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

        # ─────────────────────────────────────────────────────────────────────────
        # 5. Spike raster plot
        # ─────────────────────────────────────────────────────────────────────────

    def plot_spike_raster(self, sample_idx: int = 0, max_neurons: int = 100):
        """
        Raster plot showing which input neurons fire at which timestep.
        Each row = one neuron, each dot = one spike.

        Parameters
        ----------
        sample_idx  : index into the batch
        max_neurons : number of input neurons to display (max 784)
        """
        label = self.sample_labels[sample_idx].item()

        # flatten spatial dims: [T, B, 1, 28, 28] → [T, 784]
        spikes = self.spike_data[:, sample_idx, 0, :, :].view(self.num_steps, -1)
        spikes = spikes[:, :max_neurons].numpy()  # [T, max_neurons]

        fig, ax = plt.subplots(figsize=(12, 6))
        for neuron_idx in range(max_neurons):
            spike_times = np.where(spikes[:, neuron_idx] == 1)[0]
            ax.scatter(spike_times,
                       [neuron_idx] * len(spike_times),
                       s=4, c='steelblue', alpha=0.7)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Input Neuron Index")
        ax.set_title(f"Spike Raster Plot — {CLASS_NAMES[label]}\n"
                     f"(First {max_neurons} input neurons over {self.num_steps} timesteps)")
        ax.set_xlim(0, self.num_steps)
        ax.set_ylim(0, max_neurons)
        plt.tight_layout()
        path = f"{self.save_dir}/spike_raster_{CLASS_NAMES[label].replace('/', '_')}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

        # ─────────────────────────────────────────────────────────────────────────
        # 6. Sparsity over timesteps
        # ─────────────────────────────────────────────────────────────────────────

    def plot_sparsity_over_timesteps(self):
        """
        Fraction of silent input neurons at each timestep.
        Illustrates the sparse temporal structure of latency encoding.
        """
        # average over batch and spatial dims
        sparsity_per_step = []
        for t in range(self.num_steps):
            step_spikes = self.spike_data[t]  # [B, 1, 28, 28]
            sparsity = (step_spikes == 0).float().mean().item()
            sparsity_per_step.append(sparsity * 100)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(self.num_steps), sparsity_per_step,
                color='steelblue', linewidth=2)
        ax.fill_between(range(self.num_steps), sparsity_per_step,
                        alpha=0.2, color='steelblue')
        ax.axhline(np.mean(sparsity_per_step), color='red', linestyle='--',
                   label=f'Mean: {np.mean(sparsity_per_step):.1f}% silent')
        ax.set_xlabel("Timestep")
        ax.set_ylabel("% Silent Neurons")
        ax.set_ylim(0, 100)
        ax.set_title("Input Sparsity Over Timesteps\n"
                     "(Latency encoding — each neuron fires at most once)")
        ax.legend()
        plt.tight_layout()
        path = f"{self.save_dir}/sparsity_over_timesteps.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

        # ─────────────────────────────────────────────────────────────────────────
        # 7. Layer activity over timesteps
        # ─────────────────────────────────────────────────────────────────────────

    def plot_layer_activity(self, model: FashionSNN, sample_idx: int = 0):
        """
        Firing rate per hidden layer per timestep for a single sample.
        Shows how spike activity propagates through the network.

        Parameters
        ----------
        model      : trained FashionSNN (eval mode)
        sample_idx : index into the batch
        """
        model.eval()
        label = self.sample_labels[sample_idx].item()

        # single sample spike data [T, 1, 1, 28, 28]
        x = self.spike_data[:, sample_idx:sample_idx + 1, :, :, :].to(self.device)

        layer1_rates, layer2_rates, layer3_rates = [], [], []

        with torch.no_grad():
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()

            for t in range(self.num_steps):
                x_t = x[t].view(1, -1)

                cur1 = model.fc1(x_t)
                spk1, mem1 = model.lif1(cur1, mem1)

                cur2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur2, mem2)

                cur3 = model.fc3(spk2)
                spk3, mem3 = model.lif3(cur3, mem3)

                layer1_rates.append(spk1.mean().item())
                layer2_rates.append(spk2.mean().item())
                layer3_rates.append(spk3.mean().item())

        fig, ax = plt.subplots(figsize=(12, 5))
        timesteps = range(self.num_steps)
        ax.plot(timesteps, layer1_rates, label='Layer 1 (512 neurons)', color='#4C72B0')
        ax.plot(timesteps, layer2_rates, label='Layer 2 (256 neurons)', color='#DD8452')
        ax.plot(timesteps, layer3_rates, label='Output (10 neurons)', color='#55A868')
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Firing Rate (fraction of neurons active)")
        ax.set_title(f"Layer Activity Over Timesteps — {CLASS_NAMES[label]}")
        ax.legend()
        ax.set_xlim(0, self.num_steps - 1)
        plt.tight_layout()
        path = f"{self.save_dir}/layer_activity_{CLASS_NAMES[label].replace('/', '_')}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Membrane potential trace
    # ─────────────────────────────────────────────────────────────────────────

    def plot_membrane_potential(self, model: FashionSNN, sample_idx: int = 0,
                                neuron_idx: int = 0):
        """
        Membrane potential of a single output neuron over all timesteps.
        Illustrates LIF dynamics — charge, threshold, and reset.

        Parameters
        ----------
        model      : trained FashionSNN (eval mode)
        sample_idx : index into the batch
        neuron_idx : which output neuron to trace (0–9, corresponds to class)
        """
        model.eval()
        label = self.sample_labels[sample_idx].item()

        x = self.spike_data[:, sample_idx:sample_idx + 1, :, :, :].to(self.device)

        mem3_trace = []
        spk3_trace = []

        with torch.no_grad():
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()

            for t in range(self.num_steps):
                x_t = x[t].view(1, -1)

                cur1 = model.fc1(x_t)
                spk1, mem1 = model.lif1(cur1, mem1)

                cur2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur2, mem2)

                cur3 = model.fc3(spk2)
                spk3, mem3 = model.lif3(cur3, mem3)

                mem3_trace.append(mem3[0, neuron_idx].item())
                spk3_trace.append(spk3[0, neuron_idx].item())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"LIF Neuron Dynamics — Output Neuron {neuron_idx} "
                     f"({CLASS_NAMES[neuron_idx]})\nInput: {CLASS_NAMES[label]}", fontsize=12)

        # membrane potential
        ax1.plot(mem3_trace, color='steelblue', linewidth=1.5)
        ax1.set_ylabel("Membrane Potential")
        ax1.set_title("Membrane Potential Over Time")
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # spike events
        spike_times = [t for t, s in enumerate(spk3_trace) if s == 1]
        ax2.vlines(spike_times, 0, 1, color='red', linewidth=1.5, label='Spike')
        ax2.set_ylabel("Spike")
        ax2.set_xlabel("Timestep")
        ax2.set_title("Spike Events")
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlim(0, self.num_steps - 1)
        if spike_times:
            ax2.legend()

        plt.tight_layout()
        path = f"{self.save_dir}/membrane_potential_neuron{neuron_idx}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Run all
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(self, model: FashionSNN = None):
        """
        Generate all EDA plots.
        Pass a trained model to include layer activity and membrane potential plots.
        """
        print("\n── Dataset Visualizations ──────────────────────────")
        self.plot_sample_images()
        self.plot_class_distribution()
        self.plot_pixel_intensity()

        print("\n── Encoding Visualizations ─────────────────────────")
        self.plot_spike_timing_heatmap()
        # generate heatmaps for a few interesting classes
        for class_idx in [0, 6, 9]:  # T-shirt, Shirt, Ankle boot
            self.plot_spike_timing_heatmap(class_idx=class_idx)
        self.plot_spike_raster()
        self.plot_sparsity_over_timesteps()

        if model is not None:
            print("\n── Model Activity Visualizations ────────────────────")
            self.plot_layer_activity(model)
            self.plot_membrane_potential(model, neuron_idx=0)
            # trace the predicted class neuron too
            predicted = int(self.spike_data[:, 0].sum(0).view(-1).argmax().item())
            self.plot_membrane_potential(model, neuron_idx=predicted)

        print(f"\nAll EDA figures saved to {self.save_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("results/eda", exist_ok=True)

    # initialize EDA
    eda = FashionMNIST_EDA(
        train_csv="data/fashion-mnist_train.csv",
        test_csv="data/fashion-mnist_test.csv",
        num_steps=25,
        save_dir="results/eda",
    )

    # load trained model for layer activity + membrane potential plots
    model = FashionSNN(beta=0.95).to(device)
    model.load_state_dict(torch.load("results/models/best_model.pth", map_location=device))
    model.eval()

    # run all plots
    eda.run_all(model=model)