"""
model.py
--------
Fully Connected Leaky Integrate-and-Fire (LIF) SNN model
for Fashion-MNIST classification.

Architecture:
    Input (784) → FC → LIF → FC → LIF → FC → LIF (10 outputs)
    784 → 512 → 256 → 10

Output:
    Spike counts per class over T time steps.
    The class with the highest spike count is the predicted label.
    This is the SNN equivalent of a CNN's softmax output.

Hyperparameters (tuned from EDA):
    - Beta  : 0.95 (high membrane memory, good for sparse latency input)
    - T     : 100  (number of time steps, set in encoder.py)

Reference:
    https://snntorch.readthedocs.io/en/latest/
"""

import torch
import torch.nn as nn
import snntorch as snn

"""
   Fully connected spiking neural network with LIF neurons.

   Parameters
   ----------
   beta : float
       Membrane potential decay rate (0 < beta < 1).
       Higher = longer memory. Recommended 0.95 for sparse latency input.
   """


class FashionSNN(nn.Module):
    def __init__(self, beta: float = 0.95):
        super().__init__()

        #learnable layers
        self.fc1 = nn.Linear(784, 512)  # Input → Hidden 1
        self.fc2 = nn.Linear(512, 256)  # Hidden 1 → Hidden 2
        self.fc3 = nn.Linear(256, 10)  # Hidden 2 → Output (10 classes)

        # ── LIF neurons (one per layer) ────────────────────────────────────
        # beta is shared across all layers but can be made per-layer if needed
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)

        """
        Forward pass over T time steps.

        Parameters
        ----------
        x : FloatTensor of shape [T, B, 1, 28, 28]
            Spike-encoded input from encoder.py

        Returns
        -------
        spike_counts : FloatTensor of shape [B, 10]
            total spike count per output neuron over all T steps.
            Use argmax to get predicted class.
        mem3_rec : list of FloatTensor [B, 10]
            Membrane potential recordings at each time step (for loss).
        """

    def forward(self, x: torch.Tensor):
        T = x.shape[0]
        batch_size = x.shape[1]

        #initialize membrane potentials for all LiF layers
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        #accumulators
        spike_counts = torch.zeros(batch_size, 10, device=x.device)
        mem3_rec = [] #recording output

        for t in range(T):
            #flatten [B, 1, 28, 28] -> [B, 784]
            x_t = x[t].view(batch_size, -1)

            #layer 1: FC -> LiF
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif2(cur1, mem1)

            #layer 2: FC -> LiF
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif3(cur2, mem2)

            #layer 3 (output): FC -> LiF
            mem3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(mem3, mem3)

            #Accumulate output spikes and record membrane potential
            spike_counts += spk3
            mem3_rec.append(mem3)

        return spike_counts, mem3_rec

# ── Quick sanity check (run as script) ────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from dataloader import load_fashion_mnist
    from encode import encode_latency

    print("Loading data...")
    train_loader, _ = load_fashion_mnist(
        train_csv="data/fashion-mnist_train.csv",
        test_csv="data/fashion-mnist_test.csv",
    )

    print("Encoding first mini-batch...")
    images, labels = next(iter(train_loader))
    spike_data = encode_latency(images)  # [T, B, 1, 28, 28]

    print("Running forward pass through FashionSNN...")
    model = FashionSNN(beta=0.95)
    spike_counts, mem3_rec = model(spike_data)

    print(f"Input spike shape  : {spike_data.shape}")               # [100, 128, 1, 28, 28]
    print(f"Spike counts shape : {spike_counts.shape}")             # [128, 10]
    print(f"Mem recordings     : {len(mem3_rec)} steps")            # 100
    print(f"Predicted classes  : {spike_counts.argmax(dim=1)[:10]}") # first 10
    print(f"True labels        : {labels[:10]}")
    print("Model architecture:")
    print(model)


