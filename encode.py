"""
encoder.py
----------
Spike encoding for the Fashion-MNIST SNN project.
Converts normalized pixel tensors [0, 1] into spike trains
using latency encoding via snntorch.

Encoding Selection:
    Latency encoding is used because larger inputs spike earlier
    (values closer to 1 fire at earlier time steps), which reduces
    computational cost — a key advantage of SNNs.

    Note: Latency encoding is more susceptible to noise than rate
    encoding, so this may be revisited during training evaluation.

How Latency Encoding Works:
    - Higher pixel intensity → earlier spike time
    - Formula: t = (1 - I) * (T - 1)
    - Each neuron fires at most ONCE per image (sparse representation)

Hyperparameter Recommendations (from EDA):
    - Input Gain : Scale tensors (e.g., x3) for dim samples
    - Timesteps T: ~100 steps for sufficient temporal integration
    - Decay Beta : ~0.95 to retain charge from sparse signals
    - Threshold  : Lower to ~0.5 if dim classes show poor accuracy

Reference:
    https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html#spike-encoding
"""

import torch
from snntorch import spikegen

#default encoding hyperparams
DEFAULT_NUM_STEPS = 100   # Temporal window (T)
DEFAULT_TAU       = 5     # Time constant for latency curve
DEFAULT_THRESHOLD = 0.01  # Min pixel intensity to generate a spike
DEFAULT_NORMALIZE = True  # Normalize spike times across the time window
DEFAULT_CLIP      = True  # Clip trailing spikes (background noise removal)
DEFAULT_LINEAR    = True  # Use linear (not exponential) latency mapping

"""
   Encode a batch of normalized images into latency spike trains.

   Parameters
   ----------
   data      : FloatTensor of shape [B, C, H, W], values in [0, 1]
   num_steps : Number of simulation time steps (T)
   tau       : Time constant controlling the latency curve shape
   threshold : Pixels below this intensity will NOT generate spikes
   normalize : If True, spike times are normalized to fill [0, T-1]
   clip      : If True, removes late background spikes (recommended)
   linear    : If True, uses linear mapping instead of exponential

   Returns
   -------
   spike_data : FloatTensor of shape [T, B, C, H, W]
                Binary spike trains (0 or 1) over T time steps
   """

def encode_latency(
        data:torch.Tensor,
        num_steps:int=DEFAULT_NUM_STEPS,
        tau:float=DEFAULT_TAU,
        threshold:float=DEFAULT_THRESHOLD,
        normalize:bool=DEFAULT_NORMALIZE,
        clip:bool=DEFAULT_CLIP,
        linear:bool=DEFAULT_LINEAR,
) -> torch.Tensor:
    spike_data = spikegen.latency(
        data,
        num_steps=num_steps,
        tau=tau,
        threshold=threshold,
        normalize=normalize,
        clip=clip,
        linear=linear,
    )
    return spike_data

"""
    Generator that yields (spike_data, labels) for each mini-batch
    in the provided DataLoader.

    Parameters
    ----------
    data_loader : PyTorch DataLoader yielding (images, labels)
    num_steps   : Number of simulation time steps (T)
    tau         : Time constant for latency curve
    threshold   : Min intensity to spike
    normalize   : Normalize spike times across T
    clip        : Clip trailing background spikes
    linear      : Use linear latency mapping

    Yields
    ------
    spike_data : FloatTensor [T, B, C, H, W]
    labels     : LongTensor  [B]
    """

def encode_batch(
        data_loader,
        num_steps:int=DEFAULT_NUM_STEPS,
        tau:float=DEFAULT_TAU,
        threshold:float=DEFAULT_THRESHOLD,
        normalize:bool=DEFAULT_NORMALIZE,
        clip:bool=DEFAULT_CLIP,
        linear:bool=DEFAULT_LINEAR,
):
    for images, labels in data_loader:
        spike_data = encode_latency(
            images,
            num_steps=num_steps,
            tau=tau,
            threshold=threshold,
            normalize=normalize,
            clip=clip,
            linear=linear,
        )
        yield spike_data, labels

# ── Quick sanity check (run as script) ────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")  # ensure local modules are found

    from dataloader import load_fashion_mnist

    print("Loading data...")
    train_loader, test_loader = load_fashion_mnist(
        train_csv="data/fashion-mnist_train.csv",
        test_csv="data/fashion-mnist_test.csv",
    )

    print("Encoding first mini-batch with latency encoding...")
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    spike_data = encode_latency(images)

    print(f"Input image shape  : {images.shape}")       # [B, 1, 28, 28]
    print(f"Spike data shape   : {spike_data.shape}")   # [T, B, 1, 28, 28]
    print(f"Num time steps (T) : {spike_data.shape[0]}")
    print(f"Spike range        : [{spike_data.min():.0f}, {spike_data.max():.0f}]")
    print(f"Sparsity           : {(spike_data == 0).float().mean():.2%} zeros")
