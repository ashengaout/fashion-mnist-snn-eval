# Fashion-MNIST SNN Classifier

**Course:** Advanced Machine Learning — Georgia State University  
**Team:** Sattwik Bhattacharjee, Amelia Shengaout, Hassan Coulibaly

---

## Overview

This project investigates whether artificial latency encoding can preserve sufficient spatial structure from static images to enable competitive classification in a Spiking Neural Network (SNN). We benchmark a fully connected LIF-neuron SNN against classical ML baselines (SVM, Naive Bayes) on the Fashion-MNIST dataset, quantifying the accuracy cost and energy efficiency trade-offs of neuromorphic deployment on non-temporal data.

SNNs represent a third generation of neural networks inspired by biological neural processing, operating on discrete spike events rather than continuous-valued computations — offering potential advantages in energy efficiency on neuromorphic hardware.

---

## Research Questions

1. Can artificial latency encoding preserve sufficient spatial structure from static images to enable competitive classification in an SNN?
2. What is the accuracy cost of deploying a neuromorphic model on inherently non-temporal data compared to classical ML baselines?
3. At what point does the theoretical energy efficiency of an SNN justify its accuracy trade-off against classical approaches?

---

## Results Summary

| Model | Test Accuracy | Notes |
|-------|-------------|-------|
| FC-LIF SNN | **84.51% ± 0.00%** | Latency encoding, 25 timesteps |
| SVM | TBD | RBF kernel baseline |
| Naive Bayes | TBD | Probabilistic baseline |

**SNN Efficiency Metrics**
| Metric | Value |
|--------|-------|
| Macro F1 | 84.37% |
| Avg Inference Time | 0.382 ms/sample |
| Layer 1 Sparsity | 93.48% silent |
| Layer 2 Sparsity | 79.74% silent |
| Layer 3 Sparsity | 86.32% silent |
| Total Synaptic Ops | 1,127,899,850,240 |
| Input Sparsity (mean) | 98.0% silent |

---

## Architecture

**SNN**
- Encoding: Latency encoding via SNNTorch (`spikegen.latency`)
- Layout: Fully connected LIF network (784 → 512 → 256 → 10)
- Loss: Cross entropy on output membrane potentials averaged over T timesteps
- Optimizer: Adam (lr=1e-3)
- Regularization: Dropout (p=0.2)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)

**Baselines**
- SVM (RBF kernel)
- Naive Bayes

**Framework:** PyTorch + SNNTorch

---

## Project Structure

```
SNN Classifier/
├── data/                        # See data/README.md for download instructions
├── experiment_logs/
│   └── 01_SNN_optimization_log.md   # Full hyperparameter optimization history (14 experiments)
├── results/
│   ├── eda/                     # EDA figures — spike heatmaps, raster plots, membrane potentials
│   ├── figures/                 # Evaluation figures — confusion matrix, per-class accuracy, sparsity
│   ├── metrics/                 # snn_eval_summary.txt
│   └── models/
│       └── best_model.pth       # Final trained SNN weights
├── utils/
│   └── logger.py                # Experiment logging utility
├── dataloader.py                # Data loading and preprocessing
├── encode.py                    # Latency spike encoding
├── model.py                     # LIF SNN architecture
├── train.py                     # Training loop with scheduler and logging
├── evaluate.py                  # Full evaluation suite (metrics + plots)
├── eda.py                       # EDA and spike visualization suite
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/your-username/fashion-mnist-snn-eval
cd fashion-mnist-snn-eval
pip install -r requirements.txt
```

Then follow `data/README.md` to download the dataset.

---

## Training

```bash
python train.py --epochs 30 --lr 1e-3 --beta 0.95 --num_steps 25 --batch_size 128
```

Trained weights will be saved to `results/models/best_model.pth`.

---

## Evaluation

```bash
python evaluate.py
```

Generates accuracy (mean ± std over 5 runs), F1, confusion matrix, per-class accuracy, sparsity rates, and synaptic operation counts. All figures saved to `results/figures/`.

---

## EDA & Spike Visualizations

```bash
python eda.py
```

Generates the following figures saved to `results/eda/`:

| Figure | Description |
|--------|-------------|
| `sample_images.png` | One sample per class |
| `class_distribution.png` | Confirms perfectly balanced dataset (6,000 per class) |
| `pixel_intensity.png` | Pixel intensity histogram — motivates latency encoding (high zero density) |
| `spike_timing_heatmap_*.png` | 28×28 grid showing when each pixel fires — one per class |
| `spike_raster_*.png` | Raster plot of 100 input neurons over 25 timesteps |
| `sparsity_over_timesteps.png` | 98% mean input sparsity — supports energy efficiency argument |
| `layer_activity_*.png` | Firing rate per layer per timestep — shows spike propagation |
| `membrane_potential_neuron*.png` | LIF neuron voltage trace and spike events over time |

---

## Dataset

Fashion-MNIST CSV files — download from:  
[https://www.kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

---

## Dependencies

- Python 3.x
- PyTorch 2.10.0
- SNNTorch 0.9.4
- scikit-learn
- pandas, numpy, matplotlib, seaborn

---

## References

- Han Xiao et al., *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, arXiv:1708.07747
- Eshraghian et al., *Training Spiking Neural Networks Using Lessons From Deep Learning*, arXiv:2109.12894
- Chandarana et al., *An Adaptive Sampling and Edge Detection Approach for Encoding Static Images for SNNs*, arXiv:2110.10217
- Chowdhury et al., *Towards Ultra Low Latency SNNs*, ECCV 2022
