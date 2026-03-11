# Fashion-MNIST SNN Evaluation

**Course:** Advanced Machine Learning — Georgia State University  
**Team:** Sattwik Bhattacharjee, Amelia Shengaout, Hassan Coulibaly

## Overview
This project evaluates Spiking Neural Networks (SNNs) against classical machine 
learning baselines for image classification on the Fashion-MNIST dataset. SNNs 
represent a third generation of neural networks inspired by biological neural 
processing, operating on discrete spike events rather than continuous-valued 
computations — offering potential advantages in energy efficiency and biological 
plausibility.

## Objectives
- Establish classical ML baselines using SVM and Naive Bayes classifiers
- Design and implement an SNN using Leaky Integrate-and-Fire (LIF) neurons
- Train the SNN using surrogate gradient-based learning
- Compare accuracy, precision, recall, and F1-score across models
- Analyze the impact of latency encoding on classification performance

## Architecture
- **Encoding:** Latency encoding via SNNTorch (`spikegen.latency`)
- **SNN:** Fully connected LIF network (784 → 512 → 10)
- **Baselines:** SVM, Naive Bayes
- **Framework:** PyTorch + SNNTorch

## Project Structure
```
├── data/              # See data/README.md for download instructions
├── dataloader.py      # Data loading and preprocessing
├── encode.py          # Latency spike encoding
├── model.py           # LIF SNN architecture
├── train.py           # Training loop
├── evaluate.py        # Metrics and evaluation
└── main.py            # Entry point
```

## Setup
```bash
git clone https://github.com/your-username/fashion-mnist-snn-eval
cd fashion-mnist-snn-eval
pip install -r requirements.txt
```
Then follow `data/README.md` to download the dataset.

## Dataset
Fashion-MNIST CSV files — download from:  
https://www.kaggle.com/datasets/zalando-research/fashionmnist

## Dependencies
- Python 3.x
- PyTorch 2.10.0
- SNNTorch 0.9.4
- pandas, numpy, matplotlib

## References
- Han Xiao et al., *Fashion-MNIST: a Novel Image Dataset for Benchmarking 
  Machine Learning Algorithms*, arXiv:1708.07747
- Eshraghian et al., *Training Spiking Neural Networks Using Lessons From Deep 
  Learning*, arXiv:2109.12894
