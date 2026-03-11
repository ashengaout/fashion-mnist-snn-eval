# SNN Classifier — Experiment Log
## Project Goal
Evaluate whether a Spiking Neural Network (SNN) can classify Fashion-MNIST
more efficiently than a standard CNN in terms of computational power. This will also be compared to a baseline
classical machine learning model (SVM).
## Model
**Architecture:** Fully Connected LIF SNN

**Layout:** 784 → 512 → 256 → 10

**Encoding:** Latency encoding (snntorch)

**Loss:** Cross Entropy on output membrane potentials (averaged over T steps)

**Optimizer:** Adam

### Benchmarks (Literature)

Benchmarks (FC SNN on Fashion-MNIST — literature)

|Epochs |Expected Accuracy|
|-------|-----------------|
|3|~60–70%|
|10|~75–82%|
|Tuned|~85%|

These are the expected benchmarks according to literature on Fashion MNIST dataset. The following experiments are focused 
on optimizing hyperparameters to 

-----

### Experiment 1
Observations - the run had an incredibly high loss value (90-100) which prompted interruption of the
current experiment and a refactoring of the loss function to be an average

#Current — sums loss across all T steps
loss = sum(criterion(mem, targets) for mem in mem3_rec)

#Fixed — averages loss across all T steps
loss = sum(criterion(mem, targets) for mem in mem3_rec) / len(mem3_rec)

### Experiment 2

This experiment did not meet expected benchmark values. 

#### Hyperparameter Notes

| Parameter | Value | Reasoning                                                          |
|-----------|-------|--------------------------------------------------------------------|
|beta |0.95 | High Membrane memory - good for sparse latency input (99.5% zeros) 
|num_steps| 100| Initial Value                                                      |
|lr  |1e-4 | Initial Value                                                      |
|batch_size|128| Initial Value                                                      |
|val_split|0.2|80/20 train/val split, seed=42 for reproducibility |

### Epoch Results

|Epoch|Average Loss|Train Accuracy|Val Accuracy|Time|
|-----|------------|-----|--------|------------|
|1| 2.0642 | 34.48% | 49.88% | 179.2s|
|2|1.6633|49.13%|49.45%|176.5s|
|3|1.5158|48.64%|48.47%|182.6s|
|4|1.4243|48.59%|48.73%|180.8s|
|5|1.3616|48.11%|48.19%|179.7s|

### Obsevations

Initial baseline runs yielded a training accuracy below 50%, indicating the model is not converging effectively. Two hyperparameters were identified as 
likely causes: the learning rate and the number of time steps (num_steps). These will be systematically adjusted in subsequent runs to isolate their individual impact on convergence.