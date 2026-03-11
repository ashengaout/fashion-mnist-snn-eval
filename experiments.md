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

## Benchmarks (Literature)

Benchmarks (FC SNN on Fashion-MNIST — literature)

|Epochs |Expected Accuracy|
|-------|-----------------|
|3|~60–70%|
|10|~75–82%|
|Tuned|~85%|

These are the expected benchmarks according to literature on Fashion MNIST dataset. The following experiments are focused 
on optimizing hyperparameters to 

-----

## Experiment 1
Observations - the run had an incredibly high loss value (90-100) which prompted interruption of the
current experiment and a refactoring of the loss function to be an average

#Current — sums loss across all T steps
loss = sum(criterion(mem, targets) for mem in mem3_rec)

#Fixed — averages loss across all T steps
loss = sum(criterion(mem, targets) for mem in mem3_rec) / len(mem3_rec)

## Experiment 2

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

Final Test Accuracy : N/A

### Obsevations

Initial baseline runs yielded a training accuracy below 50%, indicating the model is not converging effectively. Two hyperparameters were identified as 
likely causes: the learning rate and the number of time steps (num_steps). These will be systematically adjusted in subsequent runs to isolate their individual impact on convergence.

## Experiment 3

In this experiment all parameters remain the same except for the learning rate for the model. This value
was increased 10x to observe the changes in model accuracy.

#### Hyperparameter Notes

| Parameter | Value        | Reasoning                                                          |
|-----------|--------------|--------------------------------------------------------------------|
|beta | 0.95         | High Membrane memory - good for sparse latency input (99.5% zeros) 
|num_steps| 100          | Initial Value                                                      |
|lr  | 1e-4 -> 1e-3 | LR was increased                                                   |
|batch_size| 128          | Initial Value                                                      |
|val_split| 0.2          | 80/20 train/val split, seed=42 for reproducibility                 |

### Epoch Results

|Epoch| Average Loss | Train Accuracy | Val Accuracy | Time   |
|-----|--------------|----------------|--------------|--------|
|1| 1.4785       | 46.27%         | 53.73%       | 187.0s |
|2| 1.1363       | 51.73%         | 53.04%       | 171.1s |
|3| 1.0416       | 54.20%         | 56.09%       | 180.4s |
|4| 0.9830       | 56.07%         | 56.60%       | 173.0s |
|5| 0.9393       | 56.80%         | 57.65%       | 170.8s |

Final Test Accuracy : 58.83%

### Observations

Run 1 (lr=1e-4):
Loss declined slowly from 2.06 → 1.36 over 5 epochs, with accuracy stagnating around 48–50% and showing no consistent upward trend. This is characteristic of a learning rate that is too low — the optimizer is taking steps too small to escape flat regions of the loss landscape.

Run 2 (lr=1e-3):
Loss declined more consistently from 1.47 → 0.94, with accuracy climbing steadily from 46% → 57% across all 5 epochs. Crucially, the model had not plateaued by epoch 5, suggesting further improvement is likely with more epochs.

While the experimental values are approaching the benchmark values, they are still low. For the next experiment,
the num_step parameter will be adjusted while retaining the current lr.

### Experiment 4

In this experiment, the num_steps parameter will be reduced from 100->50 in an attempt to reduce noise.

#### Hyperparameter Notes

| Parameter | Value        | Reasoning                                                         |
|-----------|--------------|-------------------------------------------------------------------|
|beta | 0.95         | High Membrane memory - good for sparse latency input (99.5% zeros) 
|num_steps| 100 -> 50    | Value decreased by 0.5                                            |
|lr  | 1e-3 | LR remains the same as experiment 3                               |
|batch_size| 128          | Initial Value                                                     |
|val_split| 0.2          | 80/20 train/val split, seed=42 for reproducibility                |

### Epoch Results

|Epoch| Average Loss | Train Accuracy | Val Accuracy | Time  |
|-----|--------------|----------------|--------------|-------|
|1| 1.2788       | 48.50%         | 54.08%       | 81.0s |
|2| 0.9507       | 55.62%         | 56.40%       | 83.4s |
|3| 0.8627       | 58.81%         | 60.65%       | 84.4s |
|4| 0.8088       | 61.99%         | 62.79%       | 83.1s |
|5| 0.7674       | 63.98%         | 64.14%       | 84.1s |

Final Test Accuracy : 65.12%

### Observations

Run 2 vs Run 3 (num_steps=100 vs num_steps=50, lr=1e-3)
Reducing num_steps from 100 to 50 produced the most significant improvement observed so far. Val accuracy jumped from 57.65% to 64.14%, training loss dropped more consistently, and training time was cut in half (~175s → ~83s per epoch). The model had still not plateaued by epoch 5, suggesting further gains are possible with more epochs. This confirms the hypothesis that 100 time steps was introducing excessive gradient noise during backpropagation through time (BPTT). With latency encoding producing 99.5% sparse inputs, the majority of time steps beyond 50 contribute minimal signal while adding noise to the gradient estimates.

Conclusion: num_steps=50 is a strong improvement over 100. Before extending epochs, num_steps=25 will be tested to determine whether further reduction continues to help or whether 50 is the optimal sweet spot.

### Experiment 5

In this experiment, the num_steps parameter will be reduced from 100->50 in an attempt to reduce noise.

#### Hyperparameter Notes

| Parameter | Value        | Reasoning                                                         |
|-----------|--------------|-------------------------------------------------------------------|
|beta | 0.95         | High Membrane memory - good for sparse latency input (99.5% zeros) 
|num_steps| 100 -> 50    | Value decreased by 0.5                                            |
|lr  | 1e-3 | LR remains the same as experiment 3                               |
|batch_size| 128          | Initial Value                                                     |
|val_split| 0.2          | 80/20 train/val split, seed=42 for reproducibility                |

### Epoch Results

|Epoch| Average Loss | Train Accuracy | Val Accuracy | Time  |
|-----|--------------|----------------|--------------|-------|
|1| 1.2788       | 48.50%         | 54.08%       | 81.0s |
|2| 0.9507       | 55.62%         | 56.40%       | 83.4s |
|3| 0.8627       | 58.81%         | 60.65%       | 84.4s |
|4| 0.8088       | 61.99%         | 62.79%       | 83.1s |
|5| 0.7674       | 63.98%         | 64.14%       | 84.1s |

Final Test Accuracy : 65.12%