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

### Hyperparameter Notes

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

### Hyperparameter Notes

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

## Experiment 4

In this experiment, the num_steps parameter will be reduced from 100->50 in an attempt to reduce noise.

### Hyperparameter Notes

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

Run 3 vs Run 4 (num_steps=100 vs num_steps=50, lr=1e-3)

Reducing num_steps from 100 to 50 produced the most significant improvement observed so far. Val accuracy jumped from 57.65% to 64.14%, training loss dropped more consistently, and training time was cut in half (~175s → ~83s per epoch). The model had still not plateaued by epoch 5, suggesting further gains are possible with more epochs. This confirms the hypothesis that 100 time steps was introducing excessive gradient noise during backpropagation through time (BPTT). With latency encoding producing 99.5% sparse inputs, the majority of time steps beyond 50 contribute minimal signal while adding noise to the gradient estimates.

Conclusion: num_steps=50 is a strong improvement over 100. Before extending epochs, num_steps=25 will be tested to determine whether further reduction continues to help or whether 50 is the optimal sweet spot.

## Experiment 5

In this experiment, the num_steps parameter will be reduced from 50->25 in an attempt to reduce noise.

### Hyperparameter Notes

| Parameter | Value    | Reasoning                                                         |
|-----------|----------|-------------------------------------------------------------------|
|beta | 0.95     | High Membrane memory - good for sparse latency input (99.5% zeros) 
|num_steps| 50 -> 25 | Value decreased by 0.5                                            |
|lr  | 1e-3     | LR remains the same as experiment 3                               |
|batch_size| 128      | Initial Value                                                     |
|val_split| 0.2      | 80/20 train/val split, seed=42 for reproducibility                |

### Epoch Results

|Epoch| Average Loss | Train Accuracy | Val Accuracy | Time  |
|-----|------------|----------------|--------------|-------|
|1| 1.1362     | 52.13%         | 55.36%       | 42.2s |
|2| 0.8410     | 58.44%         | 61.33%       | 41.8s |
|3| 0.7670     | 62.34%         | 64.76%       | 44.0s |
|4| 0.7217     | 65.24%         | 66.38%       | 44.0s |
|5| 0.6865     | 67.02%         | 66.14%       | 44.1s |

Final Test Accuracy : 67.15%

### Observations

Run 4 vs Run 5 (num_steps=50 vs num_steps=25, lr=1e-3)

Reducing num_steps from 50 to 25 produced another meaningful improvement, with val accuracy climbing from 64.14% to 66.14% and training time halving again (~83s → ~43s per epoch). However, a subtle but important signal emerged — val accuracy dipped from 66.38% at epoch 4 to 66.14% at epoch 5, while train accuracy continued climbing to 67.02%. This is the first sign of slight overfitting, suggesting the model is beginning to fit the training set faster than it generalises at this step count.

Conclusion: num_steps=25 is the best configuration so far, but the early overfitting signal suggests this may be close to the lower limit for this encoding. Reducing further risks losing too much temporal information. The sweet spot appears to be between 25 and 50 steps. The next step is to extend training to 10 epochs at num_steps=25 and monitor val accuracy closely — the epoch where val accuracy peaks before declining will be the optimal stopping point and inform early stopping in future runs.

## Experiment 6

In this experiment, we will be increasing the epochs from 5->10 iterations. 

### Hyperparameter Notes

| Parameter | Value | Reasoning                                                          |
|-----------|-------|--------------------------------------------------------------------|
|beta | 0.95  | High Membrane memory - good for sparse latency input (99.5% zeros) 
|num_steps| 25    | Same as experiment 5                                               |
|lr  | 1e-3  | LR remains the same as experiment 3                                |
|batch_size| 128   | Initial Value                                                      |
|val_split| 0.2   | 80/20 train/val split, seed=42 for reproducibility                 |

### Epoch Results

| Epoch | Average Loss | Train Accuracy | Val Accuracy | Time  |
|-------|--------------|----------------|--------------|-------|
| 1     | 1.1413       | 55.17%         | 59.86%       | 41.8s |
| 2     | 0.8454       | 61.41%         | 63.48%       | 41.7s |
| 3     | 0.7655       | 65.67%         | 65.85%       | 41.5s |
| 4     | 0.7149       | 68.95%         | 70.12%       | 41.3s |
| 5     | 0.6797       | 71.98%         | 72.93%       | 41.7s |
| 6     | 0.6509       | 73.96%         | 72.16%       | 42.9s |
| 7     | 0.6278       | 75.11%         | 73.81%       | 42.1s |
| 8     | 0.6065       | 76.57%         | 73.86%       | 41.5s |
| 9     | 0.5889       | 77.39%         | 75.02%       | 47.1s |
| 10    | 0.5741       | 77.50%         | 77.84%       | 46.1s |

Final Test Accuracy : 78.85%

### Observations

Run 5 vs Run 6 (5 epochs vs 10 epochs, num_steps=25, lr=1e-3)

Extending training from 5 to 10 epochs produced the most significant accuracy gain of any run so far, with val accuracy climbing from 66.14% to 77.84% and test accuracy reaching 78.85%. Crucially, the model showed no signs of overfitting — train and val accuracy remained within 0.3% of each other at epoch 10, and val accuracy continued climbing consistently through all 10 epochs with no plateau or dip. Loss also declined steadily from 1.14 to 0.57 without flattening, strongly suggesting the model has not yet reached its performance ceiling.

Conclusion: The model is still learning at epoch 10 and has significant headroom remaining. The absence of overfitting and the consistent upward trend in val accuracy indicate that extending to 15 epochs is warranted before considering further hyperparameter changes. This will establish the true performance ceiling of the current architecture at num_steps=25, lr=1e-3 before moving on to the CNN comparison.

## Experiment 7

In this experiment, we will be increasing the epochs from 10->15 iterations. 

### Hyperparameter Notes

| Parameter | Value | Reasoning                                                          |
|-----------|-------|--------------------------------------------------------------------|
|beta | 0.95  | High Membrane memory - good for sparse latency input (99.5% zeros) 
|num_steps| 25    | Same as experiment 5                                               |
|lr  | 1e-3  | LR remains the same as experiment 3                                |
|batch_size| 128   | Initial Value                                                      |
|val_split| 0.2   | 80/20 train/val split, seed=42 for reproducibility                 |

### Epoch Results

| Epoch | Average Loss | Train Accuracy | Val Accuracy | Time  |
|-------|--------------|----------------|--------------|-------|
| 1     | 1.1382       | 53.34%         | 58.76%       | 41.4s |
| 2     | 0.8428       | 60.15%         | 62.01%       | 41.6s |
| 3     | 0.7660       | 64.54%         | 67.61%       | 41.5s |
| 4     | 0.7192       | 69.05%         | 69.86%       | 41.1s |
| 5     | 0.6827       | 71.35%         | 71.03%       | 41.9s |
| 6     | 0.6535       | 72.64%         | 72.17%       | 42.0s |
| 7     | 0.6296       | 73.97%         | 74.70%       | 41.6s |
| 8     | 0.6077       | 74.72%         | 74.17%       | 41.7s |
| 9     | 0.5900       | 75.54%         | 72.59%       | 41.5s |
| 10    | 0.5743       | 75.34%         | 75.48%       | 41.8s |
| 11    | 0.5599       | 76.06%         | 73.49%       | 41.2s |
| 12    | 0.5455       | 76.49%         | 75.51%       | 41.0s |
| 13    | 0.5321       | 76.81%         | 74.02%       | 41.9s |
| 14    | 0.5207       | 76.94%         | 75.19%       | 42.1s |
| 15    | 0.5099       | 77.14%         | 72.52%       | 42.2s |

Final Test Accuracy : 74.46%

### Observations

Run 5 vs Run 6 (10 epochs vs 15 epochs, num_steps=25, lr=1e-3)

Extending training from 10 to 15 epochs produced worse results, with final val accuracy dropping to 72.52% and test accuracy falling from 78.85% to 74.46%. Examining the per-epoch val accuracy reveals the cause — rather than continuing to climb, val accuracy began oscillating erratically after epoch 10, bouncing between ~72% and ~75% with no consistent upward trend. Train accuracy however continued climbing steadily, indicating the model was still learning but failing to generalise. This is a classic symptom of a learning rate that is too large for fine-grained optimisation in the later stages of training — the optimizer is overshooting the loss minimum and oscillating around it rather than converging.

Conclusion: The model's performance ceiling at lr=1e-3 is approximately 10 epochs, after which the learning rate becomes too large for further improvement. The 10 epoch run remains the best result with a test accuracy of 78.85%. To push beyond this, a learning rate scheduler will be introduced to automatically reduce the learning rate once val accuracy plateaus, allowing the optimizer to make finer adjustments in later epochs without manual intervention. ReduceLROnPlateau will be used as it responds dynamically to val accuracy rather than requiring a fixed schedule.
Additionally, it is unlikely that a scheduler alone will suffice in increasing the accuracy to a competitive benchmark. After an experiment to see how a learning rate scheduler will affect the SNN, the latency encoding will be replaced with rate encoding.