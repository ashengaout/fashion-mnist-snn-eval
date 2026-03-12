# SNN Classifier — Experiment Log SNN Parameter Optimization
## Project Goal

**Research Questions**

1. Can artificial latency encoding preserve sufficient spatial structure from static images to enable competitive classification in a Spiking Neural Network?
2. What is the accuracy cost of deploying a neuromorphic model (FC-LIF SNN) on inherently non-temporal data compared to classical ML baselines (SVM)?
3. At what point does the biological plausibility and theoretical energy efficiency of an SNN justify its accuracy trade-off against classical and hybrid approaches?
4. Does a hybrid CNN-SNN architecture recover the accuracy gap introduced by artificial temporal encoding, and at what computational cost?

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
## Experiment 8

This experiment keeps the epoch values the same from the last run and instead implements a scheduler to see if it affects the accuracy rate positively.

**Current scheduler parameters:**
mode='max',
factor=0.5,
patience=2,
min_lr=1e-5

### Hyperparameter Notes
| Parameter | Value | Reasoning                       |
|-----------|-------|---------------------------------|
|beta|0.95| inital                          |
|num_steps|25| same as experiment 5            |
|lr|1e-3 → scheduler| changed to opimitze training LR |
|batch_size|128| inital                          |
|epochs|15| same as experiment 7            |

### Epoch Results
| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |
|-------|--------------|----------------|--------------|-----|------|
| 1 | 1.1419 | 53.44% | 57.46% | 1.00e-03 | 38.3s |
| 2 | 0.8403 | 60.59% | 62.87% | 1.00e-03 | 38.5s |
| 3 | 0.7639 | 64.41% | 65.03% | 1.00e-03 | 37.8s |
| 4 | 0.7154 | 66.67% | 66.78% | 1.00e-03 | 38.4s |
| 5 | 0.6816 | 68.94% | 67.62% | 1.00e-03 | 38.3s |
| 6 | 0.6539 | 70.56% | 71.44% | 1.00e-03 | 38.1s |
| 7 | 0.6284 | 72.31% | 70.88% | 1.00e-03 | 38.3s |
| 8 | 0.6081 | 72.86% | 72.57% | 1.00e-03 | 38.3s |
| 9 | 0.5903 | 73.34% | 71.67% | 1.00e-03 | 38.2s |
| 10 | 0.5750 | 73.79% | 73.22% | 1.00e-03 | 38.3s |
| 11 | 0.5605 | 74.29% | 72.38% | 1.00e-03 | 37.7s |
| 12 | 0.5466 | 74.46% | 72.40% | 1.00e-03 | 38.2s |
| 13 | 0.5339 | 74.75% | 72.71% | 5.00e-04 | 38.0s |
| 14 | 0.5141 | 75.76% | 73.91% | 5.00e-04 | 38.5s |
| 15 | 0.5072 | 76.00% | 72.48% | 5.00e-04 | 38.9s |

Final Test Accuracy: 73.67%


### Observations

There still appears to be some oscillation on the val accuracy, while the train accuracy is improving. This could be early signs of overfitting, however,
to determine if the scheduler is making any significant improvements the epoch number will have to be increased. The LR is not approaching the baseline parameter value so it is possible that increasing
epochs may still increase the accuracy of this model.


## Experiment 9

### Hyperparameter Notes
| Parameter | Value     | Reasoning                                        |
|-----------|-----------|--------------------------------------------------|
|beta| 0.95      | intital value                                    |
|num_steps| 25        | same as experiment 5                             |
|lr| scheduler | Same as experiment 8                             |
|batch_size| 128       | inital value                                     |
|epochs| 15 -> 25  | Observe if scheduler is improving model accuracy |

### Epoch Results
| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |
|-------|--------------|----------------|--------------|-----|------|
| 1 | 1.1405 | 51.47% | 56.03% | 1.00e-03 | 37.5s |
| 2 | 0.8424 | 57.84% | 58.84% | 1.00e-03 | 37.9s |
| 3 | 0.7681 | 60.20% | 60.52% | 1.00e-03 | 38.4s |
| 4 | 0.7178 | 62.38% | 62.79% | 1.00e-03 | 37.7s |
| 5 | 0.6830 | 65.51% | 64.24% | 1.00e-03 | 38.1s |
| 6 | 0.6549 | 66.98% | 66.17% | 1.00e-03 | 38.5s |
| 7 | 0.6298 | 69.69% | 70.91% | 1.00e-03 | 38.1s |
| 8 | 0.6071 | 70.24% | 71.43% | 1.00e-03 | 38.1s |
| 9 | 0.5888 | 71.43% | 70.17% | 1.00e-03 | 37.9s |
| 10 | 0.5730 | 72.24% | 70.65% | 1.00e-03 | 37.9s |
| 11 | 0.5573 | 73.06% | 71.87% | 1.00e-03 | 37.5s |
| 12 | 0.5430 | 73.50% | 69.58% | 1.00e-03 | 37.6s |
| 13 | 0.5309 | 73.80% | 72.47% | 1.00e-03 | 38.4s |
| 14 | 0.5195 | 74.30% | 70.67% | 1.00e-03 | 38.2s |
| 15 | 0.5080 | 74.44% | 74.08% | 1.00e-03 | 37.8s |
| 16 | 0.4970 | 75.76% | 71.97% | 1.00e-03 | 37.8s |
| 17 | 0.4885 | 75.81% | 72.44% | 1.00e-03 | 38.0s |
| 18 | 0.4784 | 76.44% | 72.84% | 5.00e-04 | 38.0s |
| 19 | 0.4588 | 76.78% | 74.94% | 5.00e-04 | 37.9s |
| 20 | 0.4527 | 77.48% | 74.11% | 5.00e-04 | 37.4s |
| 21 | 0.4475 | 78.03% | 74.97% | 5.00e-04 | 38.0s |
| 22 | 0.4420 | 78.56% | 73.54% | 5.00e-04 | 38.1s |
| 23 | 0.4374 | 78.46% | 73.18% | 5.00e-04 | 37.6s |
| 24 | 0.4325 | 78.85% | 74.67% | 2.50e-04 | 37.6s |
| 25 | 0.4222 | 79.24% | 74.53% | 2.50e-04 | 38.0s |

Final Test Accuracy: 75.56%


### Observations

Epochs matter most right now. The model was still learning at epoch 15 in every experiment. Extending to 25 epochs gave the biggest single accuracy jump (+1.1% test accuracy over exp 7).
Scheduler timing is critical. Exp 8 fired the LR reduction at epoch 13 — too early, hurt performance. Exp 9 fired at epoch 18 — better, and contributed to the best result. The fix is patience=5 to stop it reacting to val noise.
Overfitting is emerging but not critical yet. The train/val gap grew from ~0.4% at epoch 15 to ~4.7% at epoch 25. It's worth monitoring but dropout should wait until the scheduler is fixed first.
Val accuracy is noisy. Oscillations of ±2–3% between epochs are consistent across all experiments. This is partly why patience=2 was too aggressive — the scheduler was reacting to noise rather than real plateaus. 

## Experiment 10

This experiment increases the patience levels for the SNN. 

**Current Scheduler Parameters:**
mode='max',
factor=0.5,
patience=5,
min_lr=1e-5

### Hyperparameter Notes
| Parameter | Value | Reasoning            |
|-----------|-------|----------------------|
|beta|0.95| inital value         |
|num_steps|25| same as experiment 5 |
|lr|scheduler| same as experiment 8 |
|batch_size|128| inital value         |
|epochs|25| same as experiment 9 |

### Epoch Results
| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |
|-------|--------------|----------------|--------------|-----|------|
| 1 | 1.1315 | 53.20% | 58.52% | 1.00e-03 | 44.2s |
| 2 | 0.8371 | 58.93% | 61.84% | 1.00e-03 | 45.2s |
| 3 | 0.7578 | 61.84% | 63.80% | 1.00e-03 | 42.5s |
| 4 | 0.7104 | 64.78% | 65.70% | 1.00e-03 | 43.3s |
| 5 | 0.6746 | 67.70% | 70.14% | 1.00e-03 | 42.6s |
| 6 | 0.6483 | 68.64% | 67.51% | 1.00e-03 | 41.7s |
| 7 | 0.6213 | 70.13% | 69.59% | 1.00e-03 | 41.9s |
| 8 | 0.6023 | 71.53% | 71.63% | 1.00e-03 | 41.9s |
| 9 | 0.5851 | 72.44% | 71.70% | 1.00e-03 | 41.3s |
| 10 | 0.5678 | 73.42% | 72.47% | 1.00e-03 | 41.8s |
| 11 | 0.5518 | 73.86% | 71.75% | 1.00e-03 | 41.9s |
| 12 | 0.5399 | 74.35% | 72.78% | 1.00e-03 | 41.9s |
| 13 | 0.5275 | 74.78% | 72.25% | 1.00e-03 | 41.6s |
| 14 | 0.5165 | 75.13% | 71.59% | 1.00e-03 | 41.6s |
| 15 | 0.5048 | 75.94% | 72.30% | 1.00e-03 | 41.9s |
| 16 | 0.4951 | 75.64% | 74.89% | 1.00e-03 | 42.2s |
| 17 | 0.4848 | 76.22% | 72.34% | 1.00e-03 | 41.9s |
| 18 | 0.4746 | 76.52% | 74.01% | 1.00e-03 | 41.1s |
| 19 | 0.4667 | 76.74% | 75.54% | 1.00e-03 | 41.4s |
| 20 | 0.4557 | 77.56% | 74.06% | 1.00e-03 | 42.0s |
| 21 | 0.4468 | 77.88% | 74.49% | 1.00e-03 | 41.3s |
| 22 | 0.4385 | 78.01% | 73.46% | 1.00e-03 | 41.4s |
| 23 | 0.4304 | 78.45% | 71.95% | 1.00e-03 | 41.4s |
| 24 | 0.4227 | 78.78% | 74.53% | 1.00e-03 | 41.0s |
| 25 | 0.4141 | 78.94% | 75.63% | 1.00e-03 | 42.1s |

Final Test Accuracy: 76.88%


### Observations

Patience fix was the right call. Changing patience from 2 to 5 prevented premature LR reduction and delivered the best result yet (+1.32% over exp 9). The scheduler didn't fire at all in exp 10, suggesting the model benefits most from a stable LR across 25 epochs.The model is still learning. Loss was still declining at epoch 25 (0.4141) with no sign of flattening. This is the clearest signal yet that extending to 30 epochs is the right next move.Overfitting is stable, not worsening. Unlike exp 9 where the train/val gap grew to 4.7%, exp 10 stabilized at ~3.3% gap throughout the later epochs. This is a healthier pattern and means dropout is not yet urgent.Val noise is structural. The ±2–3% oscillation in val accuracy has appeared in every experiment regardless of scheduler settings. It's not a training artifact — likely a combination of val set size and the model sitting near a generalization boundary. Something to address eventually with dropout or larger val set.


## Experiment 8

### Hyperparameter Notes
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
|beta|0.95|  |
|num_steps|25|  |
|lr|scheduler|  |
|batch_size|128|  |
|epochs|30|  |

### Epoch Results
| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |
|-------|--------------|----------------|--------------|-----|------|
| 1 | 1.1418 | 54.38% | 60.09% | 1.00e-03 | 37.7s |
| 2 | 0.8411 | 63.58% | 65.74% | 1.00e-03 | 38.1s |
| 3 | 0.7620 | 69.66% | 72.66% | 1.00e-03 | 38.6s |
| 4 | 0.7142 | 73.71% | 71.67% | 1.00e-03 | 37.9s |
| 5 | 0.6792 | 74.98% | 75.24% | 1.00e-03 | 38.3s |
| 6 | 0.6499 | 77.14% | 77.72% | 1.00e-03 | 37.9s |
| 7 | 0.6247 | 78.06% | 78.98% | 1.00e-03 | 38.6s |
| 8 | 0.6034 | 79.04% | 77.77% | 1.00e-03 | 38.0s |
| 9 | 0.5867 | 78.93% | 75.60% | 1.00e-03 | 38.1s |
| 10 | 0.5718 | 79.25% | 76.19% | 1.00e-03 | 38.4s |
| 11 | 0.5556 | 79.36% | 78.26% | 1.00e-03 | 37.8s |
| 12 | 0.5415 | 79.72% | 77.30% | 1.00e-03 | 38.1s |
| 13 | 0.5305 | 79.69% | 76.80% | 5.00e-04 | 43.6s |
| 14 | 0.5106 | 80.21% | 77.67% | 5.00e-04 | 42.4s |
| 15 | 0.5040 | 80.69% | 76.97% | 5.00e-04 | 43.5s |
| 16 | 0.4973 | 80.74% | 77.38% | 5.00e-04 | 42.7s |
| 17 | 0.4921 | 80.48% | 80.09% | 5.00e-04 | 44.8s |
| 18 | 0.4862 | 81.28% | 77.59% | 5.00e-04 | 42.1s |
| 19 | 0.4812 | 81.67% | 78.82% | 5.00e-04 | 41.0s |
| 20 | 0.4749 | 81.45% | 77.33% | 5.00e-04 | 42.1s |
| 21 | 0.4691 | 81.77% | 78.57% | 5.00e-04 | 43.7s |
| 22 | 0.4647 | 82.25% | 78.52% | 5.00e-04 | 42.6s |
| 23 | 0.4591 | 82.45% | 77.93% | 2.50e-04 | 41.9s |
| 24 | 0.4490 | 82.74% | 78.88% | 2.50e-04 | 41.9s |
| 25 | 0.4453 | 83.03% | 79.56% | 2.50e-04 | 46.0s |
| 26 | 0.4433 | 83.28% | 77.95% | 2.50e-04 | 44.7s |
| 27 | 0.4399 | 83.20% | 78.88% | 2.50e-04 | 43.5s |
| 28 | 0.4374 | 83.46% | 79.67% | 2.50e-04 | 44.2s |
| 29 | 0.4349 | 83.75% | 80.06% | 1.25e-04 | 44.1s |
| 30 | 0.4295 | 83.96% | 78.81% | 1.25e-04 | 44.5s |

Final Test Accuracy: 79.83%


### Observations

Experiment 11 was the biggest single jump yet (+2.95%). The combination of 30 epochs and a well-timed scheduler firing three times (epochs 13, 23, 29) produced the best result so far. Each LR reduction gave the model a productive fine-tuning phase on top of stable early learning — this is the scheduler finally working as intended.Patience=5 validated. The scheduler fired at the right times across all three reductions, suggesting patience=5 is correctly calibrated for this model's val noise pattern. No need to revisit.30 epochs is the sweet spot. Loss was still declining at epoch 30 but the rate of improvement was slowing. Combined with the widening train/val gap, extending further without regularization would likely hurt more than help.Overfitting is now the primary bottleneck. The train/val gap has grown consistently across the last three experiments and reached 4.4% by epoch 30. Val accuracy is also oscillating ±2% in the later epochs. This is the clearest signal yet that dropout is the right next intervention.80% val accuracy was touched. Epoch 17 hit 80.09% and epoch 29 hit 80.06% — the model is capable of crossing that threshold, dropout may stabilize it there.


## Experiment 12

A bug was discovered in the model architecture, the neurons were improperly set in the forward passes. 

#WRONG (current)
mem3 = self.fc3(spk2)
spk3, mem3 = self.lif3(mem3, mem3)

#CORRECT
cur3 = self.fc3(spk2)
spk3, mem3 = self.lif3(cur3, mem3)

All three lif neurons were set incorrectly, so it was fixed in this experiment.

### Hyperparameter Notes
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
|beta|0.95|  |
|num_steps|25|  |
|lr|scheduler|  |
|batch_size|128|  |
|epochs|30|  |

### Epoch Results
| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |
|-------|--------------|----------------|--------------|-----|------|
| 1 | 1.0876 | 66.51% | 74.67% | 1.00e-03 | 38.0s |
| 2 | 0.8294 | 75.82% | 76.08% | 1.00e-03 | 38.4s |
| 3 | 0.7637 | 77.92% | 77.65% | 1.00e-03 | 38.2s |
| 4 | 0.7173 | 79.94% | 79.33% | 1.00e-03 | 37.9s |
| 5 | 0.6895 | 81.42% | 79.63% | 1.00e-03 | 38.6s |
| 6 | 0.6606 | 82.20% | 81.12% | 1.00e-03 | 38.3s |
| 7 | 0.6402 | 83.38% | 80.89% | 1.00e-03 | 38.6s |
| 8 | 0.6176 | 83.66% | 82.93% | 1.00e-03 | 38.2s |
| 9 | 0.6009 | 84.57% | 82.39% | 1.00e-03 | 37.6s |
| 10 | 0.5821 | 85.05% | 82.62% | 1.00e-03 | 38.1s |
| 11 | 0.5649 | 85.53% | 83.43% | 1.00e-03 | 38.0s |
| 12 | 0.5485 | 86.32% | 83.27% | 1.00e-03 | 38.0s |
| 13 | 0.5320 | 87.05% | 83.50% | 1.00e-03 | 37.9s |
| 14 | 0.5152 | 87.68% | 84.38% | 1.00e-03 | 37.3s |
| 15 | 0.4990 | 88.38% | 83.23% | 1.00e-03 | 38.1s |
| 16 | 0.4815 | 88.70% | 83.68% | 1.00e-03 | 38.1s |
| 17 | 0.4667 | 89.14% | 83.94% | 1.00e-03 | 37.8s |
| 18 | 0.4529 | 89.89% | 84.26% | 1.00e-03 | 37.7s |
| 19 | 0.4394 | 90.58% | 84.04% | 1.00e-03 | 37.8s |
| 20 | 0.4209 | 91.02% | 83.87% | 5.00e-04 | 38.0s |
| 21 | 0.3841 | 92.20% | 84.40% | 5.00e-04 | 37.9s |
| 22 | 0.3727 | 92.74% | 84.20% | 5.00e-04 | 37.9s |
| 23 | 0.3655 | 93.04% | 84.29% | 5.00e-04 | 37.7s |
| 24 | 0.3560 | 93.43% | 84.21% | 5.00e-04 | 37.7s |
| 25 | 0.3485 | 93.69% | 84.63% | 5.00e-04 | 37.8s |
| 26 | 0.3411 | 93.90% | 84.32% | 5.00e-04 | 37.8s |
| 27 | 0.3326 | 94.12% | 84.35% | 5.00e-04 | 37.7s |
| 28 | 0.3253 | 94.37% | 84.58% | 5.00e-04 | 37.6s |
| 29 | 0.3192 | 94.52% | 84.33% | 5.00e-04 | 37.6s |
| 30 | 0.3142 | 94.73% | 83.70% | 5.00e-04 | 37.6s |

Final Test Accuracy: 84.08%


### Observations

The bug fix was the single biggest improvement of the entire project (+4.25%). All experiments 1–11 were running with lif2 and lif3 being called instead of lif1 and lif2 respectively, and the output layer was passing membrane state as both input and state to lif3. The model was effectively operating as a degraded architecture the entire time — it is remarkable that 79.83% was achieved under these conditions.
The corrected architecture is significantly more powerful. With the proper 784→512→256→10 FC-SNN running as intended, the model reached 84.08% test accuracy — well inside territory previously expected only from hybrid CNN-SNN approaches.
Overfitting is now the primary and urgent bottleneck. The train/val gap exploded to 11% by epoch 30, with val accuracy plateauing around 84% from epoch 21 onward while train accuracy raced to 94.73%. The model has more capacity than the data can generalize without regularization.
Val accuracy plateaued at epoch 21. Epochs 21–30 produced no meaningful val improvement (84.40% → 83.70%), meaning the last 9 epochs were purely overfitting. Reducing back to 25 epochs is justified, however this will be done after adding the dropout values.


## Experiment 13

After fixing the bug, the next step was to add a dropout (val=0.2) to the model to see if the model will continue improving in val accuracy even after 20 epochs.

### Hyperparameter Notes
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
|beta|0.95|  |
|num_steps|25|  |
|lr|scheduler|  |
|batch_size|128|  |
|epochs|30|  |

### Epoch Results
| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |
|-------|--------------|----------------|--------------|-----|------|
| 1 | 1.1095 | 72.42% | 78.73% | 1.00e-03 | 49.9s |
| 2 | 0.8406 | 77.98% | 78.81% | 1.00e-03 | 49.1s |
| 3 | 0.7685 | 78.88% | 79.44% | 1.00e-03 | 49.5s |
| 4 | 0.7292 | 80.23% | 78.24% | 1.00e-03 | 48.5s |
| 5 | 0.7024 | 80.89% | 78.72% | 1.00e-03 | 49.4s |
| 6 | 0.6791 | 81.55% | 80.25% | 1.00e-03 | 49.0s |
| 7 | 0.6661 | 82.41% | 80.99% | 1.00e-03 | 48.6s |
| 8 | 0.6464 | 82.77% | 79.99% | 1.00e-03 | 49.7s |
| 9 | 0.6301 | 83.56% | 80.61% | 1.00e-03 | 49.6s |
| 10 | 0.6109 | 83.67% | 82.33% | 1.00e-03 | 48.6s |
| 11 | 0.6005 | 84.61% | 82.96% | 1.00e-03 | 48.8s |
| 12 | 0.5846 | 85.20% | 81.21% | 1.00e-03 | 48.6s |
| 13 | 0.5777 | 85.55% | 83.09% | 1.00e-03 | 48.9s |
| 14 | 0.5602 | 86.39% | 82.97% | 1.00e-03 | 48.8s |
| 15 | 0.5498 | 86.76% | 82.71% | 1.00e-03 | 49.2s |
| 16 | 0.5369 | 86.99% | 83.23% | 1.00e-03 | 49.2s |
| 17 | 0.5232 | 87.55% | 84.05% | 1.00e-03 | 49.1s |
| 18 | 0.5122 | 88.44% | 83.43% | 1.00e-03 | 49.2s |
| 19 | 0.5046 | 88.48% | 84.05% | 1.00e-03 | 49.0s |
| 20 | 0.4916 | 89.20% | 83.53% | 1.00e-03 | 49.1s |
| 21 | 0.4803 | 89.22% | 83.47% | 1.00e-03 | 48.7s |
| 22 | 0.4742 | 89.66% | 84.03% | 1.00e-03 | 49.0s |
| 23 | 0.4600 | 90.00% | 84.58% | 1.00e-03 | 49.1s |
| 24 | 0.4497 | 90.45% | 83.37% | 1.00e-03 | 49.0s |
| 25 | 0.4405 | 90.66% | 84.42% | 1.00e-03 | 48.2s |
| 26 | 0.4291 | 91.00% | 83.71% | 1.00e-03 | 48.6s |
| 27 | 0.4216 | 90.95% | 83.44% | 1.00e-03 | 48.8s |
| 28 | 0.4120 | 91.40% | 83.54% | 1.00e-03 | 48.6s |
| 29 | 0.4018 | 91.61% | 83.64% | 5.00e-04 | 48.1s |
| 30 | 0.3770 | 92.36% | 83.54% | 5.00e-04 | 48.7s |

Final Test Accuracy: 84.11%


### Observations

Dropout confirmed working but marginal (+0.03%). It achieved its structural goals — reduced train accuracy from 94.73% to 92.36% and narrowed the train/val gap from 11% to 8.8%. However it did not move the val accuracy ceiling, which has been stuck in the 83–84% band since epoch 17 of experiment 12.84% appears to be the true architecture ceiling. Val accuracy plateaued at ~84% from epoch 17 in experiment 12 and from epoch 17 in experiment 13 — across two different regularization conditions. This is a strong and consistent signal that the FC-LIF SNN architecture itself is the bottleneck, not overfitting or training duration.The train/val gap is improved but persistent. At 8.8% it is better than the 11% in experiment 12 but still significant. Further dropout tuning (0.3) is an option but unlikely to move the val ceiling — it would only reduce the gap without improving test accuracy meaningfully.Scheduler fired late and only once. With dropout slowing learning, the scheduler didn't fire until epoch 29 — too late to contribute a meaningful fine-tuning phase. This may warrant revisiting scheduler config if further experiments are run.

The model has exceeded the originally estimated pure SNN ceiling of 78–83% and appears to have found its true ceiling at approximately 84–85% test accuracy. Further FC-SNN tuning is subject to severe diminishing returns. The architecture has been pushed close to its limit with the current encoding scheme and layer structure.



## Experiment 14

### Hyperparameter Notes
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
|beta|0.95|  |
|num_steps|25|  |
|lr|scheduler|  |
|batch_size|128|  |
|epochs|30|  |

### Epoch Results
| Epoch | Average Loss | Train Accuracy | Val Accuracy | LR | Time |
|-------|--------------|----------------|--------------|-----|------|
| 1 | 1.1114 | 67.26% | 73.12% | 1.00e-03 | 61.4s |
| 2 | 0.8508 | 72.39% | 72.58% | 1.00e-03 | 61.3s |
| 3 | 0.7796 | 74.13% | 74.57% | 1.00e-03 | 59.6s |
| 4 | 0.7386 | 76.00% | 75.65% | 1.00e-03 | 59.1s |
| 5 | 0.7158 | 77.16% | 76.80% | 1.00e-03 | 57.9s |
| 6 | 0.6900 | 78.60% | 77.68% | 1.00e-03 | 55.5s |
| 7 | 0.6729 | 79.41% | 79.18% | 1.00e-03 | 58.0s |
| 8 | 0.6601 | 80.53% | 79.19% | 1.00e-03 | 57.9s |
| 9 | 0.6400 | 81.58% | 79.01% | 1.00e-03 | 59.7s |
| 10 | 0.6264 | 81.85% | 80.89% | 1.00e-03 | 58.1s |
| 11 | 0.6145 | 82.82% | 81.21% | 1.00e-03 | 59.4s |
| 12 | 0.6018 | 82.96% | 81.88% | 1.00e-03 | 56.5s |
| 13 | 0.5899 | 83.92% | 80.75% | 1.00e-03 | 57.2s |
| 14 | 0.5730 | 84.10% | 81.12% | 1.00e-03 | 57.0s |
| 15 | 0.5705 | 84.96% | 81.91% | 1.00e-03 | 55.9s |
| 16 | 0.5586 | 85.49% | 81.19% | 1.00e-03 | 56.0s |
| 17 | 0.5449 | 85.87% | 82.20% | 1.00e-03 | 55.8s |
| 18 | 0.5334 | 86.46% | 82.97% | 1.00e-03 | 55.2s |
| 19 | 0.5263 | 87.20% | 82.88% | 1.00e-03 | 55.5s |
| 20 | 0.5148 | 87.41% | 83.08% | 1.00e-03 | 56.0s |
| 21 | 0.5043 | 87.77% | 82.95% | 1.00e-03 | 55.4s |
| 22 | 0.4926 | 87.92% | 82.73% | 1.00e-03 | 55.6s |
| 23 | 0.4886 | 88.43% | 83.58% | 1.00e-03 | 55.7s |
| 24 | 0.4819 | 88.79% | 83.10% | 1.00e-03 | 55.3s |
| 25 | 0.4698 | 89.02% | 83.11% | 1.00e-03 | 55.1s |
| 26 | 0.4601 | 89.32% | 82.74% | 1.00e-03 | 55.3s |
| 27 | 0.4501 | 89.81% | 83.67% | 1.00e-03 | 56.1s |
| 28 | 0.4442 | 89.79% | 83.15% | 1.00e-03 | 54.9s |
| 29 | 0.4392 | 89.97% | 82.65% | 1.00e-03 | 54.8s |
| 30 | 0.4280 | 90.60% | 83.47% | 1.00e-03 | 54.9s |

Final Test Accuracy: 83.70%


### Observations

Dropout 0.3 over-regularized (-0.41%). While it achieved the best train/val gap of the entire project (7.1%), it suppressed learning capacity too aggressively, costing test accuracy. The scheduler also never fired across 30 epochs, meaning the model never received a fine-tuning phase at reduced LR — further limiting its ceiling.Dropout 0.2 is confirmed as the sweet spot. The three-experiment dropout comparison tells a clean story — no dropout left an 11% train/val gap, 0.2 balanced accuracy and regularization optimally, and 0.3 tipped too far into over-regularization.The pure SNN optimization phase is complete. Diminishing returns are clear and the architecture ceiling has been reached. Further tuning is not warranted.
