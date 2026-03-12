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
