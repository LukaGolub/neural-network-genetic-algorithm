# Neural Network Regression via Genetic Algorithm

A from-scratch Python implementation of a small feedforward neural network **trained not with backpropagation, but with a genetic algorithm**, for regression tasks on CSV-style datasets.

## Description

This project builds a simple feedforward neural network (with a choice of three fixed hidden-layer architectures) and optimizes its weights and biases using an **evolutionary algorithm** instead of gradient descent. Each individual in the population is a flattened vector of all the network's weights and biases; fitness is the inverse of the mean squared error (MSE) that vector produces on the training set. The algorithm evolves the population over a fixed number of generations using elitism, fitness-proportionate (roulette-wheel) selection, arithmetic crossover, and Gaussian mutation, then evaluates the best final individual on a held-out test set.

## Features

- **Configurable network architecture** — choose between three built-in topologies: a single hidden layer of 5 or 20 neurons (`5s`, `20s`), or two hidden layers of 5 neurons each (`5s5s`)
- **Sigmoid activation** on hidden layers, linear (identity) output layer — suited to regression
- **Weight/bias flattening and restoring** — the whole network's parameters can be packed into a single vector (a genetic algorithm "chromosome") and restored, layer by layer
- **Genetic algorithm optimizer**:
  - Configurable population size and number of iterations
  - **Elitism** — the top N individuals are carried over unchanged each generation
  - **Roulette-wheel selection** proportional to fitness (`1 / (MSE + ε)`)
  - **Arithmetic crossover** — offspring is the average of two selected parents
  - **Gaussian mutation** — each gene mutates with probability `p`, adding noise drawn from `N(0, K)`
- **Periodic training error logging** every 2000 generations
- **Final test-set evaluation** using the best chromosome found

## Requirements

- Python 3.7+
- [NumPy](https://numpy.org/)

```bash
pip install numpy
```

## Usage

```bash
python nn_ga.py --train <train_file> --test <test_file> --nn {5s|20s|5s5s} \
                 --popsize <int> --elitism <int> --p <float> --K <float> --iter <int>
```

### Arguments

| Flag         | Description                                                        |
|--------------|----------------------------------------------------------------------|
| `--train`    | Path to the training data file                                     |
| `--test`     | Path to the test data file                                         |
| `--nn`       | Network architecture: `5s`, `20s`, or `5s5s`                       |
| `--popsize`  | Population size for the genetic algorithm                          |
| `--elitism`  | Number of top individuals carried over unchanged each generation   |
| `--p`        | Per-gene mutation probability                                      |
| `--K`        | Standard deviation of the Gaussian mutation noise                  |
| `--iter`     | Number of generations to run                                       |

### Example

```bash
python nn_ga.py --train train.csv --test test.csv --nn 5s5s \
                 --popsize 50 --elitism 2 --p 0.1 --K 0.5 --iter 10000
```

## Input File Format

Both the train and test files use a simple CSV-like format of numeric values:

```
# Lines starting with # are comments and are ignored
x1,x2,x3,y
1.0,2.0,0.5,3.1
0.4,1.1,2.2,1.8
```

- **First non-comment line**: header row (used only to determine column count; feature names are not otherwise used)
- **Remaining lines**: comma-separated numeric values, where the **last column is the regression target** and all preceding columns are input features

## Output

```
[Train error @2000]: 0.041233
[Train error @4000]: 0.028910
...
[Test error]: 0.031500
```

- **`[Train error @N]`** — the best individual's training MSE, printed every 2000 generations
- **`[Test error]`** — the final MSE of the best-found network, evaluated on the held-out test set

## Algorithm Notes

- **Fitness** is defined as `1 / (MSE + 1e-8)` so that lower error yields higher fitness, with a small epsilon to avoid division by zero for a perfect fit.
- **Selection** picks a parent by drawing a uniform random number and walking the cumulative probability distribution built from normalized fitness values (roulette-wheel selection).
- **Crossover** is a simple average of two parent chromosomes rather than a single- or multi-point crossover.
- **Mutation** perturbs a random subset of genes (chosen independently per gene with probability `p`) by adding Gaussian noise with standard deviation `K`.
- **Elitism** guarantees the algorithm's best training error never gets worse from one generation to the next.

