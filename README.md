# Neural Network From Scratch

A neural network built from scratch using only NumPy — no ML frameworks — trained on MNIST digit classification.

## What's implemented

- Custom CSV data loader with shuffling, normalization, and train/val split
- Dense (fully-connected) layer with forward and backward pass
- ReLU and Softmax activations
- Categorical cross-entropy loss (with fused softmax gradient)
- SGD optimizer
- A `Model` class chaining it all together, trained with mini-batch gradient descent

Currently reaches ~97% validation accuracy on held-out MNIST digits.

## Project structure

```
NNFS/
├── main.py                  # training loop
├── data_loader.py           # loads and preprocesses MNIST CSV data
├── data/                    # MNIST CSV data (not tracked in git)
└── neural_network/
    ├── layers.py            # Dense layer
    ├── activations.py       # ReLU, Softmax
    ├── losses.py            # Cross-entropy loss + one-hot helper
    ├── optimizers.py        # SGD
    └── model.py             # Model class (forward/backward chaining)
```

## Data

Uses the Kaggle "Digit Recognizer" MNIST CSV format. Place `data.csv` (42,000 labeled rows: `label` + 784 `pixelN` columns) inside `data/`.

## Running

```
python main.py
```

## Future additions

- Momentum / Adam optimizer
- Learning rate scheduling
- Dropout / L2 regularization
- Gradient checking (verify backprop against numerical gradients)
- Confusion matrix and misclassified digit visualization
- Batch normalization
