import numpy as np

from data_loader import load_train_data

from neural_network.layers import Dense
from neural_network.activations import ReLU, Softmax
from neural_network.losses import CategoricalCrossEntropy,one_hot
from neural_network.optimizers import SGD
from neural_network.model import Model

X_train, y_train, X_val, y_val = load_train_data('data/data.csv')

model = Model([
    Dense(784, 128), ReLU(),
    Dense(128, 64), ReLU(),
    Dense(64, 10), Softmax()
])

loss_fn = CategoricalCrossEntropy()
optimizer = SGD(learning_rate=0.1)

epochs = 100
batch_size = 64

for epoch in range(epochs):
    # reshuffle every epoch so batches differ each pass
    perm = np.random.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    for start in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[start:start + batch_size]
        y_batch = y_shuffled[start:start + batch_size]
        y_batch_oh = one_hot(y_batch, 10)

        predictions = model.forward(X_batch)
        grad = loss_fn.backward(predictions, y_batch_oh)
        model.backward(grad)

        params = [p for layer in model.dense_layers() for p in layer.params()]
        optimizer.update(params)

    # end-of-epoch validation
    val_preds = model.forward(X_val)
    val_loss = loss_fn.forward(val_preds, one_hot(y_val, 10))
    val_acc = np.mean(np.argmax(val_preds, axis=1) == y_val)
    print(f"Epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
