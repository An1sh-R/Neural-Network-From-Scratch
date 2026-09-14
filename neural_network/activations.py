import numpy as np

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0,X)

    def backward(self, dA):
        return dA*(self.X > 0)

class Softmax:
    def forward(self, X):
        # shift to prevent huge values
        shifted = X - np.max(X,axis=1,keepdims=True)
        exp_values = np.exp(shifted)

        self.output = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        return self.output
