import numpy as np

class Dense:
    def __init__(self,n_inputs,n_outputs):
        # He Initialization
        self.W = np.random.randn(n_inputs,n_outputs)*np.sqrt(2/n_inputs)
        self.b = np.zeros((1,n_outputs))

    def forward(self,X):
        # caching forward pass of prev layer for backprop
        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z

    def backward(self,dZ):
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ,axis=0,keepdims=True)
        dX = dZ @ self.W.T
        return dX

    def params(self):
        return [(self.W,self.dW), (self.b,self.db)]